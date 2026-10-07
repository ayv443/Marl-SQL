import json
import os
import re
import sqlite3
from functools import lru_cache

# SageMaker mounts the S3 data channel at SM_CHANNEL_DATA
DATA_DIR = os.environ.get("SM_CHANNEL_DATA", os.path.join(os.path.dirname(__file__), "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # or Qwen/Qwen2.5-Coder-0.5B-Instruct
MAX_PROMPT_LEN = 2048
MAX_COMPLETION_LEN = 256

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a question, write ONE SQLite "
    "query that answers the question. Do not explain. Put only the SQL inside "
    "<answer></answer> tags."
)

ON_SAGEMAKER = "SM_MODEL_DIR" in os.environ


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def db_full_path(db_path):
    return os.path.join(DATA_DIR, db_path)


def _short(v, n=40):
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


@lru_cache(maxsize=None)
def schema_text(db_file):
    # CREATE TABLE statements with keys and 3 example values per column
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"database not found: {db_file}")
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    out = []
    for t in tables:
        cols = cur.execute(f'PRAGMA table_info("{t}")').fetchall()
        fks = cur.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()
        pks = [c[1] for c in cols if c[5] > 0]
        defs, comments = [], []
        for c in cols:
            name, ctype = c[1], c[2] or "TEXT"
            try:
                vals = [r[0] for r in cur.execute(
                    f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL LIMIT 3')]
            except sqlite3.Error:
                vals = []
            example = ", ".join(repr(_short(v)) if isinstance(v, str) else str(v) for v in vals)
            defs.append(f'"{name}" {ctype}')
            comments.append(f"  -- e.g. {example}" if example else "")
        if pks:
            defs.append("PRIMARY KEY (" + ", ".join(f'"{p}"' for p in pks) + ")")
            comments.append("")
        for fk in fks:
            ref_col = f'("{fk[4]}")' if fk[4] else ""
            defs.append(f'FOREIGN KEY ("{fk[3]}") REFERENCES "{fk[2]}"{ref_col}')
            comments.append("")
        lines = [f"  {d}{',' if i < len(defs) - 1 else ''}{cm}"
                 for i, (d, cm) in enumerate(zip(defs, comments))]
        out.append(f'CREATE TABLE "{t}" (\n' + "\n".join(lines) + "\n);")
    conn.close()
    return "\n\n".join(out)


def build_prompt(db_file, question, evidence=""):
    user = f"Database schema:\n{schema_text(db_file)}\n\n"
    if evidence:
        user += f"Hint: {evidence}\n\n"
    user += f"Question: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def completion_text(completion):
    # TRL gives either a string or a list of chat messages
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def extract_sql(text):
    # 1) inside <answer> tags, 2) inside a ``` block, 3) plain SQL starting a line
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S | re.I)
    if m is None:
        m = re.search(r"```(?:sql)?(.*?)```", text, re.S | re.I)
    if m is None:
        m = re.search(r"^\s*((?:SELECT|WITH)\b.*?)(?:;|\n\s*\n|\Z)", text, re.S | re.I | re.M)
    if m is None:
        return None
    sql = m.group(1).strip().rstrip(";").strip()
    return sql or None


def lora_config(rank=16):
    from peft import LoraConfig
    return LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def load_model(model_name=MODEL_NAME, load_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.float16}  # T4 has no bf16
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tok


def last_checkpoint(output_dir):
    from transformers.trainer_utils import get_last_checkpoint
    return get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None


def add_common_args(ap):
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--load_4bit", type=int, default=0)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--wandb_project", default="text2sql-rl")
    ap.add_argument("--notify_every", type=int, default=50)


def setup_run(method, args):
    size = "0.5b" if "0.5B" in args.model else "1.5b"
    name = f"{method}-{size}-s{args.seed}"
    out = args.output_dir or ("/opt/ml/checkpoints" if ON_SAGEMAKER else f"outputs/{name}")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ["WANDB_TAGS"] = f"{method},seed{args.seed},{size}"
    return name, out


def common_training_kwargs(args, run_name, output_dir):
    # same for DPO, GRPO and RLOO
    return dict(
        output_dir=output_dir, run_name=run_name, seed=args.seed,
        learning_rate=args.lr, max_steps=args.max_steps, warmup_ratio=0.03,
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_steps=args.save_steps, save_total_limit=None,
        report_to="wandb",
    )


def drop_long_prompts(dataset, tok, max_len=MAX_PROMPT_LEN):
    def fits(ex):
        ids = tok.apply_chat_template(ex["prompt"], tokenize=True, add_generation_prompt=True)
        return len(ids) <= max_len
    before = len(dataset)
    dataset = dataset.filter(fits)
    print(f"dropped {before - len(dataset)} examples with prompt > {max_len} tokens")
    return dataset


def save_final(trainer, output_dir):
    trainer.save_model(os.path.join(output_dir, "final"))
    if ON_SAGEMAKER:
        trainer.save_model(os.environ["SM_MODEL_DIR"])
