# GRPO and RLOO training, same settings for both.
# python train_rl.py --method grpo --max_steps 600
import argparse
import json

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from common import (MAX_COMPLETION_LEN, MAX_PROMPT_LEN, PROCESSED_DIR, add_common_args,
                    common_training_kwargs, drop_long_prompts, last_checkpoint, load_jsonl,
                    load_model, lora_config, save_final, setup_run)
from reward import execution_reward, valid_sql_reward


def load_train_set(only_mixed):
    rows = load_jsonl(f"{PROCESSED_DIR}/train.jsonl")
    if only_mixed:
        with open(f"{PROCESSED_DIR}/tags.json") as f:
            tags = json.load(f)
        rows = [r for r in rows if tags.get(r["qid"]) == "mixed"]
    print(f"training on {len(rows)} questions (only_mixed={bool(only_mixed)})")
    return Dataset.from_list([{k: r[k] for k in ("prompt", "qid", "db_path", "gold_sql")} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--method", required=True, choices=["grpo", "rloo"])
    ap.add_argument("--beta", type=float, default=0.04)  # GRPO default is 0 in TRL
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--only_mixed", type=int, default=1)
    ap.add_argument("--use_vllm", type=int, default=0)
    ap.add_argument("--init_adapter", default=None)
    args = ap.parse_args()

    run_name, out = setup_run(args.method, args)
    model, tok = load_model(args.model, bool(args.load_4bit))
    peft_config = lora_config(args.lora_rank)
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_config = None

    dataset = drop_long_prompts(load_train_set(args.only_mixed), tok)

    # one micro batch = 1 prompt x num_generations completions
    config_cls, trainer_cls = (GRPOConfig, GRPOTrainer) if args.method == "grpo" else (RLOOConfig, RLOOTrainer)
    config = config_cls(
        **common_training_kwargs(args, run_name, out),
        beta=args.beta,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.prompts_per_step,
        reward_weights=[1.0, 0.0],
        use_vllm=bool(args.use_vllm),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
    )
    trainer = trainer_cls(
        model=model,
        reward_funcs=[execution_reward, valid_sql_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint(out))
    save_final(trainer, out)


if __name__ == "__main__":
    main()
