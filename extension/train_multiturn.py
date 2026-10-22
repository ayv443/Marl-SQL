# Multi-turn RLOO / GRPO: training inside the multi-turn environment.
#
# One episode = one question. The model answers, the environment runs the SQL and (if it failed, or in
# --mode all always) sends back the error or the first rows, and the model can answer again, for up to
# --turns turns. The reward comes only at the end of the episode:
#     reward = score of the FINAL query (+1 correct, 0 wrong, -0.1 error)  -  turn_penalty * (times changed)
# so a query that is right the first time is worth a bit more than one fixed after feedback. Repeating
# the same query to confirm it does not cost anything.
#
# For each question K episodes are sampled. Advantage = RLOO leave-one-out baseline:
#     A_i = R_i - mean(R_j for j != i)
# with a per-sequence KL penalty to the starting policy subtracted from R (as in TRL's RLOO).
# --method grpo instead uses A_i = (R_i - group mean) / group std and adds the KL to the loss per token
# (k3 estimator), the same way TRL's GRPO trainer does in the single-turn runs.
# The policy gradient uses the log-probabilities of every token the model wrote in every turn of the
# episode; the prompt and the environment feedback are masked out. One update per batch (on-policy).
#
# Usage (from the repository root, T4 GPU):
#   python extension/train_multiturn.py --init_adapter <rloo checkpoint-600> --max_steps 150
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T2S = os.path.abspath(os.path.join(HERE, "..", "text2sql-rl"))
sys.path.insert(0, T2S)
sys.path.insert(0, HERE)

from common import (MAX_COMPLETION_LEN, MAX_PROMPT_LEN, MODEL_NAME, PROCESSED_DIR,  # noqa: E402
                    db_full_path, extract_sql, load_jsonl, lora_config)
from multiturn_eval import clean_sql, feedback, same_query  # noqa: E402
from reward import gold_cache, has_order_by, score  # noqa: E402


# ---------------------------------------------------------------- data

def load_questions(args, tok):
    rows = load_jsonl(os.path.join(PROCESSED_DIR, "train.jsonl"))
    tags = json.load(open(os.path.join(PROCESSED_DIR, "tags.json")))
    if args.only_mixed:
        rows = [r for r in rows if tags.get(r["qid"]) == "mixed"]
    elif args.skip_all_right:  # questions the untrained model always solved give almost no signal
        rows = [r for r in rows if tags.get(r["qid"]) != "all_right"]
    gold = gold_cache()
    keep, dropped = [], {"long prompt": 0, "missing database": 0, "missing or empty gold result": 0}
    for r in rows:
        if not os.path.exists(db_full_path(r["db_path"])):
            dropped["missing database"] += 1
        elif not gold.get(r["qid"]):
            dropped["missing or empty gold result"] += 1
        elif len(tok.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True)) > MAX_PROMPT_LEN:
            dropped["long prompt"] += 1
        else:
            keep.append(r)
    print(f"training questions: {len(keep)} (only_mixed={args.only_mixed}, skip_all_right={args.skip_all_right}) "
          f"dropped: {dropped}")
    return keep


# ---------------------------------------------------------------- model

def load_policy(args):
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    if args.init_adapter:
        # "default" is trained, "ref" is a frozen copy of the starting adapter for the KL term
        model = PeftModel.from_pretrained(base, args.init_adapter, adapter_name="default", is_trainable=True)
        model.load_adapter(args.init_adapter, adapter_name="ref", is_trainable=False)
    else:
        model = get_peft_model(base, lora_config(args.lora_rank))  # ref = base model (adapter disabled)
    model.set_adapter("default")
    set_trainable(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tok


def set_trainable(model):
    # only the "default" LoRA weights are trained (set_adapter can change requires_grad, so reset it)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n and ".default." in n


# ---------------------------------------------------------------- rollouts

def generate(model, tok, convs, args, temperature):
    import torch

    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    outs = [None] * len(texts)
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), args.gen_batch):
        idx = order[s:s + args.gen_batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_COMPLETION_LEN, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, i in enumerate(idx):
            outs[i] = dec[j]
    return outs


def rollout(model, tok, questions, args):
    """K episodes per question, all run in parallel turn by turn."""
    eps = []
    for q in questions:
        db = db_full_path(q["db_path"])
        for _ in range(args.num_generations):
            eps.append({"q": q, "db": db, "conv": list(q["prompt"]), "sqls": [], "done": False})
    for turn in range(1, args.turns + 1):
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        texts = generate(model, tok, [e["conv"] for e in live], args, args.temperature)
        for e, text in zip(live, texts):
            e["conv"].append({"role": "assistant", "content": text})
            sql = extract_sql(text)
            if e["sqls"]:  # turn 1 is read exactly as in the single-turn evaluation
                sql = clean_sql(sql)
            if sql is None and e["sqls"]:  # no new SQL after feedback = keep the previous query
                sql = e["sqls"][-1]
                e["no_sql"] = e.get("no_sql", 0) + 1
            repeated = bool(e["sqls"]) and same_query(sql, e["sqls"][-1])
            e["sqls"].append(sql)
            ran_ok, msg = feedback(sql, e["db"])
            stop = repeated or turn == args.turns or (args.mode == "errors" and ran_ok)
            if stop:
                e["done"] = True
            else:
                e["conv"].append({"role": "user", "content": msg})
    gold = gold_cache()
    for e in eps:
        q = e["q"]
        r_first = score(e["db"], e["sqls"][0], gold[q["qid"]], has_order_by(q["gold_sql"]))[0]
        r_final = score(e["db"], e["sqls"][-1], gold[q["qid"]], has_order_by(q["gold_sql"]))[0]
        e["r_first"], e["r_final"], e["turns"] = r_first, r_final, len(e["sqls"])
        # the turn cost is only paid when the model changes its query (repeating it to confirm is free)
        e["changes"] = sum(not same_query(a, b) for a, b in zip(e["sqls"], e["sqls"][1:]))
        e["reward"] = r_final - args.turn_penalty * e["changes"]
    return eps


# ---------------------------------------------------------------- log-probabilities

def tokenise_episode(tok, conv, n_prompt):
    """Token ids of the whole episode and a mask that is 1 only on tokens the model wrote."""
    ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False)
    mask = [0] * len(ids)
    for k in range(n_prompt, len(conv)):
        if conv[k]["role"] != "assistant":
            continue
        start = len(tok.apply_chat_template(conv[:k], tokenize=True, add_generation_prompt=True))
        end = len(tok.apply_chat_template(conv[:k + 1], tokenize=True, add_generation_prompt=False))
        for t in range(start, min(end, len(ids))):
            mask[t] = 1
    return ids, mask


def token_logps(model, ids, mask, device):
    """Log-probabilities of the masked tokens (only computes logits from the first answer onwards)."""
    import torch

    x = torch.tensor([ids], device=device)
    first = mask.index(1)
    keep = len(ids) - first + 1  # logits for positions first-1 ... end
    logits = model(input_ids=x, logits_to_keep=keep).logits[0, :-1]
    targets = x[0, first:]
    m = torch.tensor(mask[first:], device=device, dtype=torch.bool)
    logits = logits[m].float()  # only the model's own tokens (saves memory)
    return logits.gather(1, targets[m][:, None])[:, 0] - torch.logsumexp(logits, dim=-1)


def ref_logps(model, ids, mask, args):
    import torch

    with torch.no_grad(), torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
        if args.init_adapter:
            model.set_adapter("ref")
            lp = token_logps(model, ids, mask, args.device)
            model.set_adapter("default")
            set_trainable(model)
        else:
            with model.disable_adapter():
                lp = token_logps(model, ids, mask, args.device)
    return lp


# ---------------------------------------------------------------- training

def grpo_advantages(eps, k):
    # GRPO: (R - group mean) / group std, on the reward without KL (KL is a loss term instead)
    adv = []
    for g in range(0, len(eps), k):
        rs = [e["reward"] for e in eps[g:g + k]]
        mean = sum(rs) / k
        std = (sum((r - mean) ** 2 for r in rs) / (k - 1)) ** 0.5
        adv += [(r - mean) / (std + 1e-4) for r in rs]
    return adv


def rloo_advantages(eps, k):
    adv = []
    for g in range(0, len(eps), k):
        rs = [e["reward_kl"] for e in eps[g:g + k]]
        tot = sum(rs)
        adv += [r - (tot - r) / (k - 1) for r in rs]
    return adv


def train_step(model, tok, opt, scaler, questions, args):
    import torch

    t0 = time.time()
    eps = rollout(model, tok, questions, args)
    t_roll = time.time() - t0

    # KL to the starting policy, per sequence, subtracted from the reward (as in TRL RLOO)
    for e in eps:
        e["ids"], e["mask"] = tokenise_episode(tok, e["conv"], len(e["q"]["prompt"]))
        e["ids"], e["mask"] = e["ids"][:args.max_episode_len], e["mask"][:args.max_episode_len]
        e["ref"] = ref_logps(model, e["ids"], e["mask"], args) if 1 in e["mask"] else None
    stats = {"loss": 0.0, "kl": 0.0}
    opt.zero_grad(set_to_none=True)
    # first pass (no grad, no dropout, like the reference pass) for the KL value with the current policy
    model.eval()
    with torch.no_grad():
        for e in eps:
            if e["ref"] is None:
                e["kl"] = 0.0
                continue
            with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
                lp = token_logps(model, e["ids"], e["mask"], args.device)
            e["kl"] = float((lp - e["ref"]).sum())
    if args.method == "rloo":  # RLOO: KL subtracted from the reward, leave-one-out baseline
        for e in eps:
            e["reward_kl"] = e["reward"] - args.beta * e["kl"]
        adv = rloo_advantages(eps, args.num_generations)
    else:  # GRPO: normalised group advantage, KL added to the loss per token
        for e in eps:
            e["reward_kl"] = e["reward"]
        adv = grpo_advantages(eps, args.num_generations)

    # policy gradient: -A * mean log-prob of the episode's own tokens, averaged over episodes
    model.train()
    n_backward = 0
    for e, a in zip(eps, adv):
        kl_in_loss = args.method == "grpo" and args.beta > 0
        if e["ref"] is None or (a == 0 and not kl_in_loss):
            continue
        with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
            lp = token_logps(model, e["ids"], e["mask"], args.device)
            loss = -a * lp.mean()
            if kl_in_loss:  # k3 estimator of KL(policy || ref), per token, as in TRL GRPO
                d = e["ref"] - lp
                loss = loss + args.beta * (torch.exp(d) - d - 1).mean()
            loss = loss / len(eps)
        scaler.scale(loss).backward()
        stats["loss"] += loss.item()
        n_backward += 1
    if n_backward:  # (a batch with no learning signal at all is skipped instead of crashing the scaler)
        scaler.unscale_(opt)
        stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0))
        scaler.step(opt)
        scaler.update()

    n = len(eps)
    stats.update({
        "reward": sum(e["reward"] for e in eps) / n,
        "first_turn_correct": sum(e["r_first"] == 1.0 for e in eps) / n,
        "final_correct": sum(e["r_final"] == 1.0 for e in eps) / n,
        "final_error": sum(e["r_final"] < 0 for e in eps) / n,
        "fixed_after_feedback": sum(e["r_first"] != 1.0 and e["r_final"] == 1.0 for e in eps) / n,
        "mean_turns": sum(e["turns"] for e in eps) / n,
        "changed_query": sum(e["changes"] > 0 for e in eps) / n,
        "no_sql_reply": sum(e.get("no_sql", 0) > 0 for e in eps) / n,
        "broken_by_change": sum(e["r_first"] == 1.0 and e["r_final"] != 1.0 for e in eps) / n,
        "kl": sum(e["kl"] for e in eps) / n,
        "zero_adv_groups": sum(len({e["reward_kl"] for e in eps[g:g + args.num_generations]}) == 1
                               for g in range(0, n, args.num_generations)),
        "rollout_s": round(t_roll, 1),
        "step_s": round(time.time() - t0, 1),
    })
    return stats, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--init_adapter", default=None, help="start from this LoRA checkpoint (e.g. RLOO step 600)")
    ap.add_argument("--run_name", default="mt-rloo-1.5b-s0")
    ap.add_argument("--output_root", default=os.path.expanduser("~/outputs"))
    ap.add_argument("--method", default="rloo", choices=["rloo", "grpo"])
    ap.add_argument("--mode", default="all", choices=["errors", "all"])
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--turn_penalty", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--only_mixed", type=int, default=0)
    ap.add_argument("--skip_all_right", type=int, default=1)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--max_episode_len", type=int, default=3584)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb", type=int, default=1)
    ap.add_argument("--questions_file", default=None, help="jsonl of questions instead of train.jsonl (tests)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # print each line straight away, also under nohup

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = os.path.join(args.output_root, args.run_name)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "logs", "args.json"), "w"), indent=2)

    model, tok = load_policy(args)
    questions = load_jsonl(args.questions_file) if args.questions_file else load_questions(args, tok)
    random.shuffle(questions)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.03 * args.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / args.max_steps))
    scaler = torch.amp.GradScaler(enabled=args.device == "cuda")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "text2sql-rl"), name=args.run_name, config=vars(args))

    log = open(os.path.join(out, "logs", "steps.jsonl"), "a")
    samples = open(os.path.join(out, "logs", "samples.jsonl"), "a")
    pos = 0
    for step in range(1, args.max_steps + 1):
        if pos + args.prompts_per_step > len(questions):
            random.shuffle(questions)
            pos = 0
        batch = questions[pos:pos + args.prompts_per_step]
        pos += args.prompts_per_step
        stats, eps = train_step(model, tok, opt, scaler, batch, args)
        sched.step()
        stats.update({"step": step, "lr": sched.get_last_lr()[0]})
        print(json.dumps(stats))
        log.write(json.dumps(stats) + "\n")
        log.flush()
        if step <= 2 or step % 10 == 0:  # every episode of the first steps, then one every 10 steps
            for e in (eps if step <= 2 else eps[:1]):
                samples.write(json.dumps({"step": step, "qid": e["q"]["qid"], "reward": e["reward"],
                                          "sqls": e["sqls"], "gold_sql": e["q"]["gold_sql"],
                                          "conversation": e["conv"][len(e["q"]["prompt"]):]}) + "\n")
            samples.flush()
        if run:
            run.log(stats, step=step)
        if step % args.save_steps == 0 or step == args.max_steps:
            model.save_pretrained(os.path.join(out, f"checkpoint-{step}"), selected_adapters=["default"])
            print(f"saved checkpoint-{step}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
