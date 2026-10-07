# Greedy evaluation of one model on one split.
# python evaluate.py --split spider_dev --adapter outputs/grpo-1.5b-s0/final --tag grpo
import argparse
import json
import os
from collections import defaultdict

from common import PROCESSED_DIR, db_full_path, extract_sql, load_jsonl, save_jsonl
from generation import Generator
from reward import R_CORRECT, gold_rows_for, has_order_by, score_many
from monitor import Monitor


def one_line(sql):
    return " ".join(sql.split()) if sql else "SELECT 1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="spider_dev")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--notify", type=int, default=1)
    args = ap.parse_args()
    with Monitor(f"eval-{args.tag}-{args.split}", settings=vars(args), notify=bool(args.notify)) as mon:
        run(args, mon)


def run(args, mon):
    rows = load_jsonl(f"{PROCESSED_DIR}/{args.split}.jsonl")[:args.limit]
    gen = Generator(args.adapter, args.engine)
    outputs = [o[0] for o in gen.generate([r["prompt"] for r in rows], n=1, temperature=0.0)]
    preds = [extract_sql(o) for o in outputs]

    gold = gold_rows_for(rows)
    scores = score_many([(db_full_path(r["db_path"]), p, g, has_order_by(r["gold_sql"]))
                         for r, p, g in zip(rows, preds, gold)])

    per_q, by_diff = [], defaultdict(list)
    for r, out, p, (rew, ok) in zip(rows, outputs, preds, scores):
        correct = rew == R_CORRECT
        per_q.append({"qid": r["qid"], "db_id": r["db_id"], "question": r["question"],
                      "gold_sql": r["gold_sql"], "pred_sql": p, "raw_output": out,
                      "correct": correct, "valid": ok, "difficulty": r["difficulty"]})
        if r["difficulty"]:
            by_diff[r["difficulty"]].append(correct)

    metrics = {
        "split": args.split, "model": args.tag, "n": len(rows),
        "EX": sum(q["correct"] for q in per_q) / len(per_q),
        "valid_sql_rate": sum(q["valid"] for q in per_q) / len(per_q),
        "EX_by_difficulty": {d: sum(v) / len(v) for d, v in by_diff.items()},
    }
    print(json.dumps(metrics, indent=2))
    mon.result = metrics

    out = os.path.join(args.out_dir, args.tag, args.split)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_jsonl(per_q, f"{out}/per_question.jsonl")
    with open(f"{out}/pred.txt", "w") as f:
        f.write("\n".join(one_line(p) for p in preds) + "\n")
    with open(f"{out}/predict_dev.json", "w") as f:
        json.dump({str(i): f"{one_line(p)}\t----- bird -----\t{r['db_id']}"
                   for i, (r, p) in enumerate(zip(rows, preds))}, f, indent=2)


if __name__ == "__main__":
    main()
