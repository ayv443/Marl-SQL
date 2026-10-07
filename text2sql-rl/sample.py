# Sample n attempts per question and score them.
# python sample.py --split train --n 8 --temperature 0.8
import argparse
import json
from math import comb

from common import PROCESSED_DIR, db_full_path, extract_sql, load_jsonl, save_jsonl
from generation import Generator
from reward import R_CORRECT, gold_rows_for, has_order_by, score_many


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    rows = load_jsonl(f"{PROCESSED_DIR}/{args.split}.jsonl")[:args.limit]
    gen = Generator(args.adapter, args.engine)
    outputs = gen.generate([r["prompt"] for r in rows], n=args.n, temperature=args.temperature)

    gold = gold_rows_for(rows)
    jobs = [(db_full_path(r["db_path"]), extract_sql(c), g, has_order_by(r["gold_sql"]))
            for r, outs, g in zip(rows, outputs, gold) for c in outs]
    scores = score_many(jobs)

    samples, tags = [], {}
    for i, (r, outs) in enumerate(zip(rows, outputs)):
        rewards = [s for s, _ in scores[i * args.n:(i + 1) * args.n]]
        n_correct = sum(x == R_CORRECT for x in rewards)
        tags[r["qid"]] = "all_right" if n_correct == args.n else "all_wrong" if n_correct == 0 else "mixed"
        samples.append({"qid": r["qid"], "completions": outs, "rewards": rewards, "n_correct": n_correct})

    ks = [k for k in (1, 4, 8, 16) if k <= args.n]
    summary = {f"pass@{k}": sum(pass_at_k(args.n, s["n_correct"], k) for s in samples) / len(samples) for k in ks}
    for t in ("all_wrong", "mixed", "all_right"):
        summary[f"share_{t}"] = sum(v == t for v in tags.values()) / len(tags)
    summary["valid_sql_rate"] = sum(ok for _, ok in scores) / len(scores)
    print(json.dumps(summary, indent=2))

    name = f"{args.split}" if args.tag == "base" else f"{args.split}_{args.tag}"
    save_jsonl(samples, f"{PROCESSED_DIR}/samples_{name}.jsonl")
    with open(f"{PROCESSED_DIR}/passk_{name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    if args.split == "train" and args.adapter is None and args.limit is None:
        with open(f"{PROCESSED_DIR}/tags.json", "w") as f:
            json.dump(tags, f)


if __name__ == "__main__":
    main()
