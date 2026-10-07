# Make DPO pairs from the sampled attempts: correct > wrong > error.
# If nothing was correct, the gold SQL is used as chosen.
import random

from common import PROCESSED_DIR, load_jsonl, save_jsonl
from reward import R_CORRECT, R_WRONG

MAX_PAIRS_PER_Q = 2


def main():
    rng = random.Random(0)
    train = {r["qid"]: r for r in load_jsonl(f"{PROCESSED_DIR}/train.jsonl")}
    samples = load_jsonl(f"{PROCESSED_DIR}/samples_train.jsonl")

    pairs, used_gold = [], 0
    for s in samples:
        if s["qid"] not in train:
            continue
        ex = train[s["qid"]]
        attempts = list(zip(s["completions"], s["rewards"]))
        correct = list(dict.fromkeys(c for c, r in attempts if r == R_CORRECT))
        wrong = list(dict.fromkeys(c for c, r in attempts if r == R_WRONG))
        error = list(dict.fromkeys(c for c, r in attempts if r < R_WRONG))
        rng.shuffle(correct), rng.shuffle(wrong), rng.shuffle(error)

        rejected = (wrong + error)[:MAX_PAIRS_PER_Q]
        if not rejected:
            continue
        if not correct:
            correct = [f"<answer>{ex['gold_sql']}</answer>"]
            used_gold += 1
        for i, rej in enumerate(rejected):
            pairs.append({
                "qid": ex["qid"],
                "prompt": ex["prompt"],
                "chosen": [{"role": "assistant", "content": correct[i % len(correct)]}],
                "rejected": [{"role": "assistant", "content": rej}],
            })

    save_jsonl(pairs, f"{PROCESSED_DIR}/dpo_pairs.jsonl")
    print(f"{len(pairs)} pairs from {len({p['qid'] for p in pairs})} questions "
          f"({used_gold} questions used the gold SQL as 'chosen')")


if __name__ == "__main__":
    main()
