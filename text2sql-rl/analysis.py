# python analysis.py compare --split spider_dev --tags base dpo grpo rloo
# python analysis.py efficiency --split bird_dev --tags base dpo grpo rloo
import argparse
import itertools
import random
import statistics
import time

from scipy.stats import binomtest

from common import db_full_path, load_jsonl
from reward import execute


def load(tag, split):
    return {q["qid"]: q for q in load_jsonl(f"results/{tag}/{split}/per_question.jsonl")}


def bootstrap_ci(correct, n_boot=10_000, seed=0):
    rng = random.Random(seed)
    n = len(correct)
    means = sorted(sum(rng.choices(correct, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def mcnemar(a, b):
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, binomtest(only_a, only_a + only_b, 0.5).pvalue


def compare(args):
    res = {t: load(t, args.split) for t in args.tags}
    qids = sorted(set.intersection(*(set(r) for r in res.values())))
    corr = {t: [res[t][q]["correct"] for q in qids] for t in args.tags}
    print(f"{args.split}: {len(qids)} questions\n")
    for t in args.tags:
        lo, hi = bootstrap_ci(corr[t])
        print(f"{t:>8}  EX = {sum(corr[t]) / len(qids):.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print("\nMcNemar test:")
    for a, b in itertools.combinations(args.tags, 2):
        oa, ob, p = mcnemar(corr[a], corr[b])
        print(f"{a:>8} vs {b:<8} only_A={oa:<4} only_B={ob:<4} p={p:.4f}{'  *' if p < 0.05 else ''}")


def timed(db, sql, repeats):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        execute(db, sql, timeout=30)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def efficiency(args):
    res = {t: load(t, args.split) for t in args.tags}
    qids = [q for q in res[args.tags[0]] if all(res[t].get(q, {}).get("correct") for t in args.tags)]
    print(f"questions all models got right: {len(qids)}")
    ratios = {t: [] for t in args.tags}
    for q in qids:
        ex = res[args.tags[0]][q]
        db = db_full_path(f"bird_dev/dev_databases/{ex['db_id']}/{ex['db_id']}.sqlite"
                          if args.split == "bird_dev" else f"spider_data/database/{ex['db_id']}/{ex['db_id']}.sqlite")
        gold_t = timed(db, ex["gold_sql"], args.repeats)
        for t in args.tags:
            ratios[t].append(gold_t / max(timed(db, res[t][q]["pred_sql"], args.repeats), 1e-6))
    for t in args.tags:
        print(f"{t:>8}  mean gold-time / model-time = {statistics.mean(ratios[t]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compare", "efficiency"])
    ap.add_argument("--split", default="spider_dev")
    ap.add_argument("--tags", nargs="+", default=["base", "dpo", "grpo", "rloo"])
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    compare(args) if args.mode == "compare" else efficiency(args)
