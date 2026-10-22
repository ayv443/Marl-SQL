# Exact McNemar tests on the multi-turn evaluation results (no GPU needed).
#   python extension/multiturn_stats.py
# Reads extension/findings/multiturn/<tag>/<split>/all/per_question.jsonl
# Writes extension/findings/multiturn_statistics.md
import json
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "findings", "multiturn")
PAIRS = {"spider_dev": [("grpo", "mt-grpo"), ("rloo", "mt-rloo")],
         "val": [("grpo-400", "mt-grpo-150"), ("rloo-600", "mt-rloo-150")]}


def load(tag, split):
    path = os.path.join(F, tag, split, "all", "per_question.jsonl")
    return {r["qid"]: r["correct_per_turn"] for r in map(json.loads, open(path))}


def exact_p(b, c):
    """Two-sided exact McNemar test: binomial with p = 0.5 on the discordant pairs."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def discordant(a, b):
    return sum(a[q] and not b[q] for q in a), sum(b[q] and not a[q] for q in a)


def main():
    md = ["# Significance tests for the multi-turn evaluation", "",
          "Exact McNemar tests on paired per-question results (the same questions for both models). "
          "b = questions where only the first is true, c = only the second. Two-sided, p < 0.05 marked *.", ""]
    pooled = [0, 0]
    for split, pairs in PAIRS.items():
        md += [f"## {split}", "", "| Comparison | b | c | p |", "|---|---|---|---|"]
        for old, new in pairs:
            o, n = load(old, split), load(new, split)
            first = lambda d: {k: v[0] for k, v in d.items()}
            last = lambda d: {k: v[-1] for k, v in d.items()}
            broken = lambda d: {k: v[0] and not v[-1] for k, v in d.items()}
            rows = [(f"{old}: correct at turn 1 vs after feedback", first(o), last(o)),
                    (f"{new}: correct at turn 1 vs after feedback", first(n), last(n)),
                    (f"{old} vs {new}: correct at turn 1", first(o), first(n)),
                    (f"{old} vs {new}: correct after feedback", last(o), last(n)),
                    (f"{old} vs {new}: broken by feedback", broken(o), broken(n))]
            for name, a, b in rows:
                x, y = discordant(a, b)
                p = exact_p(x, y)
                md.append(f"| {name} | {x} | {y} | {p:.4f}{' *' if p < 0.05 else ''} |")
            if split == "spider_dev":
                x, y = discordant(broken(o), broken(n))
                pooled[0] += x
                pooled[1] += y
        md.append("")
    p = exact_p(*pooled)
    md += ["## Broken answers, both methods pooled (Spider dev)", "",
           f"Questions broken only by the single-turn model: {pooled[0]}, only by the multi-turn model: {pooled[1]}, "
           f"exact sign test p = {p:.4f}. (Pooling treats the GRPO and RLOO comparisons as one set of paired "
           "observations; it is a summary, not a separate experiment.)", ""]
    out = os.path.join(HERE, "findings", "multiturn_statistics.md")
    open(out, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
