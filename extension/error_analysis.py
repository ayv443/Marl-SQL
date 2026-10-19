# Error analysis of the saved evaluation predictions (no GPU needed).
# Which kinds of SQL mistakes does each model make, and which ones do GRPO and RLOO fix?
#   python extension/error_analysis.py
# Reads text2sql-rl/evaluation/results/<model>/<split>/per_question.jsonl
# Writes extension/findings/error_analysis.md, error_categories.csv and error_categories.png
import csv
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "text2sql-rl", "evaluation", "results")
OUT = os.path.join(HERE, "findings")

MODELS = ["base", "dpo", "grpo", "rloo"]
NAMES = {"base": "Untrained", "dpo": "DPO", "grpo": "GRPO", "rloo": "RLOO"}
SPLITS = {"spider_dev": "Spider dev", "spider_syn": "Spider-Syn", "spider_dk": "Spider-DK", "bird_dev": "BIRD dev"}

# error categories, checked in this order (the first one that applies is used)
CATEGORIES = [
    "Correct",
    "SQL error",              # the query fails to run, or no SQL was found
    "Wrong tables",           # uses a different set of tables than the gold query
    "Nesting / set operation",  # different number of sub-queries or INTERSECT / UNION / EXCEPT
    "Aggregation",            # different COUNT / SUM / AVG / MIN / MAX
    "Grouping / ordering",    # GROUP BY, HAVING, ORDER BY, LIMIT or DISTINCT differs
    "Columns / conditions",   # same structure, but other columns, values or conditions
]
AGGS = ["count", "sum", "avg", "min", "max"]
CLAUSES = ["group by", "having", "order by", "limit", "distinct"]


def load(model, split):
    p = os.path.join(RESULTS, model, split, "per_question.jsonl")
    with open(p) as f:
        return {r["qid"]: r for r in (json.loads(line) for line in f if line.strip())}


def normalise(sql):
    sql = sql or ""
    sql = re.sub(r"'[^']*'|\"[^\"]*\"", "'v'", sql)   # string values -> 'v' (keeps backtick names)
    return " ".join(sql.lower().split())


def tables(sql):
    # table names after FROM and JOIN; backtick / bracket quoted names may contain spaces
    found = re.findall(r"\b(?:from|join)\s+(`[^`]+`|\[[^\]]+\]|[a-z_][\w]*)", normalise(sql))
    return {t.strip("`[]") for t in found if t not in ("select", "(")}


def nesting(sql):
    s = normalise(sql)
    return s.count("select") + sum(s.count(op) for op in ("intersect", "union", "except"))


def aggregations(sql):
    s = normalise(sql)
    return {a for a in AGGS if re.search(rf"\b{a}\s*\(", s)}


def clauses(sql):
    s = normalise(sql)
    return {c for c in CLAUSES if c in s}


def category(r):
    if r["correct"]:
        return "Correct"
    if not r["valid"] or not r.get("pred_sql"):
        return "SQL error"
    pred, gold = r["pred_sql"], r["gold_sql"]
    if tables(pred) != tables(gold):
        return "Wrong tables"
    if nesting(pred) != nesting(gold):
        return "Nesting / set operation"
    if aggregations(pred) != aggregations(gold):
        return "Aggregation"
    if clauses(pred) != clauses(gold):
        return "Grouping / ordering"
    return "Columns / conditions"


def pct(n, total):
    return f"{100 * n / total:.1f}" if total else "n/a"


def main():
    os.makedirs(OUT, exist_ok=True)
    md = ["# Error analysis of the saved predictions", "",
          "Every prediction from the final evaluation (greedy decoding, best checkpoints: DPO step 100, "
          "GRPO step 400, RLOO step 600) is put into one category. Categories are checked in the order of "
          "the table and the first that applies is used. They come from simple pattern rules that compare "
          "the predicted SQL with the gold SQL (tables after FROM / JOIN, number of SELECTs and set "
          "operations, aggregate functions, GROUP BY / HAVING / ORDER BY / LIMIT / DISTINCT), so they are "
          "an approximation, not a full SQL parse.", ""]
    rows_csv = []
    data = {s: {m: load(m, s) for m in MODELS} for s in SPLITS}

    # 1. error categories per model and benchmark (% of all questions)
    for split, sname in SPLITS.items():
        n = len(data[split]["base"])
        md += [f"## {sname} ({n} questions): share of questions per category (%)", "",
               "| Category | " + " | ".join(NAMES[m] for m in MODELS) + " |",
               "|---|" + "---|" * len(MODELS)]
        counts = {m: Counter(category(r) for r in data[split][m].values()) for m in MODELS}
        for c in CATEGORIES:
            md.append(f"| {c} | " + " | ".join(pct(counts[m][c], n) for m in MODELS) + " |")
            for m in MODELS:
                rows_csv.append({"split": split, "model": m, "category": c, "count": counts[m][c], "total": n})
        md.append("")

    # 2. what changes from the untrained model to each trained model
    md += ["## What training changes, question by question", "",
           "Fixed = wrong before training, right after. Broken = right before, wrong after.", "",
           "| Benchmark | Model | Fixed | Broken | Net gain (questions) | Net gain (points) |", "|---|---|---|---|---|---|"]
    for split, sname in SPLITS.items():
        base = data[split]["base"]
        n = len(base)
        for m in ["dpo", "grpo", "rloo"]:
            fixed = sum(1 for q, r in data[split][m].items() if r["correct"] and not base[q]["correct"])
            broken = sum(1 for q, r in data[split][m].items() if not r["correct"] and base[q]["correct"])
            md.append(f"| {sname} | {NAMES[m]} | {fixed} | {broken} | {fixed - broken} | {100 * (fixed - broken) / n:+.1f} |")
    md.append("")

    # 3. which untrained-model errors get fixed by RL (Spider dev and BIRD dev)
    for split in ["spider_dev", "bird_dev"]:
        base = data[split]["base"]
        md += [f"## {SPLITS[split]}: which of the untrained model's errors are fixed", "",
               "For each category of error the untrained model makes: how many of those questions "
               "each trained model gets right.", "",
               "| Untrained model's error | Questions | " + " | ".join(f"fixed by {NAMES[m]}" for m in ["dpo", "grpo", "rloo"]) + " |",
               "|---|---|---|---|---|"]
        for c in CATEGORIES[1:]:
            qs = [q for q, r in base.items() if category(r) == c]
            if not qs:
                continue
            cells = []
            for m in ["dpo", "grpo", "rloo"]:
                k = sum(data[split][m][q]["correct"] for q in qs)
                cells.append(f"{k} ({pct(k, len(qs))}%)")
            md.append(f"| {c} | {len(qs)} | " + " | ".join(cells) + " |")
        md.append("")

    # 4. BIRD by difficulty
    bird = data["bird_dev"]
    levels = ["simple", "moderate", "challenging"]
    md += ["## BIRD dev: accuracy by difficulty (%)", "",
           "| Model | " + " | ".join(f"{l} ({sum(r['difficulty'] == l for r in bird['base'].values())})" for l in levels) + " |",
           "|---|---|---|---|"]
    for m in MODELS:
        cells = []
        for l in levels:
            qs = [r for r in bird[m].values() if r["difficulty"] == l]
            cells.append(pct(sum(r["correct"] for r in qs), len(qs)))
        md.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    md.append("")

    with open(os.path.join(OUT, "error_analysis.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(OUT, "error_categories.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "model", "category", "count", "total"])
        w.writeheader()
        w.writerows(rows_csv)
    plot(data)
    print("\n".join(md))


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#4A3AA7"]  # one per error category
    errors = CATEGORIES[1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, sharey=False)
    for ax, split in zip(axes, ["spider_dev", "bird_dev"]):
        n = len(data[split]["base"])
        left = [0.0] * len(MODELS)
        for c, col in zip(errors, colours):
            vals = [100 * sum(category(r) == c for r in data[split][m].values()) / n for m in MODELS]
            ax.barh([NAMES[m] for m in MODELS], vals, left=left, color=col, label=c, edgecolor="white", linewidth=1.5)
            left = [a + b for a, b in zip(left, vals)]
        for i, total in enumerate(left):
            ax.text(total + 1, i, f"{total:.1f}% wrong", va="center", fontsize=8, color="#3A3E46")
        ax.set_title(f"{SPLITS[split]}: wrong answers by type", loc="left", fontsize=10)
        ax.set_xlabel("% of all questions", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(left) + 18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[1].legend(loc="lower center", bbox_to_anchor=(-0.1, -0.42), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_categories.png"), facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
