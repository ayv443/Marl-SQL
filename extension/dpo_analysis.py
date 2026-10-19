# Why did DPO not help? Uses the DPO training log and the validation curve (no GPU needed).
#   python extension/dpo_analysis.py
# Reads text2sql-rl/proof/dpo-1.5b-s0/logs/*_steps.jsonl and text2sql-rl/evaluation/results/val_curve.csv
# Writes extension/findings/dpo_analysis.md and dpo_analysis.png
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T2S = os.path.join(HERE, "..", "text2sql-rl")
OUT = os.path.join(HERE, "findings")


def load_steps():
    path = sorted(glob.glob(os.path.join(T2S, "proof", "dpo-1.5b-s0", "logs", "*_steps.jsonl")))[-1]
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if "logps/chosen" in r]


def load_val_curve(method="dpo-1.5b-s0"):
    rows = csv.DictReader(open(os.path.join(T2S, "evaluation", "results", "val_curve.csv")))
    return [(int(float(r["step"])), float(r["val_EX"])) for r in rows if r["method"] == method]


def window_mean(rows, key, start, end):
    vals = [r[key] for r in rows if start <= r["step"] <= end]
    return sum(vals) / len(vals)


def smooth(xs, w=25):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_steps()
    curve = load_val_curve()
    last = rows[-1]["step"]
    keys = [("rewards/accuracies", "Preference accuracy (chosen scored above rejected)"),
            ("rewards/margins", "Reward margin (chosen minus rejected)"),
            ("logps/chosen", "Log-probability of the chosen answer"),
            ("logps/rejected", "Log-probability of the rejected answer")]
    windows = [(1, 10), (91, 110), (291, 310), (last - 9, last)]

    md = ["# Why DPO did not help", "",
          "DPO trains on (chosen, rejected) pairs built from the untrained model's own samples. "
          "Mean of each signal over a few windows of training steps (from the DPO training log), "
          "next to the validation accuracy of the closest checkpoint:", "",
          "| Signal | " + " | ".join(f"steps {a} to {b}" for a, b in windows) + " |",
          "|---|" + "---|" * len(windows)]
    for key, label in keys:
        md.append(f"| {label} | " + " | ".join(f"{window_mean(rows, key, a, b):.3f}" for a, b in windows) + " |")
    val = dict(curve)
    md.append("| Validation EX of the checkpoint (%) | " + " | ".join(
        f"{100 * val[s]:.1f} (step {s})" for s in [0, 100, 300, max(val)]) + " |")
    md += ["",
           "What the numbers show:",
           "- The model gets better at the DPO objective: it scores the chosen answer above the rejected one "
           "more and more often, and the margin between them grows.",
           "- But the log-probability of the chosen (correct) answers also goes down. The rejected answers "
           "go down faster, which is enough to lower the loss, but the model becomes less likely to write the "
           "correct SQL it was shown. This is called likelihood displacement.",
           "- Validation accuracy falls from the first checkpoint on, so the best DPO checkpoint is step 100, "
           "and it is still slightly below the untrained model on every test set.",
           "- GRPO and RLOO do not have this problem because they keep sampling new answers from the current "
           "model and reward the ones that actually run correctly, while DPO only sees fixed pairs from the "
           "untrained model."]
    with open(os.path.join(OUT, "dpo_analysis.md"), "w") as f:
        f.write("\n".join(md))
    plot(rows, curve)
    print("\n".join(md))


def plot(rows, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200)
    a1.plot(steps, smooth([r["rewards/accuracies"] for r in rows]), color="#2A78D6", lw=2)
    a1.set_title("Preference accuracy", loc="left", fontsize=10)
    a1.set_ylim(0, 1)
    a2.plot(steps, smooth([r["logps/chosen"] for r in rows]), color="#1BAF7A", lw=2, label="chosen (correct)")
    a2.plot(steps, smooth([r["logps/rejected"] for r in rows]), color="#EB6834", lw=2, label="rejected (wrong)")
    a2.set_title("Log-probability of the answers", loc="left", fontsize=10)
    a2.legend(fontsize=8, frameon=False)
    xs, ys = zip(*curve)
    a3.plot(xs, [100 * y for y in ys], color="#15171C", lw=2, marker="o")
    a3.set_title("Validation accuracy (%)", loc="left", fontsize=10)
    for ax in (a1, a2, a3):
        ax.set_xlabel("training step", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#ECECE6")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dpo_analysis.png"), facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
