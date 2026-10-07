# Evaluate every checkpoint on the val split and plot val EX vs step.
# python val_curve.py --runs outputs/dpo-1.5b-s0 outputs/grpo-1.5b-s0 outputs/rloo-1.5b-s0
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monitor import Monitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--include_base", type=int, default=1)
    args = ap.parse_args()
    with Monitor("val-curve", settings=vars(args)) as mon:
        run(args, mon)


def run(args, mon):
    rows = []
    if args.include_base:  # step 0 = untrained model
        if not os.path.exists("results/base/val/metrics.json"):
            subprocess.run([sys.executable, "evaluate.py", "--split", "val", "--tag", "base",
                            "--engine", args.engine, "--notify", "0"], check=True)
        base_ex = json.load(open("results/base/val/metrics.json"))["EX"]

    for run in args.runs:
        method = os.path.basename(run.rstrip("/"))
        if args.include_base:
            rows.append({"method": method, "step": 0, "val_EX": base_ex})
        ckpts = sorted(glob.glob(f"{run}/checkpoint-*"), key=lambda p: int(re.findall(r"\d+$", p)[0]))
        for ck in ckpts:
            step = int(re.findall(r"\d+$", ck)[0])
            tag = f"curve/{method}/step{step}"
            metrics_file = f"results/{tag}/val/metrics.json"
            if not os.path.exists(metrics_file):
                subprocess.run([sys.executable, "evaluate.py", "--split", "val", "--adapter", ck,
                                "--tag", tag, "--engine", args.engine, "--notify", "0"], check=True)
            rows.append({"method": method, "step": step, "val_EX": json.load(open(metrics_file))["EX"]})
            print(rows[-1])

    os.makedirs("results", exist_ok=True)
    with open("results/val_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "step", "val_EX"])
        w.writeheader()
        w.writerows(rows)

    plt.figure(figsize=(7, 4.5))
    for method in dict.fromkeys(r["method"] for r in rows):
        pts = [(r["step"], r["val_EX"]) for r in rows if r["method"] == method]
        plt.plot(*zip(*pts), marker="o", label=method)
    plt.xlabel("training step")
    plt.ylabel("validation EX")
    plt.title("Validation EX during training (held-out Spider train DBs)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/val_curve.png", dpi=150)
    print("saved results/val_curve.png")
    best = {}
    for r in rows:
        if r["val_EX"] >= best.get(r["method"], {"val_EX": -1})["val_EX"]:
            best[r["method"]] = r
    mon.result = {m: f"best val EX {r['val_EX']:.3f} at step {r['step']}" for m, r in best.items()}


if __name__ == "__main__":
    main()
