"""Benchmark the execution harness.

Acceptance target from spec Section 6: 100 queries scored end to end in under 60 seconds.
The harness is the project's bottleneck -- it is CPU-bound and the GPU waits on it -- so
this number gets measured, recorded in NOTES.md, and re-measured whenever the harness
changes.

"Scored end to end" means executed AND compared against the gold hash, because comparison
is part of what happens per candidate during a rollout.

Usage:
    python scripts/benchmark_harness.py --n 100
"""

from __future__ import annotations

import argparse
import os
import time

from src.data.loader import Example, load_examples
from src.execution.compare import score
from src.execution.harness import close_pool, execute

TRAIN_FILE = os.path.join("data", "processed", "spider_train_filtered.jsonl")


def benchmark(examples: list[Example]) -> dict:
    """Execute every example's gold query and score the result.

    Gold queries are used as the workload because they are guaranteed to be valid SQL
    against their database, so the measurement reflects execution cost rather than the
    cost of erroring out early.
    """
    started = time.monotonic()

    rewards = []
    for example in examples:
        result = execute(example.gold_sql, example.db_id)
        rewards.append(score(result, example.gold_result_hash, example.order_matters))

    elapsed = time.monotonic() - started

    # Gold queries scored against their own hashes should all come back 1.0. Anything
    # else means the harness and the preprocessing pass disagree, which is a much more
    # serious problem than a slow benchmark.
    n_correct = sum(1 for r in rewards if r == 1.0)

    return {
        "n": len(examples),
        "elapsed_s": elapsed,
        "per_query_ms": (elapsed / max(1, len(examples))) * 1000.0,
        "gold_scored_correct": n_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="harness throughput benchmark")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--file", default=TRAIN_FILE)
    parser.add_argument("--target-s", type=float, default=60.0)
    args = parser.parse_args()

    examples = load_examples(args.file, n=args.n)
    if not examples:
        raise SystemExit("no examples in %s -- run the filter pass first" % args.file)

    stats = benchmark(examples)

    print("queries            %d" % stats["n"])
    print("elapsed            %.2f s   (target < %.0f s)" % (stats["elapsed_s"], args.target_s))
    print("per query          %.1f ms" % stats["per_query_ms"])
    print("gold scored 1.0    %d / %d" % (stats["gold_scored_correct"], stats["n"]))

    if stats["gold_scored_correct"] != stats["n"]:
        print()
        print("WARNING: a gold query did not score 1.0 against its own hash.")
        print("The harness and the filtering pass disagree. Fix this before anything else.")

    print()
    if stats["elapsed_s"] < args.target_s:
        print("PASS -- record this number in NOTES.md")
    else:
        print("FAIL -- %.2f s exceeds the %.0f s target." % (stats["elapsed_s"], args.target_s))
        print("Do not relax the target. Check: connection pooling, parallelism, n_workers.")

    close_pool()


if __name__ == "__main__":
    main()
