"""The filtering pass. Implement and run this before anything else -- it gates everything.

Removes training examples where:

  1. the gold query returns ZERO ROWS. Under an execution reward this is poison: a
     completely wrong query that also returns nothing would score 1.0. This is the single
     most important preprocessing step in the project.
  2. the gold query takes longer than 5 seconds
  3. the gold query errors against its database
  4. the db_id has no corresponding sqlite file

Expect roughly 1,700 removals from Spider, based on Arctic-Text2SQL-R1. If the number is
wildly different, stop and investigate before proceeding -- it means the pass is not doing
what it looks like it is doing.

Also precomputes gold_result_hash, once, here. Re-executing the gold query every training
step would dominate wall-clock time.

Usage:
    python -m src.data.filter --report
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

from src.execution.compare import hash_rows
from src.execution.harness import close_pool, db_path, execute

RAW_TRAIN = os.path.join("data", "raw", "spider", "train_spider.json")
OUT_TRAIN = os.path.join("data", "processed", "spider_train_filtered.jsonl")


@dataclass
class FilterReport:
    total_in: int = 0
    kept: int = 0
    removed_empty_result: int = 0
    removed_timeout: int = 0
    removed_error: int = 0
    removed_missing_db: int = 0
    missing_db_ids: set = field(default_factory=set)

    @property
    def removed_total(self) -> int:
        return (
            self.removed_empty_result
            + self.removed_timeout
            + self.removed_error
            + self.removed_missing_db
        )

    def __str__(self) -> str:
        lines = [
            "filter report",
            "-------------",
            "  examples in            %d" % self.total_in,
            "  kept                   %d" % self.kept,
            "  removed (total)        %d   [expected ~1700 for Spider]" % self.removed_total,
            "    gold returned 0 rows %d" % self.removed_empty_result,
            "    gold timed out       %d" % self.removed_timeout,
            "    gold errored         %d" % self.removed_error,
            "    db file missing      %d" % self.removed_missing_db,
        ]
        if self.missing_db_ids:
            sample = sorted(self.missing_db_ids)[:10]
            lines.append("    missing db_ids       %s" % ", ".join(sample))
        return "\n".join(lines)


def _difficulty(gold_sql: str) -> str:
    """Very rough proxy for Spider's own difficulty label.

    Spider ships hardness labels but only for dev; this keeps a comparable field on train
    so error analysis can bucket by it later.
    """
    sql = gold_sql.lower()
    score = 0
    for keyword in ("join", "group by", "having", "union", "intersect", "except", "select"):
        score += sql.count(keyword)
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def _count_tables(gold_sql: str) -> int:
    """Number of distinct tables referenced, approximately."""
    tokens = gold_sql.lower().replace(",", " ").split()
    tables = set()
    for i, token in enumerate(tokens):
        if token in ("from", "join") and i + 1 < len(tokens):
            tables.add(tokens[i + 1])
    return max(1, len(tables))


def filter_examples(
    raw_path: str = RAW_TRAIN,
    out_path: str = OUT_TRAIN,
    timeout_s: float = 5.0,
    n_workers: int = 8,
) -> FilterReport:
    """Run the filtering pass and write the processed JSONL.

    One line per kept example, in the spec Section 4.5 format.
    """
    with open(raw_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    report = FilterReport(total_in=len(raw))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for index, example in enumerate(raw):
            db_id = example["db_id"]
            gold_sql = example.get("query") or example.get("SQL") or ""

            result = execute(gold_sql, db_id, timeout_s=timeout_s)

            if result.status == "timeout":
                report.removed_timeout += 1
                continue

            if result.status == "error":
                if not os.path.exists(db_path(db_id)):
                    report.removed_missing_db += 1
                    report.missing_db_ids.add(db_id)
                else:
                    report.removed_error += 1
                continue

            # THE important one. Zero rows from the gold query means any query that also
            # returns nothing -- including a completely wrong one -- would score 1.0.
            if not result.rows:
                report.removed_empty_result += 1
                continue

            order_matters = "order by" in gold_sql.lower()

            record = {
                "id": "spider_train_%05d" % index,
                "db_id": db_id,
                "question": example["question"],
                "gold_sql": gold_sql,
                "gold_result_hash": hash_rows(result.rows, order_matters),
                "gold_row_count": len(result.rows),
                "order_matters": order_matters,
                "difficulty": _difficulty(gold_sql),
                "n_tables_in_gold": _count_tables(gold_sql),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.kept += 1

    close_pool()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Spider filtering pass")
    parser.add_argument("--raw", default=RAW_TRAIN)
    parser.add_argument("--out", default=OUT_TRAIN)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", action="store_true", help="print the removal report")
    args = parser.parse_args()

    report = filter_examples(
        raw_path=args.raw,
        out_path=args.out,
        timeout_s=args.timeout,
        n_workers=args.workers,
    )

    if args.report:
        print(report)
        print()
        print("wrote %s" % args.out)
        print("Record these counts in NOTES.md.")


if __name__ == "__main__":
    main()
