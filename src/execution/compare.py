"""Result-set comparison and the reward function.

CONTRACT FILE. PR + ping before changing. Frozen end of week 6.

This module defines what "correct" means for the entire project. Two very different
looking SQL strings can be equally correct, so correctness is decided solely by the
rows a query returns when executed -- never by comparing SQL text.

Semantics, which must be documented in the report because they change every number:

  - values are compared, NOT column names
  - row order is ignored unless order_matters, in which case it is enforced
  - column order is always enforced
  - numbers are normalised to float and rounded to 6 decimal places
    (so 1 and 1.0 compare equal; so does True and 1)
  - None becomes a sentinel that no real value can collide with
  - strings are stripped of leading/trailing whitespace
  - duplicate rows are significant and are NEVER deduplicated
  - an empty predicted result never matches (see results_match)

The gold hash is computed once at preprocessing time by hash_rows, and predictions are
hashed at training time by the same function. If those two ever diverge every reward in
the project is wrong and nothing visibly breaks, which is why this lives in one module
and is the only file in the repo with mandatory unit tests.
"""

from __future__ import annotations

import hashlib
import json

# Chosen so no SQL string value can collide with a NULL.
NULL_SENTINEL = "\x00__NULL__\x00"

FLOAT_DP = 6


def _norm_value(value):
    """Normalise one cell so that equal values hash identically."""
    if value is None:
        return NULL_SENTINEL

    # bool is a subclass of int, so it has to be handled before the numeric branch
    # or True would stringify differently from 1 depending on the driver.
    if isinstance(value, bool):
        return round(float(int(value)), FLOAT_DP)

    if isinstance(value, (int, float)):
        return round(float(value), FLOAT_DP)

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (bytes, bytearray)):
        # BLOBs are rare in Spider but decode them deterministically rather than
        # letting json choke later.
        return bytes(value).hex()

    # Dates and anything else sqlite3 hands back untouched.
    return str(value)


def _row_sort_key(row):
    """Order-insensitive but type-safe sort key.

    Rows can mix ints, strings and the NULL sentinel, and Python 3 refuses to compare
    those directly. Sorting on (type name, string form) per cell is stable and total.
    """
    return [(type(cell).__name__, str(cell)) for cell in row]


def normalise_rows(rows: list[tuple], order_matters: bool) -> list[tuple]:
    """Normalise every cell, and sort rows iff row order is not significant.

    Duplicates are preserved. Column order within each row is left alone, which is how
    column order ends up enforced.
    """
    normalised = [tuple(_norm_value(cell) for cell in row) for row in rows]
    if not order_matters:
        normalised.sort(key=_row_sort_key)
    return normalised


def hash_rows(rows: list[tuple], order_matters: bool) -> str:
    """Stable hash of a normalised result set.

    Used for gold_result_hash during preprocessing AND for predictions during training.
    It must be the same function in both places.
    """
    normalised = normalise_rows(rows, order_matters)
    # Tuples serialise as JSON arrays; that is fine, it only has to be deterministic.
    payload = json.dumps(normalised, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def results_match(pred, gold_hash: str, order_matters: bool) -> bool:
    """True iff the predicted rows match the gold rows the hash was built from.

    `pred` is an ExecResult (src.execution.harness).

    An empty predicted result never matches. Preprocessing already drops every example
    whose gold query returns zero rows, because under an execution reward a completely
    wrong query that happens to return nothing would otherwise score 1.0. This guard is
    belt-and-braces for the same failure: spec Section 16, "reward spikes to 1.0
    immediately".
    """
    if pred is None:
        return False
    if pred.status != "ok":
        return False
    if not pred.rows:
        return False
    return hash_rows(pred.rows, order_matters) == gold_hash


def score(result, gold_hash: str, order_matters: bool) -> float:
    """THE REWARD FUNCTION.

    -0.1  the query failed to execute or timed out
     1.0  the query executed and returned the gold rows
     0.0  the query executed and returned something else

    Nothing else. No length penalty, no format bonus, no partial credit, no stepwise
    shaping. It is identical across all conditions and must never be tuned per
    condition -- that is invariant 1, and changing it means re-running everything.
    """
    if result.status in ("error", "timeout"):
        return -0.1
    return 1.0 if results_match(result, gold_hash, order_matters) else 0.0
