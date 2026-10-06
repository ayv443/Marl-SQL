"""Unit tests for result comparison. THE ONE MANDATORY TEST FILE.

compare.py is about a hundred lines of pure function with no GPU dependency, and it
defines correctness for the entire project. Its bugs are invisible: a comparison that
wrongly returns True for empty-vs-empty makes reward spike to 1.0 from step one, which
looks like spectacular success and is listed in spec Section 16 as a known failure mode.
Finding that in the training loop costs a week. Finding it here costs nothing.

Run:  pytest tests/test_compare.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.compare import hash_rows, normalise_rows, results_match, score
from src.execution.harness import ExecResult


def ok(rows):
    """An ExecResult that executed cleanly and returned `rows`."""
    return ExecResult(status="ok", rows=rows, error_msg=None, elapsed_ms=1.0)


def gold(rows, order_matters=False):
    return hash_rows(rows, order_matters)


# --- row order ------------------------------------------------------------------

def test_identical_rows_match():
    rows = [(1, "a"), (2, "b")]
    assert results_match(ok(rows), gold(rows), order_matters=False)


def test_row_order_ignored_when_order_does_not_matter():
    assert results_match(ok([(2, "b"), (1, "a")]), gold([(1, "a"), (2, "b")]), False)


def test_row_order_enforced_when_order_matters():
    g = gold([(1, "a"), (2, "b")], order_matters=True)
    assert not results_match(ok([(2, "b"), (1, "a")]), g, order_matters=True)
    assert results_match(ok([(1, "a"), (2, "b")]), g, order_matters=True)


# --- column order ---------------------------------------------------------------

def test_column_order_is_enforced():
    assert not results_match(ok([(1, 2)]), gold([(2, 1)]), order_matters=False)


def test_column_count_mismatch_does_not_match():
    assert not results_match(ok([(1, 2, 3)]), gold([(1, 2)]), order_matters=False)


# --- duplicates -----------------------------------------------------------------

def test_duplicate_rows_are_significant():
    assert not results_match(ok([(1,), (1,)]), gold([(1,)]), order_matters=False)
    assert not results_match(ok([(1,)]), gold([(1,), (1,)]), order_matters=False)


def test_duplicate_rows_preserved_by_normalisation():
    assert len(normalise_rows([(1,), (1,), (1,)], order_matters=False)) == 3


# --- numbers --------------------------------------------------------------------

def test_floats_rounded_to_six_dp():
    assert results_match(ok([(1.0000000001,)]), gold([(1.0,)]), order_matters=False)


def test_float_difference_above_six_dp_does_not_match():
    assert not results_match(ok([(1.001,)]), gold([(1.0,)]), order_matters=False)


def test_int_and_float_compare_equal():
    # count(*) comes back as int from one driver and float from another.
    assert results_match(ok([(1,)]), gold([(1.0,)]), order_matters=False)


# --- NULL and strings -----------------------------------------------------------

def test_null_matches_null():
    assert results_match(ok([(None,)]), gold([(None,)]), order_matters=False)


def test_null_does_not_match_empty_string():
    assert not results_match(ok([(None,)]), gold([("",)]), order_matters=False)


def test_strings_are_stripped():
    assert results_match(ok([("  hello ",)]), gold([("hello",)]), order_matters=False)


# --- empty results: the dangerous case ------------------------------------------

def test_empty_prediction_never_matches():
    """The failure that makes reward spike to 1.0 immediately."""
    assert not results_match(ok([]), gold([(1,)]), order_matters=False)


def test_empty_vs_empty_does_not_match():
    assert not results_match(ok([]), gold([]), order_matters=False)


# --- the reward function --------------------------------------------------------

def test_score_error_is_minus_point_one():
    bad = ExecResult(status="error", rows=None, error_msg="no such table", elapsed_ms=1.0)
    assert score(bad, gold([(1,)]), False) == -0.1


def test_score_timeout_is_minus_point_one():
    slow = ExecResult(status="timeout", rows=None, error_msg="interrupted", elapsed_ms=5000.0)
    assert score(slow, gold([(1,)]), False) == -0.1


def test_score_correct_is_one():
    assert score(ok([(1,)]), gold([(1,)]), False) == 1.0


def test_score_executed_but_wrong_is_zero():
    assert score(ok([(2,)]), gold([(1,)]), False) == 0.0


def test_score_has_only_three_possible_values():
    """No partial credit, no length penalty, no format bonus. Ever."""
    g = gold([(1,)])
    observed = {
        score(ok([(1,)]), g, False),
        score(ok([(2,)]), g, False),
        score(ExecResult("error", None, "boom", 1.0), g, False),
    }
    assert observed == {1.0, 0.0, -0.1}


# --- hashing is stable ----------------------------------------------------------

def test_hash_is_stable_across_calls():
    rows = [(1, "a"), (2, None), (3.5, "c")]
    assert hash_rows(rows, False) == hash_rows(rows, False)


def test_hash_differs_between_order_modes_for_unsorted_input():
    rows = [(2,), (1,)]
    assert hash_rows(rows, True) != hash_rows(rows, False)
