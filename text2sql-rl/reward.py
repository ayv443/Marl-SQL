# Reward: +1 correct result, 0 wrong result, -0.1 error / no SQL / timeout.
# Don't change this after phase 1, otherwise all methods need re-running.
# Test: python reward.py --split val
import argparse
import os
import pickle
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import PROCESSED_DIR, completion_text, db_full_path, extract_sql, load_jsonl

R_CORRECT, R_WRONG, R_ERROR = 1.0, 0.0, -0.1
TIMEOUT_S = 5.0
MAX_ROWS = 100_000
NUM_WORKERS = 8

_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)


def execute(db_file, sql, timeout=TIMEOUT_S):
    # returns (rows, error)
    if not sql:
        return None, "no sql found"
    if not sql.lstrip().lower().startswith(("select", "with")):
        return None, "not a SELECT"
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return None, f"cannot open db: {e}"
    conn.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.monotonic() + timeout
    # stops the query once the deadline passes
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS)
        return rows, None
    except Exception as e:
        return None, "timeout" if "interrupted" in str(e) else str(e)
    finally:
        conn.close()


def has_order_by(sql):
    return "order by" in " ".join(sql.lower().split())


def _normalise(rows, ordered):
    rows = [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows]
    return rows if ordered else Counter(rows)


def results_match(pred_rows, gold_rows, ordered):
    return _normalise(pred_rows, ordered) == _normalise(gold_rows, ordered)


def score(db_file, pred_sql, gold_rows, ordered):
    rows, err = execute(db_file, pred_sql)
    if err is not None:
        return R_ERROR, False
    return (R_CORRECT if results_match(rows, gold_rows, ordered) else R_WRONG), True


def score_many(jobs):
    # jobs: list of (db_file, pred_sql, gold_rows, ordered)
    return list(_pool.map(lambda j: score(*j), jobs))


_gold_cache = None


def gold_cache():
    global _gold_cache
    if _gold_cache is None:
        with open(os.path.join(PROCESSED_DIR, "gold_cache.pkl"), "rb") as f:
            _gold_cache = pickle.load(f)
    return _gold_cache


def gold_rows_for(rows):
    # train/val gold results are cached, eval sets get run here
    cache = gold_cache()
    missing = [r for r in rows if r["qid"] not in cache]
    fresh = _pool.map(lambda r: execute(db_full_path(r["db_path"]), r["gold_sql"], timeout=30)[0], missing)
    for r, res in zip(missing, fresh):
        cache[r["qid"]] = res if res is not None else []
    return [cache[r["qid"]] for r in rows]


def _jobs(completions, qid, db_path, gold_sql):
    gold = gold_cache()
    return [(db_full_path(d), extract_sql(completion_text(c)), gold[q], has_order_by(g))
            for c, q, d, g in zip(completions, qid, db_path, gold_sql)]


# reward functions for TRL (extra dataset columns come in as kwargs)
def execution_reward(prompts, completions, qid, db_path, gold_sql, **kwargs):
    return [r for r, _ in score_many(_jobs(completions, qid, db_path, gold_sql))]


def valid_sql_reward(prompts, completions, qid, db_path, gold_sql, **kwargs):
    # weight 0 in training, just so the valid SQL rate shows up in wandb
    return [float(ok) for _, ok in score_many(_jobs(completions, qid, db_path, gold_sql))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    rows = load_jsonl(os.path.join(PROCESSED_DIR, f"{args.split}.jsonl"))
    fake = [f"<answer>{r['gold_sql']}</answer>" for r in rows]
    rewards = execution_reward(None, fake, [r["qid"] for r in rows],
                               [r["db_path"] for r in rows], [r["gold_sql"] for r in rows])
    print(f"gold vs gold mean reward on {args.split}: {sum(rewards) / len(rewards):.3f} (should be 1.000)")

    r0 = rows[0]
    db = db_full_path(r0["db_path"])
    g = gold_cache()[r0["qid"]]
    print("wrong result  ->", score(db, "SELECT 12345", g, False)[0], "(should be 0.0)")
    print("syntax error  ->", score(db, "SELEC nonsense", g, False)[0], "(should be -0.1)")
    print("not a SELECT  ->", score(db, "DROP TABLE x", g, False)[0], "(should be -0.1)")
    print("no answer tag ->", execution_reward(None, ["I don't know"], [r0["qid"]],
                                               [r0["db_path"]], [r0["gold_sql"]])[0], "(should be -0.1)")
    t = time.time()
    print("timeout       ->", score(db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                                       "SELECT count(*) FROM c", g, False)[0],
          f"(should be -0.1, took {time.time() - t:.1f} s)")
