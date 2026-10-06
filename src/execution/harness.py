"""Pooled, timed, read-only SQL execution.

CONTRACT FILE. PR + ping before changing. Frozen end of week 6.

This is the performance-critical component of the project and it is CPU-bound. It, not
the GPU, is the bottleneck: every rollout executes G candidate queries per question plus
the counterfactual re-runs. If this is slow the GPU sits idle.

Requirements, from spec Section 6:
  - connection pooling per db_id (opening a connection per query destroys throughput)
  - read-only connections, so a generated query can never mutate a database
  - a timeout that actually kills the query, not one that waits for a lock
  - never raise on bad SQL -- return status="error" with the message, the Refiner reads it
  - parallel across PROCESSES, not threads (sqlite3 + the GIL)

Acceptance target: 100 queries scored end to end in under 60 seconds.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Literal, Optional

DB_ROOT = os.path.join("data", "raw", "spider", "database")

MAX_ERROR_CHARS = 500

# One pool per process. Worker processes each build their own lazily.
_CONNECTIONS: dict[str, sqlite3.Connection] = {}


@dataclass
class ExecResult:
    status: Literal["ok", "error", "timeout"]
    rows: Optional[list[tuple]]
    error_msg: Optional[str]
    elapsed_ms: float


def db_path(db_id: str) -> str:
    """Path to a Spider database file."""
    return os.path.join(DB_ROOT, db_id, db_id + ".sqlite")


def _get_connection(db_id: str) -> sqlite3.Connection:
    """Return a pooled read-only connection for db_id, opening it on first use.

    The mode=ro URI is what makes a generated DROP TABLE harmless.
    """
    conn = _CONNECTIONS.get(db_id)
    if conn is not None:
        return conn

    path = db_path(db_id)
    if not os.path.exists(path):
        raise FileNotFoundError("no sqlite file for db_id " + db_id)

    uri = "file:" + path + "?mode=ro"
    conn = sqlite3.connect(uri, check_same_thread=False)
    # Some Spider databases contain text that is not valid UTF-8.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    _CONNECTIONS[db_id] = conn
    return conn


def close_pool() -> None:
    """Release every pooled connection.

    Call at process teardown. Skipping this is what shows up as step time growing
    steadily over a long run.
    """
    for conn in _CONNECTIONS.values():
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _CONNECTIONS.clear()


def _install_timeout(conn: sqlite3.Connection, timeout_s: float, started: float) -> None:
    """Abort the query once timeout_s has elapsed.

    sqlite3's connect(timeout=...) only governs how long we wait for a lock, which is
    not what we need -- a cartesian-product join holds no locks and would run forever.
    The progress handler is called every N virtual-machine instructions and aborts the
    statement when it returns non-zero.
    """

    def _handler():
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return 1 if elapsed_ms > timeout_s else 0

    conn.set_progress_handler(_handler, 10000)


def execute(sql: str, db_id: str, timeout_s: float = 5.0) -> ExecResult:
    """Execute one query read-only against db_id.

    Never raises on bad SQL. Errors are completely normal here -- the Refiner is given
    the message and is expected to learn from it.
    """
    started = time.monotonic()
    try:
        conn = _get_connection(db_id)
    except (FileNotFoundError, sqlite3.Error) as exc:
        return ExecResult(
            status="error",
            rows=None,
            error_msg=str(exc)[:MAX_ERROR_CHARS],
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )

    _install_timeout(conn, timeout_s, started)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return ExecResult(
            status="ok",
            rows=rows,
            error_msg=None,
            elapsed_ms=time.monotonic() - started,
        )
    except sqlite3.OperationalError as exc:
        # The progress handler aborting a statement surfaces here as "interrupted".
        message = str(exc)
        status = "timeout" if "interrupt" in message.lower() else "error"
        return ExecResult(
            status=status,
            rows=None,
            error_msg=message[:MAX_ERROR_CHARS],
            elapsed_ms=time.monotonic() - started,
        )
    except sqlite3.Error as exc:
        return ExecResult(
            status="error",
            rows=None,
            error_msg=str(exc)[:MAX_ERROR_CHARS],
            elapsed_ms=time.monotonic() - started,
        )
    finally:
        conn.set_progress_handler(None, 0)


def _execute_one(item: tuple[str, str]) -> ExecResult:
    """Worker entry point. Must be module-level so it can be pickled."""
    sql, db_id = item
    return execute(sql, db_id)


def execute_batch(
    items: list[tuple[str, str]],
    n_workers: int = 8,
) -> list[ExecResult]:
    """Execute (sql, db_id) pairs in parallel.

    Results come back in the same order as `items`, which callers rely on to line each
    result up with the episode that produced it.

    Processes rather than threads: sqlite3 releases the GIL unevenly and the work is
    CPU-bound in the sqlite VM.
    """
    if not items:
        return []

    with Pool(processes=4) as pool:
        results = pool.map(_execute_one, items)
    return results
