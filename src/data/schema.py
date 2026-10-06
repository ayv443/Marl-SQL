"""Schema serialisation.

Produces the compact text representation the agents see: table names, columns with types,
primary and foreign keys, and up to three sample values per column. The sample values
matter for value grounding -- without them the model has to guess how a country is spelled
in this particular database.

Serialised schemas are cached. Rebuilding one per rollout would be pure waste: the same
db_id appears in many examples and the schema never changes.

The Selector emits a TABLE LIST, and sub_schema() materialises the reduced schema from it.
The Selector never regenerates schema text -- it hallucinates columns when it tries.
"""

from __future__ import annotations

import json
import os
import sqlite3

from src.execution.harness import db_path

TABLES_JSON = os.path.join("data", "raw", "spider", "tables.json")

_SCHEMA_CACHE: dict[str, str] = {}
_TABLES_META: dict | None = None


def _load_tables_meta() -> dict:
    """Load tables.json once, keyed by db_id."""
    global _TABLES_META
    if _TABLES_META is None:
        with open(TABLES_JSON, encoding="utf-8") as fh:
            entries = json.load(fh)
        _TABLES_META = {entry["db_id"]: entry for entry in entries}
    return _TABLES_META


def table_list(db_id: str) -> list[str]:
    """Every table name in the database.

    Used to validate the Selector's output -- a table it invented is dropped rather than
    passed downstream.
    """
    meta = _load_tables_meta().get(db_id)
    if meta is not None:
        return list(meta["table_names_original"])

    conn = sqlite3.connect("file:" + db_path(db_id) + "?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def _sample_values(conn: sqlite3.Connection, table: str, column: str, n: int) -> list[str]:
    """Up to n distinct non-null values from one column, as display strings."""
    query = "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL LIMIT %d" % (
        column,
        table,
        column,
        n,
    )
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.Error:
        # Empty table, odd column name, whatever -- sample values are a nicety.
        return []

    out = []
    for row in rows:
        text = str(row[0])
        if len(text) > 30:
            text = text[:27] + "..."
        out.append(text)
    return out


def _foreign_keys(meta: dict) -> list[str]:
    """Render tables.json's foreign key pairs as 'a.col -> b.col' strings."""
    names = meta["table_names_original"]
    columns = meta["column_names_original"]
    out = []
    for src_idx, dst_idx in meta.get("foreign_keys", []):
        src_table, src_col = columns[src_idx]
        dst_table, dst_col = columns[dst_idx]
        out.append(
            "%s.%s -> %s.%s"
            % (names[src_table], src_col, names[dst_table], dst_col)
        )
    return out


def serialise_schema(db_id: str, n_sample_values: int = 3) -> str:
    """Compact schema text for one database. Cached.

    Shape:

        # table: singer
        #   Singer_ID (number) PK  e.g. 1, 2, 3
        #   Name (text)  e.g. Joe Sharp, Timbaland
        # foreign keys:
        #   concert.Stadium_ID -> stadium.Stadium_ID
    """
    cached = _SCHEMA_CACHE.get(db_id)
    if cached is not None:
        return cached

    meta = _load_tables_meta().get(db_id)
    if meta is None:
        raise KeyError("db_id %s is not in tables.json" % db_id)

    names = meta["table_names_original"]
    columns = meta["column_names_original"]
    types = meta["column_types"]
    primary = set(meta.get("primary_keys", []))

    conn = sqlite3.connect("file:" + db_path(db_id) + "?mode=ro", uri=True)
    try:
        blocks = []
        for table_idx, table_name in enumerate(names):
            lines = ["# table: %s" % table_name]
            for col_idx, (owner, col_name) in enumerate(columns):
                if owner != table_idx:
                    continue
                parts = ["#   %s (%s)" % (col_name, types[col_idx])]
                if col_idx in primary:
                    parts.append("PK")
                if n_sample_values > 0:
                    samples = _sample_values(conn, table_name, col_name, n_sample_values)
                    if samples:
                        parts.append(" e.g. " + ", ".join(samples))
                lines.append(" ".join(parts))
            blocks.append("\n".join(lines))
    finally:
        conn.close()

    keys = _foreign_keys(meta)
    if keys:
        blocks.append("# foreign keys:\n" + "\n".join("#   " + k for k in keys))

    text = "\n".join(blocks)
    _SCHEMA_CACHE[db_id] = text
    return text


def sub_schema(db_id: str, tables: list[str], n_sample_values: int = 3) -> str:
    """Materialise the reduced schema for a subset of tables.

    This is what the Selector's output turns into. Tables that do not exist in the
    database are dropped silently -- a hallucinated table name should cost the Selector
    reward, not crash the rollout.

    If nothing survives, fall back to the full schema. An empty schema guarantees a
    failed query, which would make the Selector's reward signal useless.
    """
    full_names = table_list(db_id)
    wanted = [name for name in tables if name in full_names]

    if not wanted:
        return serialise_schema(db_id, n_sample_values=n_sample_values)

    meta = _load_tables_meta()[db_id]
    names = meta["table_names_original"]
    columns = meta["column_names_original"]
    types = meta["column_types"]
    primary = set(meta.get("primary_keys", []))

    conn = sqlite3.connect("file:" + db_path(db_id) + "?mode=ro", uri=True)
    try:
        blocks = []
        for table_name in wanted:
            table_idx = names.index(table_name)
            lines = ["# table: %s" % table_name]
            for col_idx, (owner, col_name) in enumerate(columns):
                if owner != table_idx:
                    continue
                parts = ["#   %s (%s)" % (col_name, types[col_idx])]
                if col_idx in primary:
                    parts.append("PK")
                if n_sample_values > 0:
                    samples = _sample_values(conn, table_name, col_name, n_sample_values)
                    if samples:
                        parts.append(" e.g. " + ", ".join(samples))
                lines.append(" ".join(parts))
            blocks.append("\n".join(lines))
    finally:
        conn.close()

    # Keep only foreign keys where both ends survived the reduction.
    keys = [
        k
        for k in _foreign_keys(meta)
        if k.split(".")[0] in wanted and k.split("-> ")[1].split(".")[0] in wanted
    ]
    if keys:
        blocks.append("# foreign keys:\n" + "\n".join("#   " + k for k in keys))

    return "\n".join(blocks)


def clear_cache() -> None:
    """Drop the schema cache. Only useful in tests."""
    _SCHEMA_CACHE.clear()
