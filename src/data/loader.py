"""Loading the processed examples.

Thin on purpose. The one thing it has to get right is that the training subset is
identical across conditions: invariant 2 says the credit method is the only free
variable, and "which 2000 examples" is a shared hyperparameter like any other.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

PROCESSED_DIR = os.path.join("data", "processed")


@dataclass
class Example:
    id: str
    db_id: str
    question: str
    gold_sql: str
    gold_result_hash: str
    gold_row_count: int
    order_matters: bool
    difficulty: str
    n_tables_in_gold: int


FIELDS = (
    "id",
    "db_id",
    "question",
    "gold_sql",
    "gold_result_hash",
    "gold_row_count",
    "order_matters",
    "difficulty",
    "n_tables_in_gold",
)


def _to_example(record: dict) -> Example:
    """Build an Example, ignoring any extra keys a future preprocessing pass adds."""
    return Example(**{key: record[key] for key in FIELDS})


def load_examples(
    path: str,
    n: int | None = None,
    seed: int = 42,
) -> list[Example]:
    """Read the filtered JSONL and return at most n examples.

    Shuffle first with the seed, then truncate. Doing it in that order is what makes
    n_examples=2000 select the same 2000 examples for every condition; truncating first
    would make the subset depend on file order and quietly break the control.
    """
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)

    examples = [_to_example(record) for record in records]

    if n is not None and n < len(examples):
        examples = examples[:n]

    rng = random.Random(seed)
    rng.shuffle(examples)

    return examples


def load_train(n: int | None = None, seed: int = 42) -> list[Example]:
    return load_examples(
        os.path.join(PROCESSED_DIR, "spider_train_filtered.jsonl"), n=n, seed=seed
    )


def load_dev(n: int | None = None) -> list[Example]:
    """Dev examples, in file order.

    No shuffling and no seed: evaluation is deterministic and must be reproducible
    independent of training configuration. Never train or tune on this split.
    """
    path = os.path.join(PROCESSED_DIR, "spider_dev.jsonl")
    with open(path, encoding="utf-8") as fh:
        examples = [_to_example(json.loads(line)) for line in fh if line.strip()]
    if n is not None:
        examples = examples[:n]
    return examples
