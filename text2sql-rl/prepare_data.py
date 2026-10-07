# Run once after download_data.sh. Writes everything to data/processed/
import json
import os
import pickle
import random
from collections import Counter

from tqdm import tqdm

from common import DATA_DIR, PROCESSED_DIR, build_prompt, db_full_path, save_jsonl
from reward import _pool, execute
from monitor import Monitor

SEED = 42
VAL_TARGET = 400

SPIDER = "spider_data"
BIRD = "bird_dev"


def find_db(db_id, roots=(f"{SPIDER}/database", f"{SPIDER}/test_database")):
    for root in roots:
        rel = f"{root}/{db_id}/{db_id}.sqlite"
        if os.path.exists(os.path.join(DATA_DIR, rel)):
            return rel
    return None


def drop_missing_dbs(rows, name):
    missing = sorted({r["db_id"] for r in rows if r["db_path"] is None})
    if missing:
        n = sum(r["db_path"] is None for r in rows)
        print(f"WARNING {name}: skipped {n} questions, database not found: {missing}")
    return [r for r in rows if r["db_path"] is not None]


def spider_row(qid, ex):
    db_path = find_db(ex["db_id"])
    question = ex.get("SpiderSynQuestion") or ex["question"]
    return {"qid": qid, "db_id": ex["db_id"], "db_path": db_path, "question": question,
            "evidence": "", "gold_sql": ex["query"], "difficulty": None}


def add_prompts(rows):
    for r in tqdm(rows, desc="building prompts"):
        r["prompt"] = build_prompt(db_full_path(r["db_path"]), r["question"], r["evidence"])
    return rows


def load_json(rel):
    with open(os.path.join(DATA_DIR, rel)) as f:
        return json.load(f)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # filter train: drop gold errors, timeouts and empty results
    raw = load_json(f"{SPIDER}/train_spider.json") + load_json(f"{SPIDER}/train_others.json")
    rows = drop_missing_dbs([spider_row(f"train_{i}", ex) for i, ex in enumerate(raw)], "train")
    print(f"Spider train: {len(rows)} questions")

    results = list(tqdm(_pool.map(lambda r: execute(db_full_path(r["db_path"]), r["gold_sql"]), rows),
                        total=len(rows), desc="running gold SQL"))
    kept, gold_cache, reasons = [], {}, Counter()
    for r, (res, err) in zip(rows, results):
        if err == "timeout":
            reasons["timeout_over_5s"] += 1
        elif err is not None:
            reasons["gold_error"] += 1
        elif len(res) == 0:
            reasons["empty_result"] += 1
        else:
            kept.append(r)
            gold_cache[r["qid"]] = res
    reasons["kept"] = len(kept)
    print("filter:", dict(reasons))

    # hold out whole databases for validation
    dbs = sorted({r["db_id"] for r in kept})
    random.Random(SEED).shuffle(dbs)
    per_db = Counter(r["db_id"] for r in kept)
    val_dbs, n = [], 0
    for db in dbs:
        if n >= VAL_TARGET:
            break
        if per_db[db] > 80:
            continue
        val_dbs.append(db)
        n += per_db[db]
    val = [r for r in kept if r["db_id"] in val_dbs]
    train = [r for r in kept if r["db_id"] not in val_dbs]
    print(f"train: {len(train)} questions | val: {len(val)} questions from {len(val_dbs)} DBs")

    save_jsonl(add_prompts(train), f"{PROCESSED_DIR}/train.jsonl")
    save_jsonl(add_prompts(val), f"{PROCESSED_DIR}/val.jsonl")
    with open(f"{PROCESSED_DIR}/gold_cache.pkl", "wb") as f:
        pickle.dump(gold_cache, f)
    with open(f"{PROCESSED_DIR}/filter_log.json", "w") as f:
        json.dump({**reasons, "train": len(train), "val": len(val), "val_dbs": val_dbs}, f, indent=2)

    # eval sets, not filtered
    dev = drop_missing_dbs([spider_row(f"spider_dev_{i}", ex)
                            for i, ex in enumerate(load_json(f"{SPIDER}/dev.json"))], "spider_dev")
    save_jsonl(add_prompts(dev), f"{PROCESSED_DIR}/spider_dev.jsonl")

    variants = {"spider_syn": "spider_variants/spider_syn.json",
                "spider_dk": "spider_variants/spider_dk.json",
                "spider_realistic": "spider_variants/spider_realistic.json"}
    for name, rel in variants.items():
        if not os.path.exists(os.path.join(DATA_DIR, rel)):
            print(f"skipping {name}: {rel} not found")
            continue
        rows = drop_missing_dbs([spider_row(f"{name}_{i}", ex) for i, ex in enumerate(load_json(rel))], name)
        save_jsonl(add_prompts(rows), f"{PROCESSED_DIR}/{name}.jsonl")

    bird = []
    for ex in load_json(f"{BIRD}/dev.json"):
        bird.append({"qid": f"bird_dev_{ex['question_id']}", "db_id": ex["db_id"],
                     "db_path": find_db(ex["db_id"], roots=(f"{BIRD}/dev_databases",)),
                     "question": ex["question"], "evidence": ex.get("evidence", ""),
                     "gold_sql": ex["SQL"], "difficulty": ex.get("difficulty")})
    save_jsonl(add_prompts(drop_missing_dbs(bird, "bird_dev")), f"{PROCESSED_DIR}/bird_dev.jsonl")
    print("done ->", PROCESSED_DIR)
    return {**reasons, "train": len(train), "val": len(val)}


if __name__ == "__main__":
    with Monitor("prepare-data") as mon:
        mon.result = main()
