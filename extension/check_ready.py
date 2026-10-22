# Checks that the data, saved results and adapters are ready for the multi-turn runs (no GPU needed).
#   python extension/check_ready.py --rloo <rloo checkpoint-600> --grpo <grpo checkpoint-400>
# Prints OK / PROBLEM for each check; fix every PROBLEM before starting a long run.
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T2S = os.path.abspath(os.path.join(HERE, "..", "text2sql-rl"))
sys.path.insert(0, T2S)
sys.path.insert(0, HERE)

from common import MAX_PROMPT_LEN, MODEL_NAME, PROCESSED_DIR, SYSTEM_PROMPT, db_full_path, extract_sql, load_jsonl  # noqa: E402
from multiturn_eval import EVAL_DIR, clean_sql, feedback  # noqa: E402
from reward import R_CORRECT, gold_cache, gold_rows_for, has_order_by, score  # noqa: E402

problems = []


def check(ok, msg):
    print(("OK       " if ok else "PROBLEM  ") + msg)
    if not ok:
        problems.append(msg)


def data_checks(tok):
    print("\n== training data ==")
    for f in ["train.jsonl", "tags.json", "gold_cache.pkl", "spider_dev.jsonl", "bird_dev.jsonl"]:
        check(os.path.exists(os.path.join(PROCESSED_DIR, f)), f"{f} exists in {PROCESSED_DIR}")
    if not all(os.path.exists(os.path.join(PROCESSED_DIR, f)) for f in ["train.jsonl", "tags.json", "gold_cache.pkl"]):
        return
    rows = load_jsonl(os.path.join(PROCESSED_DIR, "train.jsonl"))
    tags = json.load(open(os.path.join(PROCESSED_DIR, "tags.json")))
    gold = gold_cache()
    print(f"         train questions: {len(rows)}")
    check(all(k in rows[0] for k in ("prompt", "qid", "db_path", "gold_sql")), "train rows have prompt, qid, db_path, gold_sql")
    plens = {r["qid"]: len(tok.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True)) for r in rows}
    untagged = [r for r in rows if r["qid"] not in tags]
    check(all(plens[r["qid"]] > MAX_PROMPT_LEN for r in untagged),
          f"every train question that fits the prompt limit has a tag ({len(untagged)} untagged, all over "
          f"{MAX_PROMPT_LEN} tokens and dropped, as in the original runs)")
    missing_gold = sum(r["qid"] not in gold for r in rows)
    empty_gold = sum(r["qid"] in gold and not gold[r["qid"]] for r in rows)
    check(missing_gold == 0, f"every train question has a cached gold result (missing: {missing_gold})")
    check(empty_gold == 0, f"no train question has an empty gold result (empty: {empty_gold})")
    dbs = {r["db_path"] for r in rows}
    missing_db = [d for d in dbs if not os.path.exists(db_full_path(d))]
    check(not missing_db, f"all {len(dbs)} training databases exist (missing: {len(missing_db)})")
    check(all(r["prompt"][0]["role"] == "system" and r["prompt"][0]["content"] == SYSTEM_PROMPT for r in rows),
          "every prompt starts with the same system prompt as training")
    used = [r for r in rows if tags.get(r["qid"]) == "mixed"]
    lens = [plens[r["qid"]] for r in used]
    print(f"         questions used (mixed, as in the single-turn runs): {len(used)}, longest prompt {max(lens)} tokens, "
          f"over {MAX_PROMPT_LEN}: {sum(n > MAX_PROMPT_LEN for n in lens)} (these are dropped)")
    worst = max(lens[i] for i in range(len(lens)) if lens[i] <= MAX_PROMPT_LEN)
    # prompt + 3 answers of 256 tokens + 2 feedback messages (at most ~250 tokens each)
    check(worst + 3 * 256 + 2 * 250 <= 3584, f"longest kept episode fits in max_episode_len 3584 ({worst + 3 * 256 + 2 * 250})")

    # gold queries must score as correct against their own cached result
    random.seed(0)
    usable = [r for r in rows if gold.get(r["qid"]) and os.path.exists(db_full_path(r["db_path"]))]
    sample = random.sample(usable, min(200, len(usable)))
    ok = sum(score(db_full_path(r["db_path"]), r["gold_sql"], gold[r["qid"]], has_order_by(r["gold_sql"]))[0] == R_CORRECT
             for r in sample)
    check(ok == len(sample), f"gold SQL scores as correct on {len(sample)} random train questions ({ok} / {len(sample)})")

    # feedback never contains the gold SQL
    r = sample[0]
    _, msg = feedback(r["gold_sql"], db_full_path(r["db_path"]))
    check(r["gold_sql"] not in msg, "feedback message does not contain the gold SQL")
    print("         example feedback:", msg[:200].replace("\n", " | "))


def eval_checks():
    print("\n== saved single-turn results (turn 1 of the multi-turn evaluation) ==")
    for split in ["spider_dev", "bird_dev"]:
        if not os.path.exists(os.path.join(PROCESSED_DIR, f"{split}.jsonl")):
            check(False, f"{split}.jsonl exists")
            continue
        rows = load_jsonl(os.path.join(PROCESSED_DIR, f"{split}.jsonl"))
        qids = {r["qid"] for r in rows}
        sample = random.Random(1).sample(rows, 150)
        golds = gold_rows_for(sample)
        for tag in ["base", "dpo", "grpo", "rloo"]:
            p = os.path.join(EVAL_DIR, tag, split, "per_question.jsonl")
            if not os.path.exists(p):
                check(False, f"{tag}/{split}: saved results exist")
                continue
            saved = {r["qid"]: r for r in load_jsonl(p)}
            check(qids <= set(saved), f"{tag}/{split}: saved answers for all {len(qids)} questions")
            check("raw_output" in next(iter(saved.values())), f"{tag}/{split}: raw model output is saved")
            same_sql = sum(extract_sql(saved[r["qid"]]["raw_output"]) == saved[r["qid"]]["pred_sql"] for r in sample)
            same_score = sum((score(db_full_path(r["db_path"]), saved[r["qid"]]["pred_sql"], g, has_order_by(r["gold_sql"]))[0]
                              == R_CORRECT) == bool(saved[r["qid"]]["correct"]) for r, g in zip(sample, golds))
            check(same_sql == 150 and same_score >= 148,
                  f"{tag}/{split}: re-reading and re-scoring the saved answers gives the same result "
                  f"(SQL {same_sql}/150, score {same_score}/150)")


def adapter_checks(name, path):
    print(f"\n== {name} adapter ==")
    if not path:
        print("         (not given, skipped)")
        return
    check(os.path.exists(os.path.join(path, "adapter_model.safetensors")), f"{path} has adapter_model.safetensors")
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        check(False, f"{path} has adapter_config.json")
        return
    cfg = json.load(open(cfg_path))
    check(cfg.get("r") == 16 and cfg.get("lora_alpha") == 32, f"LoRA r=16, alpha=32 (found r={cfg.get('r')}, alpha={cfg.get('lora_alpha')})")
    check(MODEL_NAME in str(cfg.get("base_model_name_or_path")), f"base model is {MODEL_NAME} (found {cfg.get('base_model_name_or_path')})")
    want = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    check(want <= set(cfg.get("target_modules") or []), "LoRA covers all 7 projections")


def clean_checks():
    print("\n== reading answers after feedback ==")
    cases = {"(SELECT 1)": "SELECT 1", "Yes": None, "SELECT a FROM t;": "SELECT a FROM t", None: None}
    check(all(clean_sql(k) == v for k, v in cases.items()), "brackets, ';' and non-SQL answers are handled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rloo", default=None)
    ap.add_argument("--grpo", default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_checks(tok)
    eval_checks()
    adapter_checks("RLOO", args.rloo)
    adapter_checks("GRPO", args.grpo)
    clean_checks()
    print("\n" + ("ALL CHECKS PASSED" if not problems else f"{len(problems)} PROBLEM(S):\n  - " + "\n  - ".join(problems)))


if __name__ == "__main__":
    main()
