# Multi-turn evaluation: the model sees what happened when its SQL was run and can fix it.
#
# Turn 1 is the normal answer. After each turn the environment runs the SQL on the database and sends
# back the error message, or the number of rows and the first rows. The model can then keep the query
# or write a new one, for up to --turns turns. The gold SQL is never shown to the model; it is only used
# afterwards to score each turn.
#
# Modes:
#   errors : only answers that fail to run (SQL error, no SQL, timeout) get feedback and another turn
#   all    : every answer gets its result back; a question stops when the model repeats its query
#
# Run from the repository root (needs the GPU, the data in text2sql-rl/data and the adapters):
#   python extension/multiturn_eval.py --tag rloo --adapter <path to checkpoint-600> --split spider_dev --mode all
#   python extension/multiturn_eval.py --summary          (table of every finished run)
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T2S = os.path.abspath(os.path.join(HERE, "..", "text2sql-rl"))
sys.path.insert(0, T2S)

from common import PROCESSED_DIR, db_full_path, extract_sql, load_jsonl, save_jsonl  # noqa: E402
from reward import R_CORRECT, execute, gold_rows_for, has_order_by, score  # noqa: E402

EVAL_DIR = os.path.join(T2S, "evaluation", "results")
OUT_DIR = os.path.join(HERE, "findings", "multiturn")
ANSWER_RULE = "Put only the SQL inside <answer></answer> tags."


def feedback(sql, db_file):
    """What the environment tells the model about its last query. Never uses the gold SQL."""
    if not sql:
        return False, f"I could not find a SQL query in your answer. Write one SQLite query. {ANSWER_RULE}"
    rows, err = execute(db_file, sql)
    if err is not None:
        return False, (f"Running your query gave this error: {err}\n"
                       f"Fix the query so that it runs and answers the question. {ANSWER_RULE}")
    shown = str(rows[:5])
    if len(shown) > 600:
        shown = shown[:600] + " ..."
    return True, (f"Your query ran and returned {len(rows)} row{'s' if len(rows) != 1 else ''}. "
                  f"The first rows are: {shown}\n"
                  f"Check that this result answers the question. If it does, write the same query again. "
                  f"If it does not, write a corrected query. {ANSWER_RULE}")


def clean_sql(sql):
    """For answers after feedback: drop a trailing ';' and brackets around the whole query, and treat an
    answer that is not a query at all (for example 'Yes') as no new SQL."""
    if not sql:
        return None
    s = sql.strip().rstrip(";").strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for k, ch in enumerate(s):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and k < len(s) - 1:
                break
        else:
            s = s[1:-1].strip()
            continue
        break
    return s if s.lower().startswith(("select", "with")) else None


def same_query(a, b):
    return " ".join((a or "").lower().split()).rstrip(";") == " ".join((b or "").lower().split()).rstrip(";")


def turn_one(args, rows):
    """Turn 1 = the saved evaluation answer (same greedy decoding), or a fresh one with --regenerate."""
    saved = os.path.join(args.eval_dir, args.tag, args.split, "per_question.jsonl")
    if os.path.exists(saved) and not args.regenerate:
        by_qid = {r["qid"]: r for r in load_jsonl(saved)}
        if all(r["qid"] in by_qid for r in rows):
            print(f"turn 1: reusing the saved answers from {saved}")
            return ([by_qid[r["qid"]]["raw_output"] for r in rows],
                    [bool(by_qid[r["qid"]]["correct"]) for r in rows], None)
    print("turn 1: generating")
    gen = make_generator(args)
    return [o[0] for o in gen.generate([r["prompt"] for r in rows], n=1, temperature=0.0)], None, gen


def make_generator(args):
    from generation import Generator
    return Generator(args.adapter, args.engine)


def run(args):
    rows = load_jsonl(os.path.join(PROCESSED_DIR, f"{args.split}.jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    golds = gold_rows_for(rows)
    db_files = [db_full_path(r["db_path"]) for r in rows]
    ordered = [has_order_by(r["gold_sql"]) for r in rows]

    first, saved_correct, gen = turn_one(args, rows)
    convs = [list(r["prompt"]) + [{"role": "assistant", "content": out}] for r, out in zip(rows, first)]
    sqls = [[extract_sql(out)] for out in first]
    active = list(range(len(rows)))
    t0 = time.time()

    for turn in range(2, args.turns + 1):
        todo = []
        for i in active:
            ran_ok, msg = feedback(sqls[i][-1], db_files[i])
            if args.mode == "errors" and ran_ok:
                continue  # only failed queries get another turn
            convs[i].append({"role": "user", "content": msg})
            todo.append(i)
        if not todo:
            break
        if gen is None:
            gen = make_generator(args)
        print(f"turn {turn}: {len(todo)} questions")
        outs = gen.generate([convs[i] for i in todo], n=1, temperature=0.0)
        active = []
        for i, o in zip(todo, outs):
            text = o[0]
            convs[i].append({"role": "assistant", "content": text})
            new_sql = clean_sql(extract_sql(text)) or sqls[i][-1]  # no new SQL after feedback = keep the previous query
            repeated = same_query(new_sql, sqls[i][-1])
            sqls[i].append(new_sql)
            if not repeated:
                active.append(i)  # it changed its answer, so it gets feedback again next turn

    # score every turn against the gold result (the model never saw it)
    per_q, by_turn = [], [0] * args.turns
    for i, r in enumerate(rows):
        # a query is scored once; a turn that repeats the turn-1 query reuses the saved turn-1 result, so
        # turn 1 matches the single-turn evaluation exactly and slow (BIRD) queries near the 5 s limit
        # cannot show up as fake fixes or breaks
        saved_first = saved_correct[i] if saved_correct is not None else None
        cache = {}
        correct = []
        for k, s in enumerate(sqls[i]):
            key = " ".join((s or "").lower().split()).rstrip(";")
            if key not in cache:
                if saved_first is not None and same_query(s, sqls[i][0]):
                    cache[key] = saved_first
                else:
                    cache[key] = score(db_files[i], s, golds[i], ordered[i])[0] == R_CORRECT
            correct.append(cache[key])
        for t in range(args.turns):
            by_turn[t] += correct[min(t, len(correct) - 1)]
        per_q.append({"qid": r["qid"], "db_id": r["db_id"], "question": r["question"], "gold_sql": r["gold_sql"],
                      "sqls": sqls[i], "correct_per_turn": correct, "turns_used": len(sqls[i]),
                      "conversation": convs[i][len(r["prompt"]):]})
    n = len(rows)
    first_ok = [q["correct_per_turn"][0] for q in per_q]
    last_ok = [q["correct_per_turn"][-1] for q in per_q]
    metrics = {
        "tag": args.tag, "split": args.split, "mode": args.mode, "turns": args.turns, "n": n,
        "EX_per_turn": [c / n for c in by_turn],
        "fixed": sum(1 for a, b in zip(first_ok, last_ok) if not a and b),
        "broken": sum(1 for a, b in zip(first_ok, last_ok) if a and not b),
        "mean_turns": sum(q["turns_used"] for q in per_q) / n,
        "seconds": round(time.time() - t0, 1),
    }
    out = os.path.join(OUT_DIR, args.tag, args.split, args.mode)
    os.makedirs(out, exist_ok=True)
    save_jsonl(per_q, os.path.join(out, "per_question.jsonl"))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def summary():
    lines = ["# Multi-turn evaluation", "",
             "EX after each turn (%). Fixed = wrong at turn 1, right at the end. "
             "Broken = right at turn 1, wrong at the end.", "",
             "| Benchmark | Mode | Model | Questions | Turn 1 | Turn 2 | Turn 3 | Fixed | Broken | Mean turns |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    found = []
    for root, _, files in os.walk(OUT_DIR):
        if "metrics.json" in files:
            found.append(json.load(open(os.path.join(root, "metrics.json"))))
    order = {"base": 0, "dpo": 1, "grpo": 2, "rloo": 3}
    for m in sorted(found, key=lambda m: (m["split"], m["mode"], order.get(m["tag"], 9))):
        ex = [f"{100 * v:.1f}" for v in m["EX_per_turn"]] + ["", ""]
        lines.append(f"| {m['split']} | {m['mode']} | {m['tag']} | {m['n']} | {ex[0]} | {ex[1]} | {ex[2]} | "
                     f"{m['fixed']} | {m['broken']} | {m['mean_turns']:.2f} |")
    text = "\n".join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="base, dpo, grpo, rloo, or a new name such as mt-rloo (then turn 1 is generated)")
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint folder (none for base)")
    ap.add_argument("--split", default="spider_dev", choices=["spider_dev", "spider_syn", "spider_dk", "bird_dev", "val"])
    ap.add_argument("--mode", default="all", choices=["errors", "all"])
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (for a quick test)")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--eval_dir", default=EVAL_DIR)
    ap.add_argument("--regenerate", action="store_true", help="generate turn 1 again instead of reusing it")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return
    if args.tag is None:
        ap.error("--tag is required")
    if args.tag != "base" and not args.adapter:
        ap.error("--adapter is required for dpo, grpo and rloo")
    run(args)


if __name__ == "__main__":
    main()
