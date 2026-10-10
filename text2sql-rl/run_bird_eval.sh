#!/usr/bin/env bash
# BIRD dev EX, Soft-F1 and R-VES with the official mini_dev scripts (after evaluate.py --split bird_dev)
#   pip install func-timeout psycopg2-binary pymysql      (once; the scripts import these)
#   bash run_bird_eval.sh base dpo grpo rloo
# R-VES runs every correct query 100 times, so it is slow: run it when nothing else uses the machine.
# Each finished metric gets results/<tag>/bird_dev/official_bird_<metric>_done.txt and is skipped next
# time, so after a restart just run the same command again. To redo everything:
#   rm -f results/*/bird_dev/official_bird* results/*/bird_dev/predict_dev_checked.json
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
M=eval_repos/mini_dev/evaluation
# the official scripts run 8 processes at once; cap each at ~3.5 GB so a huge query stops with an
# error instead of freezing the whole space (like the Spider run did)
ulimit -v 3500000 2>/dev/null || echo "note: could not set the memory cap, carrying on without it"

# the scripts read difficulty from a .jsonl file, BIRD ships dev.json
python - <<'PY'
import json
rows = json.load(open("data/bird_dev/dev.json"))
with open("data/bird_dev/dev.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "BIRD dev questions")
PY

# Same idea as run_official_eval.sh: the official scripts' timeout can't interrupt a running sqlite
# query and they load the whole result into memory. Run each prediction once with a real 30 s limit
# (the official meta_time_out); ones that time out or return 100,000+ rows (when the gold answer is
# small) are replaced by a query that is wrong straight away. The official scripts score both as
# wrong anyway. Writes predict_dev_checked.json, which is what gets evaluated.
check() {  # $1 = tag
  [ -f results/$1/bird_dev/predict_dev_checked.json ] && return
  python - "$1" <<'PY'
import json, sys
from reward import _pool, execute, MAX_ROWS
tag = sys.argv[1]
preds = json.load(open(f"results/{tag}/bird_dev/predict_dev.json"))
gold = [l.rstrip("\n").rsplit("\t", 1) for l in open("data/bird_dev/dev.sql") if l.strip()]
assert len(preds) == len(gold), (len(preds), len(gold))
def bad(i):
    sql, db = preds[str(i)].split("\t----- bird -----\t")
    f = f"data/bird_dev/dev_databases/{db}/{db}.sqlite"
    out, err = execute(f, sql, timeout=30)
    if err == "timeout":
        return True
    if out is not None and len(out) >= MAX_ROWS:
        g, _ = execute(f, gold[i][0], timeout=60)
        return g is not None and len(g) < MAX_ROWS
    return False
flagged = [i for i, b in zip(range(len(gold)), _pool.map(bad, range(len(gold)))) if b]
for i in flagged:
    db = preds[str(i)].split("\t----- bird -----\t")[1]
    preds[str(i)] = "SELECT 'timeout'\t----- bird -----\t" + db
json.dump(preds, open(f"results/{tag}/bird_dev/predict_dev_checked.json", "w"), indent=4)
msg = f"{tag} bird_dev: {len(flagged)} of {len(gold)} predictions over 30 s or 100,000+ rows scored as wrong"
print(msg)
open(f"results/{tag}/bird_dev/official_bird_kept.txt", "w").write(msg + "\nindexes: " + json.dumps(flagged) + "\n")
PY
}

run() {  # $1 = tag, $2 = ex / f1 / ves
  DONE=results/$1/bird_dev/official_bird_$2_done.txt
  if [ -f $DONE ]; then echo "=== $1: BIRD $2 already done, skipped (delete $DONE to redo)"; return; fi
  echo "=== $1: BIRD $2"
  (cd $M && python -u ./evaluation_$2.py \
      --db_root_path $ROOT/data/bird_dev/dev_databases/ \
      --predicted_sql_path $ROOT/results/$1/bird_dev/predict_dev_checked.json \
      --ground_truth_path $ROOT/data/bird_dev/dev.sql \
      --diff_json_path $ROOT/data/bird_dev/dev.jsonl \
      --num_cpus 8 --meta_time_out 30.0 --sql_dialect SQLite \
      --output_log_path $ROOT/results/$1/bird_dev/official_bird.txt)
  date > $DONE
}

for TAG in "$@"; do check $TAG; done
for TAG in "$@"; do run $TAG ex; run $TAG f1; done
for TAG in "$@"; do run $TAG ves; done
echo "done: results/<tag>/bird_dev/official_bird.txt"
