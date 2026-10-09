#!/usr/bin/env bash
# BIRD dev EX, Soft-F1 and R-VES with the official mini_dev scripts (after evaluate.py --split bird_dev)
#   pip install func-timeout psycopg2-binary pymysql      (once; the scripts import these)
#   bash run_bird_eval.sh base dpo grpo rloo
# R-VES runs every correct query 100 times, so it is slow: run it when nothing else uses the machine.
set -e
cd "$(dirname "$0")"
ROOT=$(pwd)
M=eval_repos/mini_dev/evaluation

# the scripts read difficulty from a .jsonl file, BIRD ships dev.json
python - <<'EOF'
import json
rows = json.load(open("data/bird_dev/dev.json"))
with open("data/bird_dev/dev.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "BIRD dev questions")
EOF

run() {  # $1 = tag, $2 = ex / f1 / ves
  echo "=== $1: BIRD $2"
  (cd $M && python -u ./evaluation_$2.py \
      --db_root_path $ROOT/data/bird_dev/dev_databases/ \
      --predicted_sql_path $ROOT/results/$1/bird_dev/predict_dev.json \
      --ground_truth_path $ROOT/data/bird_dev/dev.sql \
      --diff_json_path $ROOT/data/bird_dev/dev.jsonl \
      --num_cpus 8 --meta_time_out 30.0 --sql_dialect SQLite \
      --output_log_path $ROOT/results/$1/bird_dev/official_bird.txt)
}

for TAG in "$@"; do rm -f results/$TAG/bird_dev/official_bird.txt; done
for TAG in "$@"; do run $TAG ex; run $TAG f1; done
for TAG in "$@"; do run $TAG ves; done
echo "done: results/<tag>/bird_dev/official_bird.txt"
