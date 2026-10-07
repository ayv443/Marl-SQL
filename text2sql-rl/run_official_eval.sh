#!/usr/bin/env bash
# bash run_official_eval.sh grpo   (after evaluate.py)
set -e
TAG=${1:-base}
TS=eval_repos/test-suite-sql-eval
R=results/$TAG

# gold file in the format the spider script wants
python - <<'EOF'
import json, os
for split in ["spider_dev", "spider_syn", "spider_dk", "spider_realistic"]:
    p = f"data/processed/{split}.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        with open(f"data/processed/{split}_gold.sql", "w") as f:
            f.write("\n".join(" ".join(r["gold_sql"].split()) + "\t" + r["db_id"] for r in rows) + "\n")
EOF

for SPLIT in spider_dev spider_syn spider_dk spider_realistic; do
  [ -f $R/$SPLIT/pred.txt ] || continue
  echo "=== $TAG / $SPLIT : EX (original databases) ==="
  python $TS/evaluation.py --gold data/processed/${SPLIT}_gold.sql --pred $R/$SPLIT/pred.txt \
      --db data/spider_data/database --table data/spider_data/tables.json --etype exec | tee $R/$SPLIT/official_ex.txt
  echo "=== $TAG / $SPLIT : TS (test-suite databases) ==="
  python $TS/evaluation.py --gold data/processed/${SPLIT}_gold.sql --pred $R/$SPLIT/pred.txt \
      --db data/testsuite_databases --table data/spider_data/tables.json --etype exec | tee $R/$SPLIT/official_ts.txt
done

# BIRD: edit the paths in eval_repos/mini_dev/evaluation/run_evaluation.sh to:
#   predicted sql json : results/$TAG/bird_dev/predict_dev.json
#   ground truth sql   : data/bird_dev/dev.sql
#   db root            : data/bird_dev/dev_databases/
#   difficulty json    : data/bird_dev/dev.json
#   dialect            : SQLite
