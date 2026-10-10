#!/usr/bin/env bash
# Official Spider EX + test-suite accuracy (after evaluate.py)
#   bash run_official_eval.sh grpo                    (all Spider splits)
#   bash run_official_eval.sh grpo spider_dk          (only the splits you name)
# Questions whose GOLD query fails on the database are left out for every model (Spider-DK has
# a few broken gold queries), and predictions slower than 60 s are scored as wrong (the official
# timeout doesn't work); both counts are printed and saved in results/<tag>/<split>/official_kept.txt
# A finished split gets results/<tag>/<split>/official_done.txt and is skipped next time, so after a
# restart you can just run the same command again.
set -e
TAG=${1:-base}
shift || true
SPLITS=${@:-spider_dev spider_syn spider_dk spider_realistic}
TS=eval_repos/test-suite-sql-eval
R=results/$TAG
# test_database has every Spider database (train + dev + test), use it if it's there
DB=data/spider_data/database
[ -d data/spider_data/test_database ] && DB=data/spider_data/test_database

for SPLIT in $SPLITS; do
  [ -f $R/$SPLIT/pred.txt ] || continue
  # finished in an earlier run (e.g. before the space was restarted): skip it
  if [ -f $R/$SPLIT/official_done.txt ]; then
    echo "=== $TAG / $SPLIT : already done, skipped (delete $R/$SPLIT/official_done.txt to redo) ==="
    continue
  fi

  # gold + pred files the spider script wants, without the questions whose gold SQL is broken
  python - "$SPLIT" "$TAG" "$DB" <<'EOF'
import json, sqlite3, sys
split, tag, db = sys.argv[1:4]
rows = [json.loads(l) for l in open(f"data/processed/{split}.jsonl")]
preds = open(f"results/{tag}/{split}/pred.txt").read().splitlines()
per_q = [json.loads(l) for l in open(f"results/{tag}/{split}/per_question.jsonl")]
assert len(rows) == len(preds) == len(per_q), (len(rows), len(preds), len(per_q))
bad = []
for i, r in enumerate(rows):
    try:
        c = sqlite3.connect(f"file:{db}/{r['db_id']}/{r['db_id']}.sqlite?mode=ro", uri=True)
        c.text_factory = lambda b: b.decode(errors="ignore")   # same as the official script and reward.py
        c.execute(r["gold_sql"]).fetchall()
        c.close()
    except Exception:
        bad.append(i)
keep = [i for i in range(len(rows)) if i not in set(bad)]
# the official script's 60 s timeout can't interrupt a running sqlite query, so one very slow
# prediction blocks it for hours. Run each prediction once with a real 60 s limit; ones that
# time out get a query that is wrong straight away (the official script would score them wrong).
from reward import _pool, execute
def slow(i):
    return execute(f"{db}/{rows[i]['db_id']}/{rows[i]['db_id']}.sqlite", preds[i], timeout=60)[1] == "timeout"
timed_out = [i for i, t in zip(keep, _pool.map(slow, keep)) if t]
preds = [("SELECT 'timeout'" if i in set(timed_out) else p) for i, p in enumerate(preds)]
with open(f"data/processed/{split}_gold.sql", "w") as f:
    f.write("\n".join(" ".join(rows[i]["gold_sql"].split()) + "\t" + rows[i]["db_id"] for i in keep) + "\n")
with open(f"results/{tag}/{split}/pred_kept.txt", "w") as f:
    f.write("\n".join(preds[i] for i in keep) + "\n")
our_ex = sum(per_q[i]["correct"] for i in keep) / len(keep)
msg = (f"{split}: {len(keep)} of {len(rows)} questions kept ({len(bad)} with a broken gold query left out), "
       f"{len(timed_out)} predictions over 60 s scored as wrong; our EX on the kept questions: {our_ex:.3f}")
print(msg)
open(f"results/{tag}/{split}/official_kept.txt", "w").write(msg + "\nleft out (index in the split): " + json.dumps(bad)
                                                         + "\nslow predictions (index in the split): " + json.dumps(timed_out) + "\n")
EOF

  TABLE=data/spider_data/tables.json
  # Spider-DK has its own tables.json that includes its 3 extra databases
  [ $SPLIT = spider_dk ] && TABLE=eval_repos/Spider-DK/tables.json
  echo "=== $TAG / $SPLIT : EX (original databases) ==="
  python $TS/evaluation.py --gold data/processed/${SPLIT}_gold.sql --pred $R/$SPLIT/pred_kept.txt \
      --db $DB --table $TABLE --etype exec | tee $R/$SPLIT/official_ex.txt
  if [ $SPLIT = spider_dk ]; then
    echo "=== $TAG / spider_dk : no TS (test-suite databases don't include Spider-DK's extra databases) ==="
    date > $R/$SPLIT/official_done.txt
    continue
  fi
  echo "=== $TAG / $SPLIT : TS (test-suite databases) ==="
  python $TS/evaluation.py --gold data/processed/${SPLIT}_gold.sql --pred $R/$SPLIT/pred_kept.txt \
      --db data/testsuite_databases --table $TABLE --etype exec | tee $R/$SPLIT/official_ts.txt
  date > $R/$SPLIT/official_done.txt
done

# BIRD: bash run_bird_eval.sh base dpo grpo rloo
