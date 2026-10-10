#!/usr/bin/env bash
# Live progress of the official evaluation runs (11.2 Spider, 11.3 BIRD). Updates every 30 s.
#   bash progress_official.sh logs_spider_official5.txt          (Spider)
#   bash progress_official.sh logs_bird.txt bird                  (BIRD)
# Ctrl+C closes this view only; the run itself keeps going.
LOG=${1:-logs_spider_official5.txt}
if [ "$2" = bird ]; then
  NAME="BIRD eval (EX, Soft-F1, R-VES)"; DONE="results/*/bird_dev/official_bird_*_done.txt"; SCRIPT=run_bird_eval.sh
  TOTAL=12   # 4 models x 3 metrics (R-VES is the slow part, it runs last)
else
  NAME="Official Spider eval"; DONE="results/*/spider_*/official_done.txt"; SCRIPT=run_official_eval.sh
  TOTAL=12   # 4 models x 3 Spider splits
fi
while true; do
  clear
  n=$(ls $DONE 2>/dev/null | wc -l)
  bar=$(printf '%*s' $n '' | tr ' ' '#')$(printf '%*s' $((TOTAL - n)) '' | tr ' ' '.')
  echo "$NAME  [$bar]  $n / $TOTAL done      ($(date +%H:%M))"
  echo
  echo "Now on:      $(grep '^=== ' $LOG | tail -1)"
  echo "Last line:   $(tail -1 $LOG | cut -c1-110)"
  echo "Memory:      $(free -g | awk '/Mem/ {print $3 " GB used of " $2 " GB"}')"
  if ! pgrep -f $SCRIPT > /dev/null; then
    echo
    if [ $n -ge $TOTAL ]; then echo "ALL DONE."; else echo "Not running any more but not finished: look at  tail -30 $LOG"; fi
    break
  fi
  sleep 30
done
