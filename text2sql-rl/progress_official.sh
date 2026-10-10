#!/usr/bin/env bash
# Live progress of the official Spider run (11.2). Updates every 30 s.
#   bash progress_official.sh                       (reads logs_spider_official5.txt)
#   bash progress_official.sh logs_other.txt
# Ctrl+C closes this view only; the run itself keeps going.
LOG=${1:-logs_spider_official5.txt}
TOTAL=12   # 4 models x 3 Spider splits
while true; do
  clear
  n=$(ls results/*/spider_*/official_done.txt 2>/dev/null | wc -l)
  bar=$(printf '%*s' $n '' | tr ' ' '#')$(printf '%*s' $((TOTAL - n)) '' | tr ' ' '.')
  echo "Official Spider eval  [$bar]  $n / $TOTAL done      ($(date +%H:%M))"
  echo
  echo "Now on:      $(grep '^=== ' $LOG | tail -1)"
  echo "Last line:   $(tail -1 $LOG | cut -c1-110)"
  echo "Memory:      $(free -g | awk '/Mem/ {print $3 " GB used of " $2 " GB"}')"
  if ! pgrep -f run_official_eval.sh > /dev/null; then
    echo
    if [ $n -ge $TOTAL ]; then echo "ALL DONE."; else echo "Not running any more but not finished: look at  tail -30 $LOG"; fi
    break
  fi
  sleep 30
done
