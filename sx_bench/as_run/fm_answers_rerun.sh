#!/bin/bash
# fm_answers_rerun.sh : FM arm (our fork, MTP k=4, util 0.87) started again ONLY for the answer check that was lost on the
# first FM run (kvq_check asked for a 32K-word prompt on a 32K-context server). Waits for run_all_1003.sh, so nothing else
# runs on GPU 4-7. Same memory / production-health watchdog as the main chain.
set -u
W=/mnt/2t/build/cmp1003
M=Swift-1.5-Qwen3.8-Flash-Next
until grep -q RUN_ALL_DONE $W/run_all_1003.log; do sleep 30; done
cd $W
rm -f ABORT
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
m=$(awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo)
[ "$m" -lt 100 ] && { echo "MemAvailable ${m} GiB < 100, not starting"; echo "FM_ANSWERS_DONE"; exit 1; }
python3 $W/arm_compose_1003.py FM > /dev/null && docker compose -f $W/compose.FM.yaml config -q || { echo BAD_COMPOSE; echo FM_ANSWERS_DONE; exit 1; }
echo "start FM (answers only) $(date +%T) MemAvailable ${m} GiB"
docker compose -f $W/compose.FM.yaml up -d 2>&1 | tail -1
( bad=0; n=0
  while docker inspect sx-cmp-fm >/dev/null 2>&1; do
    mm=$(awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo)
    if [ "$mm" -lt 25 ]; then echo "$(date +%T) STOP sx-cmp-fm: MemAvailable ${mm} GiB" >> $W/watch.log; docker stop -t 20 sx-cmp-fm >/dev/null 2>&1; break; fi
    if [ $((n % 6)) = 0 ]; then h=$(curl -s -m 8 -o /dev/null -w "%{http_code}" localhost:8031/health); if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) sx-cmp-fm(answers) mem_avail=${mm}GiB 8031=$h" >> $W/watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) STOP sx-cmp-fm: 8031 health failed" >> $W/watch.log; docker stop -t 20 sx-cmp-fm >/dev/null 2>&1; break; fi; fi
    n=$((n+1)); sleep 10
  done ) &
t0=$(date +%s)
until curl -sf -m 5 localhost:8141/health -o /dev/null; do
  st=$(docker inspect -f "{{.State.Status}}" sx-cmp-fm 2>/dev/null)
  if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 3600 ]; then echo "ARM_FAIL FM ($st)"; break; fi
  sleep 15
done
if curl -sf -m 5 localhost:8141/health -o /dev/null; then
  echo "[FM] healthy after $(( $(date +%s) - t0 ))s"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/FM_answers.json 2>&1 | tail -8 | cut -c1-200
fi
docker logs sx-cmp-fm > $W/engine_FM_answers.log 2>&1
docker compose -f $W/compose.FM.yaml down 2>&1 | tail -1
echo "FM_ANSWERS_DONE $(date +%T)"
