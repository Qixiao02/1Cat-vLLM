#!/bin/bash
# zz_cut_round5.sh - cut round 5 (1004ldmt2) of the load-flag chain.
# Decision: user picked "only cut round 5" (m02869 option A) - keep 1004ldctl2 as the
# drift control, drop the second multithread repeat to reach parity ~17 min earlier.
# Round 4 (1004ldctl2) must finish normally first; we only kill once round 5 has started.
W=/mnt/2t/build/cmp1003
cd "$W" || exit 1
echo "=== CUT watcher start $(date +%T)"
seen=0
for i in $(seq 1 1200); do
  if grep -q "load round 1004ldmt2" "$W/startup_ab_load.log" 2>/dev/null; then
    seen=1
    break
  fi
  sleep 10
done
if [ "$seen" != "1" ]; then
  echo "=== CUT_MARKER_NOT_SEEN (chain may already be over) $(date +%T)"
  exit 0
fi
echo "=== round 5 detected at $(date +%T); cutting"
echo "--- round 4 result (kept as drift control)"
grep -E "load round 1004ldctl2|Loading weights took" "$W/startup_ab_load.log" | tail -6
pkill -f 'run_1004ldmt2\.sh'
pkill -f 'tee run_1004ldmt2\.log'
sleep 3
echo "--- survivors: $(pgrep -af 'run_1004ldmt2' | wc -l)"
docker compose -f "$W/compose.PA.yaml" down --remove-orphans >/dev/null 2>&1
docker rm -f sx-cmp-pa >/dev/null 2>&1
for i in $(seq 1 60); do
  n=$(docker ps --format '{{.Names}}' | grep -c '^sx-cmp-pa$')
  [ "$n" = "0" ] && break
  sleep 5
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=4 {gsub(/ /,"",$2); if (($2+0)>1500) c++} END {print c+0}')
echo "=== container sx-cmp-pa running: $(docker ps --format '{{.Names}}' | grep -c '^sx-cmp-pa$') ; GPU4-7 busy: $busy"
echo "=== CUT_ROUND5_DONE $(date +%T)"
