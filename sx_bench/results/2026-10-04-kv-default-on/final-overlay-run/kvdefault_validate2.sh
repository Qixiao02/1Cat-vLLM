#!/bin/bash
# kvdefault_validate.sh : SX_OPT_KV_STEADY_BUDGET unset (the new default) on the 1003 image with the changed vllm/config/vllm.py and
# vllm/v1/worker/kv_steady_budget.py overlaid. Per lane: baseline (switch 0) at the first util, then the variable UNSET at every util.
# Each trial: start, idle, stress (4x8K + 2x16K at once, then one 32K), nvidia-smi samples every 2 s, answers of the engine log audited.
# One heavy job at a time; watchdog: MemAvailable < 25 GiB or production 8031 failing 3 health checks stops the running trial (file ABORT).
set -u
K=/mnt/2t/build/kvdefault
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main
SP=/usr/local/lib/python3.12/dist-packages
cd $K
rm -f ABORT

watch_kv() {
  local bad=0 n=0 m h
  while [ ! -f $K/WATCH_STOP ]; do
    m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    if [ "$m" -lt 25 ]; then echo "$(date +%T) STOP: MemAvailable ${m} GiB" >> $K/watch.log; docker ps -q --filter name=sx-kvsteady | xargs -r docker stop -t 20 >/dev/null 2>&1; touch $K/ABORT; return; fi
    if [ $((n % 6)) = 0 ]; then
      h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health)
      if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) mem_avail=${m}GiB 8031=$h" >> $K/watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) STOP: 8031 health failed" >> $K/watch.log; docker ps -q --filter name=sx-kvsteady | xargs -r docker stop -t 20 >/dev/null 2>&1; touch $K/ABORT; return; fi
    fi
    n=$((n + 1)); sleep 10
  done
}

rm -f $K/WATCH_STOP
watch_kv & WP=$!
run_lane() {  # run_lane <lane> <utils> <out>
  [ -f $K/ABORT ] && { echo "ABORT present, skipping $1"; return 1; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  local m; m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$1] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  echo "##### lane $1 utils \"$2\"  $(date +%T)  MemAvailable ${m} GiB"
  bash $K/tools/run_on_v100.sh --image $IMG --models-dir /mnt/2t/models --lane "$1" --utils "$2" --switch auto \
    --patch-dir $K/patch --site-packages $SP --out "$3" --cache-dir /opt/shixiang-inference/cache-kvsteady-tp4 2>&1 | cut -c1-250
  echo "##### lane $1 finished $(date +%T) exit ${PIPESTATUS[0]}"
}
run_lane no-MTP "0.90 0.94" $K/out2-nomtp
run_lane MTP "0.87 0.93" $K/out2-mtp
touch $K/WATCH_STOP; kill $WP 2>/dev/null
echo "KVDEFAULT_DONE $(date +%T)"
