#!/bin/bash
# run_all_1003.sh : our fork e673bd168 vs official main@e53d02171, one arm after the other on GPU 4-7 (arm_compose_1003.py).
# Waits for the official image build (build3.log "== ALL DONE"), removes the temporary egress-proxy container, then runs
# O1, F1, O2, F2 (no MTP) and OM, FM, FM2 (MTP). Only ONE heavy thing runs at a time. A watchdog stops the running arm if the
# host's MemAvailable drops under 25 GiB or production 8031 (GPU 0-3) fails its health check three times in a row, and then
# the chain stops (file ABORT). Production 8031 is never touched.
set -u
W=/mnt/2t/build/cmp1003
H=/mnt/2t/build/cmp1001          # mtp_bench.py
B=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
mkdir -p $W
cd $W
rm -f ABORT

wait_build() {  # waits for the official image build; removes the temporary egress-proxy container; 1 = build not OK
  until grep -qE '^== (ALL DONE|FAILED|ABORTED)' /mnt/2t/build/official-main/build3.log; do sleep 30; done
  docker rm -f sx-build-egress-proxy >/dev/null 2>&1 && echo "egress proxy container removed"
  grep -q '^== ALL DONE' /mnt/2t/build/official-main/build3.log
}
echo "=== start $(date '+%F %T')  (fork arms first; the official image is still being built)"

watch_arm() {  # watch_arm <container> : background; stops the container on memory shortage or production trouble
  local c=$1 bad=0 n=0 m h
  while docker inspect $c >/dev/null 2>&1; do
    m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    if [ "$m" -lt 25 ]; then echo "$(date +%T) STOP $c: MemAvailable ${m} GiB" >> $W/watch.log; docker stop -t 20 $c >/dev/null 2>&1; touch $W/ABORT; return; fi
    if [ $((n % 6)) = 0 ]; then
      h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health)
      if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) $c mem_avail=${m}GiB load=$(cut -d' ' -f1 /proc/loadavg) 8031=$h" >> $W/watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) STOP $c: 8031 health failed" >> $W/watch.log; docker stop -t 20 $c >/dev/null 2>&1; touch $W/ABORT; return; fi
    fi
    n=$((n + 1)); sleep 10
  done
}

bench() {  # bench <arm> <name> <pfx_bench args...>
  local a=$1 name=$2; shift 2
  python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/${a}_${name}.json "$@" > $W/${a}_${name}.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback|WAVE" $W/${a}_${name}.log | sed "s/^/[$a $name] /" | cut -c1-250
}

tests() {  # tests <arm> <std|mtp>
  local a=$1
  echo "=== tests $a $(date +%T)"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  if [ "$2" = std ]; then
    for c in 1 2 4 8 16 24; do
      [ -f $W/ABORT ] && return 1
      [ $c = 24 ] && [ "${a:0:1}" = O ] && continue      # official arms run max-num-seqs 16
      bench $a sweep_c$c --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
    done
    bench $a long --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 2 --seed 2026100103
    bench $a l128k --conc 1 --lengths 128000 --gen 256 --passes 1 --seed 2026100104
  else
    python3 $H/mtp_bench.py 8141 $M $W/${a}_mtp.json 2>&1 | grep mtp_bench | sed "s/^/[$a] /"
    bench $a mtp_c1_8k --conc 1 --lengths 8000 --gen 256 --passes 1 --temperature 0 --seed 2026100101
    bench $a mtp_c4_8k --conc 4 --lengths 8000 --gen 256 --passes 1 --temperature 0 --seed 2026100101
  fi
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
}

keylines() {
  docker logs $1 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished|speculative_config=|Auto-setting|Auto-enabling|KV steady|packed|Asynchronous scheduling" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | cut -c1-210 | head -14
}

arm() {  # arm <arm> <std|mtp>
  local a=$1 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st
  [ -f $W/ABORT ] && { echo "ABORT file present, skipping $a"; return 1; }
  # GPUs 4-7 free and enough host memory (the official PLE guard needs (MemAvailable - 25% MemTotal)/4 >= 12 GiB)
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$a] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  python3 $W/arm_compose_1003.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)  MemAvailable ${m} GiB"
  docker compose -f $W/compose.$a.yaml up -d 2>&1 | tail -1
  watch_arm $n & local wp=$!
  t0=$(date +%s)
  until curl -sf -m 5 localhost:8141/health -o /dev/null; do
    st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null)
    if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 3600 ]; then
      echo "ARM_FAIL $a ($st)"; docker logs --tail 30 $n 2>&1 | cut -c1-250
      docker logs $n > $W/engine_$a.log 2>&1; docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1; kill $wp 2>/dev/null; return 1
    fi
    sleep 15
  done
  echo "[$a] healthy after $(( $(date +%s) - t0 ))s"
  keylines $n | sed "s/^/[$a] /"
  curl -s -m 20 localhost:8141/v1/sm70/acceleration > $W/${a}_accel.json 2>/dev/null; echo "[$a] acceleration report: $(head -c 300 $W/${a}_accel.json | tr '\n' ' ')"
  tests $a $2; local rc=$?
  echo "[$a] errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  kill $wp 2>/dev/null
  return $rc
}

arm F1 std
arm F2 std
arm FM mtp
arm FM2 mtp
if wait_build; then
  echo "=== official image ready $(date +%T)"
  arm O1 std
  arm O2 std
  arm OM mtp || arm OM2 mtp
else
  echo "OFFICIAL_BUILD_NOT_OK $(date +%T)"
fi
echo "RUN_ALL_DONE $(date +%T)"
