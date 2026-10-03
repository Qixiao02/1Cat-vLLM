#!/bin/bash
# prof_1003.sh : where does the time of ONE decode step of the production (no MTP) lane go, per concurrency and context length?
# Two instances, one after the other, both on GPU 4-7 (production 8031 on GPU 0-3 is only health-checked):
#   PROF_E  eager + record_shapes : decode-only 3 s windows at C1 / C8 / C24 (200-token prompts), C8 with 8K prompts, C1 with 32K prompt
#   PROF_G  CUDA graphs on        : decode-only 3 s windows at 8K x C1, 8K x C4, 8K x C8, 4K x C16, 2K x C24, 32K x C1
# The profiler starts only after every request has its first token (prof_window.py) and runs for 3 seconds, so a trace holds ~10-20 eager or
# ~100-200 graph steps, never a whole request. Watchdog: MemAvailable < 25 GiB or 8031 health failing 3x stops the instance (file ABORT).
set -u
W=/mnt/2t/build/cmp1003
H=/mnt/2t/build/cmp1001
M=Swift-1.5-Qwen3.8-Flash-Next
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main
CACHE=/opt/shixiang-inference/cache-forkwheel-1003-flashnext-tp4
PROF=$CACHE/prof
. $W/arm_functions.sh
cd $W
rm -f ABORT
export FORK_IMAGE=$IMG
mkdir -p $PROF && chown 997:983 $PROF

run_instance() {  # run_instance <arm> <window list "len:conc ...">
  local a=$1 wins=$2 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st wp
  [ -f $W/ABORT ] && { echo "ABORT present, skipping $a"; return 1; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$a] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  python3 $W/arm_compose_1003.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)  MemAvailable ${m} GiB"
  docker compose -f $W/compose.$a.yaml up -d 2>&1 | tail -1
  watch_arm $n & wp=$!
  t0=$(date +%s)
  until curl -sf -m 5 localhost:8141/health -o /dev/null; do
    st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null)
    if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 2400 ]; then
      echo "ARM_FAIL $a ($st)"; docker logs --tail 25 $n 2>&1 | cut -c1-220; docker logs $n > $W/engine_$a.log 2>&1
      docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1; docker rm -f $n >/dev/null 2>&1; kill $wp 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "[$a] healthy after $(( $(date +%s) - t0 ))s"
  python3 $H/prof_window.py 8141 $M 1 1000 48 1 > /dev/null 2>&1          # warm-up request; its trace is discarded
  curl -s -m 600 -X POST localhost:8141/stop_profile -o /dev/null; sleep 3; rm -rf $PROF/* 2>/dev/null
  for spec in $wins; do
    [ -f $W/ABORT ] && break
    len=${spec%%:*}; c=${spec#*:}
    out=$W/prof_${a}_${len}_c$c
    echo "--- window $a: $len-token prompts x C$c $(date +%T)"
    python3 $H/prof_window.py 8141 $M $c $len 260 3 2>&1 | cut -c1-240
    sleep 8
    rm -rf $out; mkdir -p $out; mv $PROF/* $out/ 2>/dev/null
    echo "    trace size: $(du -sm $out | cut -f1) MB, files: $(ls $out | grep -c rank)"
    r0=$(ls $out/*rank0*.gz 2>/dev/null | head -1)
    if [ -n "$r0" ]; then
      if [ "$a" = PROF_E ]; then python3 $H/analyze_eager.py $r0 "$a $len x C$c" 45 > $out/analysis.txt 2>&1
      else python3 $H/analyze_trace.py $r0 "$a $len x C$c" > $out/analysis.txt 2>&1; fi
      echo "    analysis: $(wc -l < $out/analysis.txt) lines -> $out/analysis.txt"
    fi
  done
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')  errors in engine log: $(docker logs $n 2>&1 | grep -cE 'OutOfMemory|Traceback|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  docker rm -f $n >/dev/null 2>&1
  kill $wp 2>/dev/null
  return 0
}

run_instance PROF_E "200:1 200:8 200:24 8000:8 32000:1"
run_instance PROF_G "8000:1 8000:4 8000:8 4000:16 2000:24 32000:1"
echo "PROF_1003_DONE $(date +%T)"
