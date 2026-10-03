#!/bin/bash
# run_main_final.sh : the MAIN TRACK, one heavy thing at a time, in this order:
#   1. wait for the repro chain's running arm to finish (REPRO_DONE) and for the v1.5.1 image build (BUILD_EXIT 0)
#   2. image pre-flight (glibc symbols of the wheel's .so files resolve, vllm imports)
#   3. official v1.5.1 arms V1 -> V2 -> VM (VM2 if VM fails)           [official release wheel, Ubuntu 24.04 image]
#   4. FM answers-only rerun (the check lost on the first FM run)
# Watchdog (arm_functions.sh) and pre-start checks as in run_all_1003.sh (MemAvailable, production 8031, GPUs free).
set -u
W=/mnt/2t/build/cmp1003
cd $W
IMG=shixiang/1cat-vllm-v100:official-v1.5.1-ubuntu2404-sm70main
until grep -q MAIN_DONE $W/run_main_final2.log; do sleep 20; done
until grep -q "== BUILD_EXIT" /mnt/2t/build/official-v151/build.log; do sleep 20; done
if ! grep -q "== BUILD_EXIT 0" /mnt/2t/build/official-v151/build.log; then echo "V151_BUILD_FAILED"; echo "MAIN_DONE $(date +%T)"; exit 1; fi
echo "=== v1.5.1 image ready $(date +%T)"
mkdir -p /opt/shixiang-inference/cache-official-v151-flashnext-tp4 && chown 997:983 /opt/shixiang-inference/cache-official-v151-flashnext-tp4

# watch_arm
. "$(dirname "$0")/arm_functions.sh"
rm -f $W/ABORT
export OFFICIAL_IMAGE=$IMG
run_std() {  # run_std <arm> : the standard test set of the main comparison (see run_all_1003.sh tests())
  local a=$1
  echo "=== tests $a $(date +%T)"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 Swift-1.5-Qwen3.8-Flash-Next $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  for c in 1 2 4 8 16; do
    [ -f $W/ABORT ] && return 1
    python3 /mnt/2t/build/pfx_ab/pfx_bench.py --port 8141 --model Swift-1.5-Qwen3.8-Flash-Next --gpus 4,5,6,7 --out $W/${a}_sweep_c$c.json --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101 > $W/${a}_sweep_c$c.log 2>&1
    grep -E "pass [0-9]|ERROR|Traceback" $W/${a}_sweep_c$c.log | sed -E 's/ \| ttft.*prefill +([0-9.]+) tok\/s \| decode +([0-9.]+) tok\/s per stream, +([0-9.]+) agg.*kv peak ([0-9.]+%).*/ | prefill \1 dec \2 agg \3 kv \4/' | sed "s/^/[$a sweep_c$c] /" | cut -c1-200
  done
  python3 /mnt/2t/build/pfx_ab/pfx_bench.py --port 8141 --model Swift-1.5-Qwen3.8-Flash-Next --gpus 4,5,6,7 --out $W/${a}_long.json --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 2 --seed 2026100103 > $W/${a}_long.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback" $W/${a}_long.log | sed -E 's/ \| ttft.*prefill +([0-9.]+) tok\/s \| decode +([0-9.]+) tok\/s per stream, +([0-9.]+) agg.*kv peak ([0-9.]+%).*/ | prefill \1 dec \2 agg \3 kv \4/' | sed "s/^/[$a long] /" | cut -c1-200
  python3 /mnt/2t/build/pfx_ab/pfx_bench.py --port 8141 --model Swift-1.5-Qwen3.8-Flash-Next --gpus 4,5,6,7 --out $W/${a}_l128k.json --conc 1 --lengths 128000 --gen 256 --passes 1 --seed 2026100104 > $W/${a}_l128k.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback" $W/${a}_l128k.log | sed -E 's/ \| ttft.*prefill +([0-9.]+) tok\/s \| decode +([0-9.]+) tok\/s per stream, +([0-9.]+) agg.*kv peak ([0-9.]+%).*/ | prefill \1 dec \2 agg \3 kv \4/' | sed "s/^/[$a l128k] /" | cut -c1-200
}
run_mtp() {  # run_mtp <arm> : MTP test set (mtp_bench natural prompts + 8K C1/C4)
  local a=$1
  echo "=== tests $a $(date +%T)"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 Swift-1.5-Qwen3.8-Flash-Next $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  python3 /mnt/2t/build/cmp1001/mtp_bench.py 8141 Swift-1.5-Qwen3.8-Flash-Next $W/${a}_mtp.json 2>&1 | grep mtp_bench | sed "s/^/[$a] /"
  for c in 1 4; do
    python3 /mnt/2t/build/pfx_ab/pfx_bench.py --port 8141 --model Swift-1.5-Qwen3.8-Flash-Next --gpus 4,5,6,7 --out $W/${a}_mtp_c${c}_8k.json --conc $c --lengths 8000 --gen 256 --passes 1 --temperature 0 --seed 2026100101 > $W/${a}_mtp_c${c}_8k.log 2>&1
    grep -E "pass [0-9]|ERROR|Traceback" $W/${a}_mtp_c${c}_8k.log | sed -E 's/ \| ttft.*prefill +([0-9.]+) tok\/s \| decode +([0-9.]+) tok\/s per stream, +([0-9.]+) agg.*kv peak ([0-9.]+%).*/ | prefill \1 dec \2 agg \3 kv \4/' | sed "s/^/[$a mtp_c${c}_8k] /" | cut -c1-200
  done
}
# a run_arm variant that runs a test function instead of the fixed C1/C2 repro cells
run_arm_tests() {  # run_arm_tests <arm> <test function>
  local a=$1 fn=$2 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st
  [ -f $W/ABORT ] && { echo "ABORT present, skipping $a"; return 1; }
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
    if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 2400 ]; then
      echo "ARM_FAIL $a ($st) after $(( $(date +%s) - t0 ))s"; docker logs --tail 25 $n 2>&1 | cut -c1-220
      docker logs $n > $W/engine_$a.log 2>&1; docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1; docker rm -f $n >/dev/null 2>&1; kill $wp 2>/dev/null; return 1
    fi
    sleep 10
  done
  echo "[$a] healthy after $(( $(date +%s) - t0 ))s"
  docker logs $n 2>&1 | grep -E "GPU KV cache size|Model loading took|Graph capturing finished|Maximum concurrency|Auto-setting|Asynchronous scheduling|speculative_config=" | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | head -10 | cut -c1-200 | sed "s/^/[$a] /"
  curl -s -m 20 localhost:8141/v1/sm70/acceleration > $W/${a}_accel.json 2>/dev/null; echo "[$a] acceleration report: $(head -c 400 $W/${a}_accel.json | tr '\n' ' ')"
  $fn $a; local rc=$?
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')  errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  docker rm -f $n >/dev/null 2>&1
  kill $wp 2>/dev/null
  return $rc
}

run_arm_tests V2b run_std
echo "V2B_DONE $(date +%T)"
