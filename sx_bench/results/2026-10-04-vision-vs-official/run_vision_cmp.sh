#!/bin/bash
# run_vision_cmp.sh : the comparison "fork 1004 vs official v1.5.1" again, this time BOTH SIDES WITH THE VISION TOWER LOADED
# (--language-model-only dropped, same --limit-mm-per-prompt / --mm-processor-cache-gb on both; arms of add_arms_vision.py), GPU 4-7.
# Same tests as the first comparison (answers, 8K greedy sweep, 4 x 8K..64K, 128K, MTP bench) plus image requests (image_check.py,
# image_bench.py). Fork = the 1004 validation image, official = the v1.5.1 wheel on Ubuntu 24.04 (official-v1.5.1-ubuntu2404-sm70main).
# Pairs run next to each other: FV1/VV2b (prefix on), FV2/VV1 (prefix off), FVM/VVMa (MTP). A failed arm does not stop the chain.
# One heavy job at a time; watchdog from arm_functions.sh (MemAvailable < 25 GiB or production 8031 down stops the arm).
set -u
W=/mnt/2t/build/cmp1003
H=/mnt/2t/build/cmp1001          # mtp_bench.py
B=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
export FORK_IMAGE=shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main
export OFFICIAL_IMAGE=shixiang/1cat-vllm-v100:official-v1.5.1-ubuntu2404-sm70main
. $W/arm_functions.sh
cd $W
rm -f ABORT
python3 $W/add_arms_vision.py

bench() {  # bench <arm> <name> <pfx_bench args...>
  local a=$1 name=$2; shift 2
  python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/${a}_${name}.json "$@" > $W/${a}_${name}.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback|WAVE" $W/${a}_${name}.log | sed "s/^/[$a $name] /" | cut -c1-250
}

tests() {  # tests <arm> <std|mtp>
  local a=$1
  echo "=== tests $a $(date +%T)"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  python3 $W/image_check.py 8141 $M 2>&1 | sed "s/^/[$a] /"
  python3 $W/image_bench.py 8141 $M $W/${a}_img.json 2>&1 | sed "s/^/[$a] /"
  if [ "$2" = std ]; then
    for c in 1 2 4 8 16 24; do
      [ -f $W/ABORT ] && return 1
      [ $c = 24 ] && [ "${a:0:2}" = VV ] && continue      # official arms run max-num-seqs 16
      bench $a sweep_c$c --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
    done
    bench $a long --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 2 --seed 2026100103
    bench $a l128k --conc 1 --lengths 128000 --gen 256 --passes 1 --seed 2026100104
  else
    python3 $H/mtp_bench.py 8141 $M $W/${a}_mtp.json 2>&1 | grep mtp_bench | sed "s/^/[$a] /"
    bench $a mtp_c1_8k --conc 1 --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
    bench $a mtp_c4_8k --conc 4 --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
  fi
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
}

keylines() {
  docker logs $1 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished|speculative_config=|Auto-setting VLLM_SM70_QWEN38|Auto-enabling|KV steady budget:|packed|Asynchronous scheduling|multimodal|encoder cache|language_model_only|OutOfMemory|out of memory" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | cut -c1-210 | head -16
}

arm() {  # arm <arm> <std|mtp>
  local a=$1 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st
  [ -f $W/ABORT ] && { echo "ABORT file present, skipping $a"; return 1; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$a] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  python3 $W/arm_compose_1003.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)  MemAvailable ${m} GiB  image $(grep -m1 'image:' $W/compose.$a.yaml | sed 's/.*image: //')"
  echo "[$a] flags: $(sed -n '/command:/,$p' $W/compose.$a.yaml | grep -E "^      - " | sed "s/^      - //; s/'//g" | tail -n +2 | tr '\n' ' ' | cut -c1-760)"
  docker compose -f $W/compose.$a.yaml up -d 2>&1 | tail -1
  watch_arm $n & local wp=$!
  t0=$(date +%s)
  until curl -sf -m 5 localhost:8141/health -o /dev/null; do
    st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null)
    if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 3600 ]; then
      echo "ARM_FAIL $a ($st) after $(( $(date +%s) - t0 ))s"; docker logs --tail 30 $n 2>&1 | cut -c1-250
      docker logs $n > $W/engine_$a.log 2>&1; docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1; docker rm -f $n >/dev/null 2>&1; kill $wp 2>/dev/null; return 1
    fi
    sleep 15
  done
  echo "[$a] healthy after $(( $(date +%s) - t0 ))s; version $(curl -s -m 5 localhost:8141/version)"
  keylines $n | sed "s/^/[$a] /"
  curl -s -m 20 localhost:8141/v1/sm70/acceleration > $W/${a}_accel.json 2>/dev/null; echo "[$a] acceleration report: $(head -c 300 $W/${a}_accel.json | tr '\n' ' ')"
  tests $a $2; local rc=$?
  echo "[$a] errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')  OOM lines: $(docker logs $n 2>&1 | grep -cE 'OutOfMemory|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  docker rm -f $n >/dev/null 2>&1
  kill $wp 2>/dev/null
  return $rc
}

arm FV1 std
arm VV2b std
arm FV2 std
arm VV1 std
arm FVM mtp
arm VVMa mtp
echo "VISION_CMP_DONE $(date +%T)"
