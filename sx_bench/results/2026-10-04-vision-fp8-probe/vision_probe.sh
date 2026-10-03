#!/bin/bash
# vision_probe.sh : which launch flag costs speed? One factor at a time against the production flags.
#   F1        production flags of the 8031 service (no MTP, FP16 KV, --language-model-only)
#   T_VISION  the same flags without --language-model-only, plus --limit-mm-per-prompt / --mm-processor-cache-gb 2
#   T_FP8     the same flags as F1 with --kv-cache-dtype fp8_e4m3 (vision still off)
# Both on the 1003 image, GPU 4-7, same harness as the main comparison: 8K input / 256 greedy tokens at C1 and C2 (2 passes), then the
# answers check; T_VISION also gets two image requests (a red and a blue square). One heavy job at a time (starts after PROF_1003_DONE).
# Watchdog: MemAvailable < 25 GiB or 8031 health failing 3x stops the arm (file ABORT).
set -u
W=/mnt/2t/build/cmp1003
B=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
. $W/arm_functions.sh
cd $W
rm -f ABORT
export FORK_IMAGE=shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main

run_arm() {
  local a=$1 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st wp
  [ -f $W/ABORT ] && { echo "ABORT present, skipping $a"; return 1; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$a] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  python3 $W/arm_compose_1003.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)  MemAvailable ${m} GiB"
  echo "[$a] language-model-only in flags: $(grep -c 'language-model-only' $W/compose.$a.yaml)   mm flags: $(grep -cE 'limit-mm|mm-processor' $W/compose.$a.yaml)"
  docker compose -f $W/compose.$a.yaml up -d 2>&1 | tail -1
  watch_arm $n & wp=$!
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
  docker logs $n 2>&1 | grep -E "GPU KV cache size|Model loading took|Using SM70 Flash-V100 0.0.3 compile|dual-compile|multimodal|vision|encoder cache|language_model_only|Auto-" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | sed -E 's/\r/\n/g' | grep -vE "^\s*$" | awk '!s[$0]++' | head -16 | cut -c1-210 | sed "s/^/[$a] /"
  for c in 1 2; do
    python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/${a}_vp_c$c.json --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101 > $W/${a}_vp_c$c.log 2>&1
    grep -E "pass [0-9]|ERROR|Traceback" $W/${a}_vp_c$c.log | sed -E 's/ \| ttft.*prefill +([0-9.]+) tok\/s \| decode +([0-9.]+) tok\/s per stream, +([0-9.]+) agg.*kv peak ([0-9.]+%).*/ | prefill \1 dec \2 agg \3 kv \4/' | sed "s/^/[$a] /"
  done
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/${a}_vp_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  [ "$a" = T_VISION ] && python3 $W/image_check.py 8141 $M 2>&1 | sed "s/^/[$a] /"
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')  errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
  docker logs $n > $W/engine_${a}_vp.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  docker rm -f $n >/dev/null 2>&1
  kill $wp 2>/dev/null
  return 0
}

for a in F1 T_VISION T_FP8; do run_arm $a || break; done
echo "VISION_PROBE_DONE $(date +%T)"
