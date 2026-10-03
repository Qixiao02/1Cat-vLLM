#!/bin/bash
# repro_theirs.sh : a third party runs our fork with the launch below and cannot reproduce our prefill / decode speed.
# THEIRS  = their exact flags and env on our fork image (e673bd168), our harness: FP8 E4M3 KV, --compilation-config
#           cudagraph_mode=piecewise, --max-num-seqs 2, --max-model-len 262144, no --language-model-only (vision tower loaded,
#           --limit-mm-per-prompt, --mm-processor-cache-gb 2), --quantization modelopt_fp4, thinking on by default
# Then one factor put back at a time, to see which one costs what:
#   T_FULL  THEIRS with FULL cuda graphs again (no --compilation-config)
#   T_FP16  THEIRS with FP16 KV (--kv-cache-dtype auto)
#   T_LM    THEIRS with --language-model-only (vision tower not loaded)
# Per arm: startup time and KV capacity, which graph modes were captured, then 8K input / 256 greedy tokens at C1 and C2
# (2 passes each, the same prompts and seed as the main comparison), then the answers check. One heavy thing at a time;
# watchdog: MemAvailable < 25 GiB or production 8031 failing 3 health checks stops the arm and the chain.
set -u
W=/mnt/2t/build/cmp1003
B=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
cd $W
rm -f ABORT

watch_arm() {
  local c=$1 bad=0 n=0 m h
  while docker inspect $c >/dev/null 2>&1; do
    m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    if [ "$m" -lt 25 ]; then echo "$(date +%T) STOP $c: MemAvailable ${m} GiB" >> $W/watch.log; docker stop -t 20 $c >/dev/null 2>&1; touch $W/ABORT; return; fi
    if [ $((n % 6)) = 0 ]; then
      h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health)
      if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) $c mem_avail=${m}GiB 8031=$h" >> $W/watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) STOP $c: 8031 health failed" >> $W/watch.log; docker stop -t 20 $c >/dev/null 2>&1; touch $W/ABORT; return; fi
    fi
    n=$((n + 1)); sleep 10
  done
}

run_arm() {
  local a=$1 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st
  [ -f $W/ABORT ] && { echo "ABORT present, skipping $a"; return 1; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$a] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  python3 $W/arm_compose_1003.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)  MemAvailable ${m} GiB"
  echo "[$a] flags: $(sed -n '/command:/,$p' $W/compose.$a.yaml | grep -E "^      - " | sed "s/^      - //; s/'//g" | tail -n +2 | tr '\n' ' ' | cut -c1-700)"
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
  docker logs $n 2>&1 | grep -E "GPU KV cache size|Model loading took|Graph capturing finished|Capturing CUDA graphs \((FULL|PIECEWISE)\)[^|]*100%|fp8_e4m3|scale|Using .* data type to store kv|multimodal|vision|encoder cache|cudagraph_mode|CUDAGraphMode|sm70.*warn" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | sed -E 's/\r/\n/g' | grep -vE "^\s*$" | awk '!s[$0]++' | head -14 | cut -c1-210 | sed "s/^/[$a] /"
  for c in 1 2; do
    python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/${a}_c$c.json --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101 > $W/${a}_c$c.log 2>&1
    grep -E "pass [0-9]|ERROR|Traceback" $W/${a}_c$c.log | sed -E 's/ \| ttft.*prefill +([0-9.]+) tok\/s \| decode +([0-9.]+) tok\/s per stream, +([0-9.]+) agg.*kv peak ([0-9.]+%).*/ | prefill \1 dec \2 agg \3 kv \4/' | sed "s/^/[$a] /" | cut -c1-200
  done
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')  errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  docker rm -f $n >/dev/null 2>&1
  kill $wp 2>/dev/null
  return 0
}

for a in THEIRS T_FULL T_FP16 T_LM; do run_arm $a || break; done
echo "REPRO_DONE $(date +%T)"
