#!/bin/bash
# Can the weight-loading phase be shortened with flags only? (no wheel rebuild)
# Rounds interleave control / multithread / prefetch so host drift is visible.
W=/mnt/2t/build/cmp1003
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main
CACHE=/opt/shixiang-inference/cache-forkwheel-flashnext-tp4
OUT=$W/startup_load
mkdir -p "$OUT"
cd "$W" || exit 1
rm -f PA_*.json

run_one() {
  local TAG=$1 MODE=$2
  echo
  echo "=== load round $TAG mode=$MODE $(date +%T)  MemAvailable $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) GiB"
  echo "--- resident share of the checkpoint before the round"
  if command -v vmtouch >/dev/null 2>&1; then
    vmtouch -v /models 2>/dev/null | tail -2
  else
    awk '/^Cached:/{printf "pagecache %.1f GiB\n", $2/1048576}' /proc/meminfo
  fi
  TAG=$TAG IMG=$IMG CACHE=$CACHE EXTRA='{}' CONCS="1" LONG=0 MODE=$MODE \
    python3 make_load_arm.py | tail -2 || { echo "GEN_FAIL $TAG"; return 1; }
  python3 "arm_compose_$TAG.py" PA > /dev/null || { echo "ARM_FAIL $TAG"; return 1; }
  docker compose -f "$W/compose.PA.yaml" config -q || { echo "BAD_COMPOSE $TAG"; return 1; }
  echo "--- loader flags in compose: $(grep -c -E 'model-loader-extra-config|safetensors-load-strategy' "$W/compose.PA.yaml")"
  # 15 s health polling is too coarse; tighten to 5 s for a 3x finer total.
  sed -i 's/sleep 15$/sleep 5/' "run_$TAG.sh"
  echo "--- health poll interval: $(grep -c 'sleep 5$' "run_$TAG.sh") x 5s"
  cp -f "$W/compose.PA.yaml" "$OUT/compose.$TAG.yaml"
  bash "run_$TAG.sh" 2>&1 | tee "run_$TAG.log"
  mkdir -p "$OUT/$TAG"
  cp -f "run_$TAG.log" "$OUT/$TAG/" 2>/dev/null
  [ -f engine_PA.log ] && cp -f engine_PA.log "$OUT/$TAG/"
  for f in PA_*.json; do [ -f "$f" ] && mv -f "$f" "$OUT/$TAG/"; done
  echo "--- round $TAG metrics"
  grep -hoE 'Loading weights took [0-9.]+ s' "$OUT/$TAG/engine_PA.log" 2>/dev/null | tr '\n' ' '; echo
  grep -hoE 'Model loading took [0-9.]+ GiB and [0-9.]+ s' "$OUT/$TAG/engine_PA.log" 2>/dev/null | tr '\n' ' '; echo
  grep -hoE 'Graph capturing finished in [0-9]+ secs' "$OUT/$TAG/engine_PA.log" 2>/dev/null | tail -1
  grep -hoE '\[PA\] healthy after [0-9]+s' "run_$TAG.log" 2>/dev/null | tail -1
  grep -hoE 'GPU KV cache size: [0-9,]+ tokens' "$OUT/$TAG/engine_PA.log" 2>/dev/null | tail -1
  return 0
}

run_one 1004ldctl  ctl
run_one 1004ldmt   mt
run_one 1004ldpf   pf
run_one 1004ldctl2 ctl
run_one 1004ldmt2  mt

echo
echo "########## SUMMARY ##########"
for d in "$OUT"/*/; do
  t=$(basename "$d")
  lw=$(grep -hoE 'Loading weights took [0-9.]+ s' "$d"engine_PA.log 2>/dev/null | tr '\n' '/')
  ml=$(grep -hoE 'Model loading took [0-9.]+ GiB and [0-9.]+ s' "$d"engine_PA.log 2>/dev/null | tail -1)
  h=$(grep -hoE '\[PA\] healthy after [0-9]+s' "$d"run_*.log 2>/dev/null | tail -1)
  echo "$t | loading=$lw | $ml | $h"
done
echo "LOAD_AB_DONE $(date +%T)"
