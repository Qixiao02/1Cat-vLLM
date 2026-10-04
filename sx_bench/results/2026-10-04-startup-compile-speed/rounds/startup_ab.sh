#!/usr/bin/env bash
# startup_ab.sh : interleaved A/B of the torch.compile cache lever on the bench host.
#
# Control arm  = exactly the F1C32 configuration (image 1004, seqs 32, MoE cap 320, cache disabled).
# Switch arm   = control + SX_OPT_COMPILE_CACHE=1 and VLLM_DISABLE_COMPILE_CACHE=0 (probe-verified:
#                plan_policy() -> mode=subgraph, cache_disabled=False, AOT forced to 0).
#
# Interleaved C, S, C, S so host drift (page cache, neighbours) cancels instead of biasing one side.
# Fast acceptance (C8 + needle answers) on the A/B runs; run_startup_full.sh does the full sweep.
set -uo pipefail
W=/mnt/2t/build/cmp1003
cd "$W" || exit 1
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main
CACHE=/opt/shixiang-inference/cache-forkwheel-flashnext-tp4
OUT=$W/startup_ab
mkdir -p "$OUT"
CTL_EXTRA='{}'
SU_EXTRA='{"SX_OPT_COMPILE_CACHE": "1", "VLLM_DISABLE_COMPILE_CACHE": "0"}'

run_one () {
  local TAG=$1 EXTRA=$2 CONCS=$3 LONG=$4
  echo "=== run_one $TAG concs='$CONCS' long=$LONG $(date +%T) avail=$(free -g | awk '/Mem:/{print $7}')GiB"
  TAG=$TAG IMG=$IMG CACHE=$CACHE EXTRA="$EXTRA" ARMS=std CONCS="$CONCS" LONG="$LONG" \
    python3 make_port_arm.py | tail -3 || return 1
  python3 "arm_compose_$TAG.py" PA >/dev/null || return 1
  docker compose -f compose.PA.yaml config -q || { echo "COMPOSE_FAIL $TAG"; return 1; }
  cp -f compose.PA.yaml "$OUT/compose.$TAG.yaml"
  bash "run_$TAG.sh" 2>&1 | tee "run_$TAG.log"
  mkdir -p "$OUT/$TAG"
  cp -f "run_$TAG.log" engine_PA.log "$OUT/$TAG/" 2>/dev/null
  cp -f PA_*.json "$OUT/$TAG/" 2>/dev/null
  echo "--- $TAG timings"
  grep -oE '\[PA\] healthy after [0-9]+s' "run_$TAG.log" | tail -2
  grep -E 'Dynamo compiled model in|Inductor compiled model in|torch.compile took|Graph capturing finished|Model loading took|compile_cache|SX_OPT_COMPILE_CACHE' \
    engine_PA.log 2>/dev/null | head -20
  echo "=== done $TAG $(date +%T)"
}

run_one 1004ctl "$CTL_EXTRA" "8" 0
run_one 1004su  "$SU_EXTRA"  "8" 0
run_one 1004ctl2 "$CTL_EXTRA" "8" 0
run_one 1004su2 "$SU_EXTRA"  "8" 0
echo "STARTUP_AB_DONE $(date +%T)"
