#!/usr/bin/env bash
# startup_ab_rev.sh : order-effect control for the compile-cache A/B. The first two pairs ran control-then-switch
# (C,S,C,S) and both showed exactly -30 s. If that were host warming rather than the lever, a switch-then-control
# pair (S,C) would show the switch SLOWER by about the same margin. Tags 1004su3 / 1004ctl3 keep the files apart.
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
    python3 make_port_arm.py | tail -2 || return 1
  python3 "arm_compose_$TAG.py" PA >/dev/null || return 1
  docker compose -f compose.PA.yaml config -q || { echo "COMPOSE_FAIL $TAG"; return 1; }
  cp -f compose.PA.yaml "$OUT/compose.$TAG.yaml"
  bash "run_$TAG.sh" 2>&1 | tee "run_$TAG.log"
  mkdir -p "$OUT/$TAG"
  cp -f "run_$TAG.log" engine_PA.log "$OUT/$TAG/" 2>/dev/null
  cp -f PA_answers.json PA_sweep_c8.json "$OUT/$TAG/" 2>/dev/null
  echo "--- $TAG timings"
  grep -oE '\[PA\] healthy after [0-9]+s' "run_$TAG.log" | tail -1
  grep -E 'torch.compile took|Graph capturing finished|Model loading took|SX_OPT_COMPILE_CACHE=' engine_PA.log 2>/dev/null | head -6
  echo "=== done $TAG $(date +%T)"
}

run_one 1004su3  "$SU_EXTRA"  "8" 0
run_one 1004ctl3 "$CTL_EXTRA" "8" 0
echo "STARTUP_AB_REV_DONE $(date +%T)"
