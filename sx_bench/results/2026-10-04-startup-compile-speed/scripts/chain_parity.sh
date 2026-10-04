#!/bin/bash
# chain_parity.sh : after the load-flag chain is done, run the fork's own
# sx_tests/compile-cache parity harness on GPU 4-7 against the 1004 image.
# This is the gate our README requires before SX_OPT_COMPILE_CACHE may be
# enabled in production.  Nothing on GPU 0-3 / port 8031 is touched.
set -u
W=/mnt/2t/build/cmp1003
D=$W/sx_tests/compile-cache
LOG=$W/chain_parity.log
say() { echo "$*" >> "$LOG"; }

say "=== chain_parity start $(date +%T)"
# 1) wait for the load chain to disappear
for i in $(seq 1 240); do
  pgrep -f 'bash chain_load.sh' > /dev/null 2>&1 || break
  sleep 30
done
say "--- chain_load gone $(date +%T) (waited ${i}x30s)"
# 2) wait for the round scripts and GPUs 4-7
for i in $(seq 1 120); do
  if pgrep -f 'bash run_1004[a-z0-9]*\.sh' > /dev/null 2>&1; then sleep 20; continue; fi
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=4 && $2+0>1500' | wc -l)
  [ "$busy" = "0" ] && break
  sleep 15
done
say "--- gpu4-7 free at $(date +%T) (busy=$busy)"

# 3) run the parity harness: nomtp lane (production shape) + mtp lane, 4 arms each
export COMPOSE=/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml.bak-1003-pre-upgrade-20261004022023
export IMAGE=shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main
export GPUS=4,5,6,7 PORT=8141
export WORK=/mnt/2t/build/cache_parity
export LANES="nomtp"
export MODES="subgraph"
export ARMS="off,off2,cold,warm"
export STOP_FIRST= RESTART_AFTER=
export EXTRA="--env VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS=320"
mkdir -p "$WORK"
say "=== harness start $(date +%T) IMAGE=$IMAGE ARMS=$ARMS"
bash "$D/cache_parity.sh" >> "$LOG" 2>&1
rc=$?
say "=== CACHE_PARITY_RC=$rc $(date +%T)"
if [ $rc -ne 0 ]; then
  say "--- failure details"
  grep -nE 'DIFFER|FAIL|MISMATCH|parity|do not enable' "$WORK"/*.log 2>/dev/null | tail -30 >> "$LOG"
fi
say "CHAIN_PARITY_DONE $(date +%T)"
