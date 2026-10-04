#!/bin/bash
# chain_top1.sh : after the startup-lever chain releases GPU 4-7, measure the LM_HEAD_TOP1 arm.
#
# The arm is arm_compose_1004top1.py / run_1004top1.sh: image 1004, cache forkwheel, --max-num-seqs 32,
# MoE cap 320 (that is exactly arm F1C32) plus the single env var VLLM_SM70_LM_HEAD_TOP1=1. It runs the
# C1/C8/C16/C24/C32 sweep, the 4-length long run, the 128K run, the needle/short answers, one sampled C8
# cell and the greedy-vs-sampled smoke check. Outputs are copied to top1_arm/ so later runs cannot hide them.
set -uo pipefail
cd /mnt/2t/build/cmp1003 || exit 1
echo "=== chain_top1 start $(date +%T)"
while pgrep -f 'chain_levers.sh' >/dev/null 2>&1; do sleep 30; done
echo "=== levers chain gone $(date +%T)"
sleep 15
bash run_1004top1.sh 2>&1 | tee run_1004top1.log
D=/mnt/2t/build/cmp1003/top1_arm
mkdir -p "$D"
for f in run_1004top1.log engine_PA.log compose.PA.yaml PA_answers.json PA_sweep_c1.json PA_sweep_c8.json \
         PA_sweep_c16.json PA_sweep_c24.json PA_sweep_c32.json PA_long.json PA_l128k.json PA_sampled_c8.json; do
  [ -f "$f" ] && cp -f "$f" "$D/"
done
ls -l "$D" | tail -n +2
echo "=== top1 facts"
grep -oE 'GPU KV cache size: [0-9,]+ tokens|Model loading took [0-9.]+ GiB and [0-9.]+ s|Graph capturing finished in [0-9]+ secs|LM_HEAD_TOP1=[01]' engine_PA.log | sort -u | head -6
grep -E 'Auto-setting VLLM_SM70_LM_HEAD_TOP1' engine_PA.log | head -2
echo "CHAIN_TOP1_DONE $(date +%T)"
