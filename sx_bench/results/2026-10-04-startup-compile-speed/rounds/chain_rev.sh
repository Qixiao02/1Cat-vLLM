#!/bin/bash
# chain_rev.sh : after the TOP1 arm, run the reverse-order compile-cache pair (switch then control).
set -uo pipefail
cd /mnt/2t/build/cmp1003 || exit 1
echo "=== chain_rev start $(date +%T)"
while pgrep -f 'chain_top1.sh' >/dev/null 2>&1; do sleep 30; done
echo "=== top1 chain gone $(date +%T)"
sleep 15
bash startup_ab_rev.sh 2>&1 | tee startup_ab_rev.out
echo "CHAIN_REV_DONE $(date +%T)"
