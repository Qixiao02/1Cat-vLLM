#!/bin/bash
# bench_prod8031_seq32.sh : same instrument as the comparison arms, run against production 8031 (GPU 0-3)
# to confirm the 24 -> 32 switch on the live compose. C8 for the mid point, C32 for the new ceiling.
set -u
W=/mnt/2t/build/cmp1003
M=Swift-1.5-Qwen3.8-Flash-Next
B=/mnt/2t/build/pfx_ab
cd $W
for c in 8 32; do
  echo "=== prod8031 C$c $(date +%T)"
  python3 $B/pfx_bench.py --port 8031 --model $M --gpus 0,1,2,3 \
      --out $W/prod8031_seq32_sweep_c$c.json \
      --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101 \
      > $W/prod8031_seq32_sweep_c$c.log 2>&1
  echo "exit=$?"
  grep -E -e 'pass [0-9]' -e ERROR -e Traceback -e WAVE $W/prod8031_seq32_sweep_c$c.log | tail -6
done
echo "PROD_BENCH_DONE $(date +%T)"
