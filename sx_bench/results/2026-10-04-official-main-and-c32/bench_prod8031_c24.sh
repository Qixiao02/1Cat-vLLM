#!/bin/bash
# bench_prod8031_c24.sh : the one cell that gets slower after the 24 -> 32 switch (24-lane waves fall out of the
# captured graph set), measured on production 8031 so the trade-off is quantified on the live compose.
set -u
W=/mnt/2t/build/cmp1003
M=Swift-1.5-Qwen3.8-Flash-Next
B=/mnt/2t/build/pfx_ab
c=24
cd $W
echo "=== prod8031 C$c $(date +%T)"
python3 $B/pfx_bench.py --port 8031 --model $M --gpus 0,1,2,3 \
    --out $W/prod8031_seq32_sweep_c$c.json \
    --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101 \
    > $W/prod8031_seq32_sweep_c$c.log 2>&1
echo "exit=$?"
grep -E -e 'pass [0-9]' -e ERROR -e Traceback -e WAVE $W/prod8031_seq32_sweep_c$c.log | tail -6
echo "PROD_BENCH_C24_DONE $(date +%T)"
