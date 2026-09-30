#!/bin/bash
# ab_detail.sh : the afternoon benchmark (concurrency 4, cold prompts, 8K/16K/32K/64K, 2 passes) with the detailed
# record (pfx_bench.py schema 2), first on the fork (8031, GPU 0-3), then on the official image (8011, GPU 4-7).
# Same model and the same prompts (same seed) on both; one lane at a time so they do not compete for the host CPU.
set -u
W=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
cd $W
run() {  # run <lane> <port> <gpus> <container>
  echo "=== $1 (port $2, GPU $3) $(date +%T) version $(curl -s -m 5 localhost:$2/version)"
  docker inspect $4 > $W/launch_$1.json
  docker logs $4 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Prefix caching is|SX align multi-block|Initializing a V1 LLM engine" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/with config:.*//' | sort -u | tail -6 | cut -c1-230
  python3 $W/pfx_bench.py --port $2 --model $M --gpus $3 --seed 2026093020 --out $W/detail_$1_c4.json 2>&1 | tee $W/detail_$1_c4.log
}
run fork 8031 0,1,2,3 shixiang-inference-swift15-flashnext-tp4
run official 8011 4,5,6,7 shixiang-inference-official-flashnext-tp4
echo "DETAIL_DONE $(date +%T)"
