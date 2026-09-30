#!/bin/bash
# wheel_check.sh : run the fork's wheel-built image on GPU 4-7 with production's flags (no MTP) and compare it with
# the production image on 8031: version, greedy answers, and the 4-way cold-prompt benchmark of 2026-09-30
# (same script, seed and lengths as detail_fork_c4.json). The MTP + E4M3 instance on these GPUs is stopped first
# (container kept). Production 8031 is only read (greedy answer check).
set -u
W=/mnt/2t/build/fork-wheel-1001
B=/mnt/2t/build/pfx_ab
K=/mnt/2t/build/mtp_kv
D=/opt/shixiang-inference/docker-dflash2
M=Swift-1.5-Qwen3.8-Flash-Next
C=$D/compose.forkwheel-flashnext-tp4-gpu4567.yaml
N=shixiang-inference-forkwheel-flashnext-tp4
E=$D/compose.swift15-flashnext-mtp-e4m3-tp4-gpu4567.yaml
cd $W
python3 $W/mk_forkwheel_compose.py && docker compose -f $C config -q || { echo "CHECK_FAIL compose"; exit 1; }
echo "=== stop the MTP + E4M3 instance $(date +%T): running $(curl -s -m 5 localhost:8011/metrics | grep -E '^vllm:num_requests_running\{' | awk '{print $NF}')"
docker compose -f $E stop 2>&1 | tail -1
mkdir -p /opt/shixiang-inference/cache-forkwheel-flashnext-tp4 && chown 997:983 /opt/shixiang-inference/cache-forkwheel-flashnext-tp4
docker compose -f $C up -d 2>&1 | tail -1
T0=$(date +%s)
until curl -sf -m 5 localhost:8141/health -o /dev/null; do
  st=$(docker inspect -f '{{.State.Status}}' $N 2>/dev/null)
  [ "$st" != "running" ] && { echo "CHECK_FAIL container $st"; docker logs --tail 60 $N 2>&1 | cut -c1-300; docker logs $N > $W/engine_forkwheel.fail.log 2>&1; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 3600 ] && { echo "CHECK_FAIL health timeout"; docker logs --tail 40 $N 2>&1 | cut -c1-300; exit 1; }
  sleep 15
done
echo "healthy after $(( $(date +%s) - T0 ))s $(date +%T)"
echo "version: wheel image $(curl -s -m 5 localhost:8141/version)  production $(curl -s -m 5 localhost:8031/version)"
docker logs $N 2>&1 | grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Graph capturing finished|Prefix caching is|SX align multi-block|SM70 Qwen3.8" \
  | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //' | awk '!s[$0]++' | cut -c1-230 | head -20
echo "gpu mem after start (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
python3 $K/kvq_check.py 8141 $M $W/answers_wheel.json 2>&1 | tee $W/answers_wheel.log
python3 $K/kvq_check.py 8031 $M $W/answers_prod8031.json 2>&1 | tee $W/answers_prod8031.log
python3 - <<'EOF'
import json
a = json.load(open('/mnt/2t/build/fork-wheel-1001/answers_wheel.json'))
b = json.load(open('/mnt/2t/build/fork-wheel-1001/answers_prod8031.json'))
same = sum(x['answer'] == y['answer'] for k in ('needle', 'short') for x, y in zip(a[k], b[k]))
print('greedy answers identical to production: %d of %d' % (same, len(a['needle']) + len(a['short'])))
EOF
python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --seed 2026093020 --out $W/detail_wheel_c4.json 2>&1 | tee $W/detail_wheel_c4.log
echo "gpu mem after tests (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
echo "errors in engine log: $(docker logs $N 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
docker logs $N > $W/engine_forkwheel.log 2>&1
echo "CHECK_DONE $(date +%T)"
