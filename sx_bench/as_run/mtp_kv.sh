#!/bin/bash
# mtp_kv.sh <tag> <gpu_mem_util> <max_num_seqs> <max_model_len> <max_batched_tokens> [K=V,K=V extra env] [patch dir]
# One MTP (k=4) trial of the fork image on GPU 4-7 (test port 127.0.0.1:8141): start, report the KV budget and the
# memory lines, run the cold-prompt benchmark in several shapes, report the peak GPU memory, stop and remove the
# test container. Production 8031 (GPU 0-3) is not touched. GPU 4-7 must be free before calling this.
set -u
TAG=${1:?tag}; UTIL=${2:?util}; SEQS=${3:?max_num_seqs}; MAXLEN=${4:?max_model_len}; BATCHED=${5:?max_batched_tokens}
EXTRA=${6:-}
PATCH=${7:-}
P=/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml
W=/mnt/2t/build/mtp_kv
B=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
TC=sx-mtp-$TAG
mkdir -p $W /opt/shixiang-inference/cache-mtp-test-tp4 && chown 997:983 /opt/shixiang-inference/cache-mtp-test-tp4
cd $W

python3 $W/mtp_compose.py "$P" "$TAG" "$UTIL" "$SEQS" "$MAXLEN" "$BATCHED" "$EXTRA" "$PATCH" || { echo "BAD_COMPOSE"; exit 1; }
docker compose -f compose.$TAG.yaml config -q || { echo "BAD_COMPOSE"; exit 1; }

cleanup() {
  docker logs $TC > $W/engine_$TAG.log 2>&1 || true
  docker compose -f $W/compose.$TAG.yaml down 2>&1 | tail -1
  echo "MTP_TRIAL_DONE $TAG $(date +%T)"
}
trap cleanup EXIT

echo "=== $TAG: util $UTIL, max_num_seqs $SEQS, max_model_len $MAXLEN, batched $BATCHED, extra [$EXTRA], patch [$PATCH] $(date +%T)"
free -g | sed -n 2p
docker compose -f compose.$TAG.yaml up -d 2>&1 | tail -1
T0=$(date +%s)
until curl -sf -m 5 localhost:8141/health -o /dev/null; do
  st=$(docker inspect -f '{{.State.Status}}' $TC 2>/dev/null)
  [ "$st" != "running" ] && { echo "TEST_CONTAINER_$st"; docker logs --tail 30 $TC 2>&1 | cut -c1-240; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && { echo "TEST_HEALTH_TIMEOUT"; docker logs --tail 30 $TC 2>&1 | cut -c1-240; exit 1; }
  sleep 15
done
echo "healthy after $(( $(date +%s) - T0 ))s"
docker logs $TC 2>&1 | grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Graph capturing finished|verify cudagraph token shapes|Prefix caching is|SX align multi-block" \
  | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //' | awk '!s[$0]++' | cut -c1-250
echo "gpu mem after start (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"

run() {  # run <name> <pfx_bench args...>
  local name=$1; shift
  python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --seed 2026093021 --out $W/${name}_$TAG.json "$@" 2>&1 \
    | sed "s/^/[$name] /" | tee $W/${name}_$TAG.log
}
run c1s   --conc 1 --lengths 8000 --passes 1                      # single stream, server-default sampling
run c1g   --conc 1 --lengths 8000 --passes 1 --temperature 0      # single stream, greedy
run c4s2k --conc 4 --lengths 2000 --passes 2 --gen 300            # the 2026-09-27 shape: short prompt, 300 tokens
run c4g2k --conc 4 --lengths 2000 --passes 2 --gen 300 --temperature 0
run c4    --conc 4 --lengths 8000,16000,32000 --passes 1          # real context lengths
python3 $W/mtp_summary.py $W $TAG
echo "gpu mem after tests (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
echo "errors in engine log: $(docker logs $TC 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
