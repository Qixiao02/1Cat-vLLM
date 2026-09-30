#!/bin/bash
# e4m3_chain.sh [overlay dir name] [calibration run name]
# Bring the fork with the 2026-09-30 memory work (image heavily-modified-v1-0930-r1) up on GPU 4-7 with native MTP
# (k=4) and the calibrated E4M3 QSA KV cache:
#   1. stop the official comparison lane (8011), start an eager FP16-KV MTP instance with the K/V observer on,
#      send the calibration traffic, stop that instance;
#   2. turn the observations into 24 target + 2 MTP scales and an overlay checkpoint directory next to the base
#      checkpoint (relative symlinks plus two small scale shards and a merged index; the base is not modified);
#   3. start the serving instance (127.0.0.1:8141 and 0.0.0.0:8011) on the overlay with --kv-cache-dtype fp8_e4m3,
#      check answers against production 8031 (FP16 KV), run the cold-prompt benchmark, and leave it running.
# If a step fails the test containers are removed and the official lane is started again.
# Production 8031 (GPU 0-3) is only read: health, and the three greedy prompts of the answer check.
set -u
W=/mnt/2t/build/mtp_kv
B=/mnt/2t/build/pfx_ab
D=/opt/shixiang-inference/docker-dflash2
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-0930-r1
BASE=Swift-1.5-Qwen3.8-Flash-Next-NVFP4
OVL=${1:-$BASE-e4m3kv}
RUN=${2:-run1}
M=Swift-1.5-Qwen3.8-Flash-Next
TAG=e4m3-u87s24
OFF=$D/compose.official-flashnext-tp4-gpu4567.yaml
CC=$W/compose.e4m3-calib.yaml
SC=$D/compose.swift15-flashnext-mtp-e4m3-tp4-gpu4567.yaml
CN=sx-e4m3-calib
SN=shixiang-inference-swift15-flashnext-mtp-e4m3-tp4
cd $W

START=${START:-1}   # START=3: the overlay checkpoint exists, only start and test the serving instance
if [ $START -le 1 ]; then
[ -e /mnt/2t/models/$OVL ] && { echo "CHAIN_FAIL overlay directory /mnt/2t/models/$OVL already exists"; exit 1; }
[ -e $W/calib/$RUN ] && { echo "CHAIN_FAIL calibration run $W/calib/$RUN already exists"; exit 1; }
[ -e $W/calib/pack-$RUN ] && { echo "CHAIN_FAIL pack $W/calib/pack-$RUN already exists"; exit 1; }
fi
python3 $W/e4m3_compose.py calib $RUN $OVL && python3 $W/e4m3_compose.py serve $RUN $OVL \
  && docker compose -f $CC config -q && docker compose -f $SC config -q || { echo "CHAIN_FAIL compose"; exit 1; }

fail() {
  echo "CHAIN_FAIL $* $(date +%T)"
  for c in $CN $SN; do
    docker inspect $c > /dev/null 2>&1 && docker logs $c > $W/engine_$c.fail.log 2>&1
  done
  docker compose -f $CC down 2>&1 | tail -1
  docker compose -f $SC down 2>&1 | tail -1
  docker compose -f $OFF start 2>&1 | tail -1
  echo "official lane started again on 8011 $(date +%T)"
  exit 1
}

wait_health() {  # wait_health <container> <timeout s>
  local t0=$(date +%s) st
  until curl -sf -m 5 localhost:8141/health -o /dev/null; do
    st=$(docker inspect -f '{{.State.Status}}' $1 2>/dev/null)
    [ "$st" != "running" ] && { docker logs --tail 40 $1 2>&1 | cut -c1-260; fail "$1 is $st"; }
    [ $(( $(date +%s) - t0 )) -gt $2 ] && { docker logs --tail 40 $1 2>&1 | cut -c1-260; fail "$1 health timeout"; }
    sleep 15
  done
  echo "$1 healthy after $(( $(date +%s) - t0 ))s $(date +%T)"
}

keylines() {
  docker logs $1 2>&1 | grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Graph capturing finished|kv_cache_dtype|E4M3|e4m3|k_scale|v_scale|Prefix caching is" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //' | awk '!s[$0]++' | cut -c1-230 | head -30
}

if [ $START -le 1 ]; then
echo "=== 1. calibration instance $(date +%T)"
echo "official lane before stop: running $(curl -s -m 5 localhost:8011/metrics | grep -E '^vllm:num_requests_running\{' | awk '{print $NF}') waiting $(curl -s -m 5 localhost:8011/metrics | grep -E '^vllm:num_requests_waiting\{' | awk '{print $NF}')"
docker compose -f $OFF stop 2>&1 | tail -1
mkdir -p $W/calib/$RUN /opt/shixiang-inference/cache-mtp-test-tp4 && chown 997:983 $W/calib $W/calib/$RUN
docker compose -f $CC up -d 2>&1 | tail -1
wait_health $CN 2400
keylines $CN
echo "mixed-v1" > $W/calib/$RUN/COLLECTING
python3 $W/calib_traffic.py 8141 $M $W/calib/traffic_$RUN.json 2>&1 | tee $W/calib/traffic_$RUN.log
[ ${PIPESTATUS[0]} -eq 0 ] || fail "calibration traffic"
mv $W/calib/$RUN/COLLECTING $W/calib/$RUN/COLLECTED
echo "observer files: $(ls $W/calib/$RUN/*.jsonl 2>/dev/null | wc -l), records $(cat $W/calib/$RUN/*.jsonl 2>/dev/null | wc -l), $(du -sh $W/calib/$RUN | cut -f1)"
echo "errors in calibration engine log: $(docker logs $CN 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
docker logs $CN > $W/engine_e4m3-calib.log 2>&1
docker compose -f $CC down 2>&1 | tail -1
[ "$(ls $W/calib/$RUN/*.jsonl 2>/dev/null | wc -l)" -ge 4 ] || fail "no observer records"

echo "=== 2. scales and overlay checkpoint $(date +%T)"
docker run --rm --runtime=runc --user 0:0 --entrypoint bash -v /mnt/2t/models:/models -v $W:/work $IMG -c "
set -e
T=/work/patch664/tools/qwen4_exp; C=/work/calib
python \$T/qsa_kv_calibration.py summarize --input-dir \$C/$RUN --output \$C/report_full_$RUN.json --expected-layers 13
python \$T/sx_scale_pack.py target-report --report \$C/report_full_$RUN.json --output \$C/report_target_$RUN.json --mtp-layer-id 48
python \$T/qsa_kv_calibration.py overlay --base-checkpoint /models/$BASE --report \$C/report_target_$RUN.json --output-dir \$C/pack-$RUN --expected-layers 12
python \$T/sx_scale_pack.py manifest --pack-dir \$C/pack-$RUN --base-checkpoint /models/$BASE --report \$C/report_target_$RUN.json --artifact-id swift15-flashnext-e4m3-20260930-$RUN
python \$T/materialize_qsa_scale_overlay.py --base-checkpoint /models/$BASE --output-dir /models/$OVL --pack-dir \$C/pack-$RUN --mtp-report \$C/report_full_$RUN.json --mtp-layer-id 48
python - <<'PY'
import json
r = json.load(open('/work/calib/report_full_$RUN.json'))
print('layer  K max_abs   K scale    V max_abs   V scale')
t = r['tensors']
for lid in sorted({int(n.split('.')[2]) for n in t}):
    k, v = t['model.layers.%d.self_attn.k_scale' % lid], t['model.layers.%d.self_attn.v_scale' % lid]
    print('%5d  %9.3f  %9.6f  %9.3f  %9.6f' % (lid, k['max_abs'], k['scale'], v['max_abs'], v['scale']))
PY
" 2>&1 | cut -c1-240 || true
[ -r /mnt/2t/models/$OVL/model.safetensors.index.json ] || fail "overlay checkpoint was not written"
echo "overlay: $(ls /mnt/2t/models/$OVL | wc -l) entries, real files: $(find /mnt/2t/models/$OVL -maxdepth 1 -type f -printf '%f ' ), scale tensors in index: $(grep -o '_scale"' /mnt/2t/models/$OVL/model.safetensors.index.json | wc -l)"

# The scale shard is copied with the 0600 mode safetensors gave it; the engine runs as uid 997.
find /mnt/2t/models/$OVL -maxdepth 1 -type f -exec chmod 644 {} +
else
  [ -r /mnt/2t/models/$OVL/model-kvscales.safetensors ] || { echo "CHAIN_FAIL no overlay checkpoint /mnt/2t/models/$OVL"; exit 1; }
  docker compose -f $OFF stop 2>&1 | tail -1
fi

echo "=== 3. serving instance, E4M3 KV + MTP $(date +%T)"
docker compose -f $SC up -d 2>&1 | tail -1
wait_health $SN 2400
keylines $SN
echo "gpu mem after start (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
python3 $W/kvq_check.py 8141 $M $W/kvq_$TAG.json 2>&1 | tee $W/kvq_$TAG.log
python3 $W/kvq_check.py 8031 $M $W/kvq_prod8031.json 2>&1 | tee $W/kvq_prod8031.log
run() {  # run <name> <pfx_bench args...>
  local name=$1; shift
  python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --seed 2026093021 --out $W/${name}_$TAG.json "$@" 2>&1 \
    | sed "s/^/[$name] /" | tee $W/${name}_$TAG.log
}
run c1s   --conc 1 --lengths 8000 --passes 1
run c1g   --conc 1 --lengths 8000 --passes 1 --temperature 0
run c4s2k --conc 4 --lengths 2000 --passes 2 --gen 300
run c4g2k --conc 4 --lengths 2000 --passes 2 --gen 300 --temperature 0
run c4    --conc 4 --lengths 8000,16000,32000 --passes 1
python3 $W/mtp_summary.py $W $TAG
echo "gpu mem after tests (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
echo "errors in engine log: $(docker logs $SN 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
docker logs $SN > $W/engine_$TAG.log 2>&1
echo "CHAIN_DONE serving instance left running on 8011 $(date +%T)"
