#!/bin/bash
# validate_wheel_1003.sh : after the 1003 wheel build (build_1003.sh) has finished:
#   1. take the wheel out of the build image, run the SM70 release artifact check, checksum
#   2. a validation image = the 1002 image (same dependencies, CUDA toolkit, Ubuntu 22.04) with 1cat-vllm and flash_attn_v100
#      replaced by the new wheel (--no-deps)
#   3. run it on GPU 4-7 with the production flags (arm F1W = arm F1 of arm_compose_1003.py, fresh cache directory so the first start
#      compiles everything as it would for someone who installs the wheel): answers, 8K greedy at C1 and C4, 4 cold prompts of
#      8K/16K/32K/64K; compare with arm F1 (the same code as a Python overlay on the 1002 build)
# One heavy thing at a time; watchdog from arm_functions.sh (MemAvailable < 25 GiB or production 8031 down stops the arm).
set -u
S=/mnt/2t/build/fork-wheel-1001
W=/mnt/2t/build/cmp1003
M=Swift-1.5-Qwen3.8-Flash-Next
until grep -q "== BUILD_EXIT" $S/build_1003.log; do sleep 20; done
grep -q "== BUILD_EXIT 0" $S/build_1003.log || { echo "WHEEL_BUILD_FAILED"; echo "VALIDATE_DONE"; exit 1; }
echo "== wheel build finished $(date +%T)"
docker rm -f sx-build-egress-proxy >/dev/null 2>&1 && echo "egress proxy container removed"
mkdir -p $S/artifacts-1003
docker run --rm -v $S/artifacts-1003:/out -v /mnt/2t/build/official-d304698/src/tools/check_sm70_release_artifact.py:/chk.py:ro vllm-ci:build-image-sm70-fork-1003 \
  bash -c 'rm -rf /out/dist; cp -r dist /out && chmod -R a+rw /out; ls -la dist; python3 /chk.py dist/*.whl; echo "upstream artifact check exit $?"' 2>&1 | tail -8 | cut -c1-200
WHL=$(ls $S/artifacts-1003/dist/*.whl | head -1)
echo "wheel: $WHL"
echo "sha256: $(sha256sum $WHL | cut -d' ' -f1)  size: $(stat -c %s $WHL) bytes"
python3 - "$WHL" <<'EOF'
import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1])
meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
t = z.read(meta).decode("utf-8", "replace")
print("wheel METADATA: Version", re.search(r"^Version: (.*)$", t, re.M).group(1), "| files", len(z.namelist()), "| .so", sum(n.endswith(".so") for n in z.namelist()))
EOF

echo "== validation image"
V=$S/val1003; rm -rf $V; mkdir -p $V; cp $WHL $V/
cat > $V/Dockerfile <<EOF
FROM shixiang/1cat-vllm-v100:heavily-modified-v1-1002-sm70main
COPY $(basename $WHL) /tmp/
RUN uv pip uninstall --system 1cat-vllm flash_attn_v100 2>&1 | tail -3 \
 && uv pip install --system --no-deps /tmp/$(basename $WHL) 2>&1 | tail -3 \
 && rm -f /tmp/*.whl && python3 -c "import vllm; print('vllm', vllm.__version__)"
EOF
docker build -q -t shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main $V > /dev/null || { echo "VALIDATION_IMAGE_FAILED"; echo "VALIDATE_DONE"; exit 1; }
docker run --rm --entrypoint python3 shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main -c "import vllm; print('validation image imports vllm', vllm.__version__)" 2>&1 | tail -1

. $W/arm_functions.sh
cd $W
rm -f ABORT
mkdir -p /opt/shixiang-inference/cache-forkwheel-1003-flashnext-tp4 && chown 997:983 /opt/shixiang-inference/cache-forkwheel-1003-flashnext-tp4
export FORK_IMAGE=shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
MA=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); [ "$MA" -lt 100 ] && { echo "MemAvailable ${MA} GiB < 100, not starting"; echo "VALIDATE_DONE"; exit 1; }
python3 $W/arm_compose_1003.py F1W > /dev/null && docker compose -f $W/compose.F1W.yaml config -q || { echo "BAD_COMPOSE F1W"; echo "VALIDATE_DONE"; exit 1; }
n=sx-cmp-f1w
echo "=== start F1W $(date +%T)  MemAvailable ${MA} GiB"
docker compose -f $W/compose.F1W.yaml up -d 2>&1 | tail -1
watch_arm $n & wp=$!
t0=$(date +%s)
until curl -sf -m 5 localhost:8141/health -o /dev/null; do
  st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null)
  if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 3000 ]; then echo "ARM_FAIL F1W ($st)"; docker logs --tail 25 $n 2>&1 | cut -c1-220; docker logs $n > $W/engine_F1W.log 2>&1; docker compose -f $W/compose.F1W.yaml down 2>&1 | tail -1; docker rm -f $n >/dev/null 2>&1; kill $wp 2>/dev/null; echo "VALIDATE_DONE"; exit 1; fi
  sleep 10
done
echo "[F1W] healthy after $(( $(date +%s) - t0 ))s; version $(curl -s -m 5 localhost:8141/version)"
docker logs $n 2>&1 | grep -E "GPU KV cache size|Model loading took|Graph capturing finished|Maximum concurrency" | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | head -5 | cut -c1-160 | sed 's/^/[F1W] /'
python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/F1W_answers.json 2>&1 | tail -1 | sed 's/^/[F1W] /'
for c in 1 4; do
  python3 /mnt/2t/build/pfx_ab/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/F1W_sweep_c$c.json --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101 > $W/F1W_sweep_c$c.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback" $W/F1W_sweep_c$c.log | cut -c1-240 | sed 's/^/[F1W sweep] /'
done
python3 /mnt/2t/build/pfx_ab/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/F1W_long.json --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 2 --seed 2026100103 > $W/F1W_long.log 2>&1
grep -E "pass [0-9]|ERROR|Traceback" $W/F1W_long.log | cut -c1-240 | sed 's/^/[F1W long] /'
echo "[F1W] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')  errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
docker logs $n > $W/engine_F1W.log 2>&1
docker compose -f $W/compose.F1W.yaml down 2>&1 | tail -1
docker rm -f $n >/dev/null 2>&1
kill $wp 2>/dev/null
echo "VALIDATE_DONE $(date +%T)"
