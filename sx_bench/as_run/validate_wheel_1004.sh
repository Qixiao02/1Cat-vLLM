#!/bin/bash
# validate_wheel_1004.sh : after the 1004 wheel build (build_1004.sh) has finished:
#   1. take the wheel out of the build image, run the SM70 release artifact check, checksum
#   2. a validation image = the 1003 validation image (same dependencies and CUDA toolkit) with 1cat-vllm and flash_attn_v100
#      replaced by the new wheel (--no-deps)
#   3. run it on GPU 4-7 with the production flags, SX_OPT_KV_STEADY_BUDGET unset (the new default) and, as same-session controls,
#      set to 0 (arms of add_arms_1004.py, fresh cache directory so the first start compiles everything as it would for someone
#      who installs the wheel):
#        F1A  no MTP, util 0.90, switch unset          F1Z  the same with SX_OPT_KV_STEADY_BUDGET=0
#        FMA  MTP k=4, util 0.87, switch unset         FMZ  the same with SX_OPT_KV_STEADY_BUDGET=0
#        FMB  MTP k=4, util 0.93, switch unset (no SX_OPT_KV_STEADY_RESERVE_MIB)
#        F1B  no MTP, util 0.94, switch unset (above the util where the physical bound limits the KV cache)
#      per arm: answers (needle + 4 short questions), KV capacity and the steady-budget lines of the engine log; no MTP: 8K greedy at
#      C1/C4/C24 (2 passes) and 4 cold prompts of 8K/16K/32K/64K; MTP: mtp_bench plus 8K greedy at C1/C4 (2 passes)
# One heavy thing at a time; watchdog from arm_functions.sh (MemAvailable < 25 GiB or production 8031 down stops the arm).
set -u
S=/mnt/2t/build/fork-wheel-1001
W=/mnt/2t/build/cmp1003
B=/mnt/2t/build/pfx_ab
H=/mnt/2t/build/cmp1001
M=Swift-1.5-Qwen3.8-Flash-Next
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main
CACHE=/opt/shixiang-inference/cache-forkwheel-1004-flashnext-tp4
until grep -q "== BUILD_EXIT" $S/build_1004.log; do sleep 20; done
grep -q "== BUILD_EXIT 0" $S/build_1004.log || { echo "WHEEL_BUILD_FAILED"; echo "VALIDATE_DONE"; exit 1; }
echo "== wheel build finished $(date +%T)"
docker rm -f sx-build-egress-proxy >/dev/null 2>&1 && echo "egress proxy container removed"
mkdir -p $S/artifacts-1004
docker run --rm -v $S/artifacts-1004:/out -v /mnt/2t/build/official-d304698/src/tools/check_sm70_release_artifact.py:/chk.py:ro vllm-ci:build-image-sm70-fork-1004 \
  bash -c 'rm -rf /out/dist; cp -r dist /out && chmod -R a+rw /out; ls -la dist; python3 /chk.py dist/*.whl; echo "upstream artifact check exit $?"' 2>&1 | tail -8 | cut -c1-200
WHL=$(ls $S/artifacts-1004/dist/*.whl | head -1)
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
V=$S/val1004; rm -rf $V; mkdir -p $V; cp $WHL $V/
cat > $V/Dockerfile <<EOF
FROM shixiang/1cat-vllm-v100:heavily-modified-v1-1003-sm70main
COPY $(basename $WHL) /tmp/
RUN uv pip uninstall --system 1cat-vllm flash_attn_v100 2>&1 | tail -3 \
 && uv pip install --system --no-deps /tmp/$(basename $WHL) 2>&1 | tail -3 \
 && rm -f /tmp/*.whl && python3 -c "import vllm; print('vllm', vllm.__version__)"
EOF
docker build -q -t $IMG $V > /dev/null || { echo "VALIDATION_IMAGE_FAILED"; echo "VALIDATE_DONE"; exit 1; }
docker run --rm --entrypoint python3 $IMG -c "import vllm; print('validation image imports vllm', vllm.__version__)" 2>&1 | tail -1

. $W/arm_functions.sh
cd $W
rm -f ABORT
python3 $W/add_arms_1004.py
mkdir -p $CACHE && chown 997:983 $CACHE
export FORK_IMAGE=$IMG

bench() {  # bench <arm> <name> <pfx_bench args...>
  local a=$1 name=$2; shift 2
  python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/${a}_${name}.json "$@" > $W/${a}_${name}.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback|WAVE" $W/${a}_${name}.log | sed "s/^/[$a $name] /" | cut -c1-250
}

tests() {  # tests <arm> <std|mtp>
  local a=$1
  echo "=== tests $a $(date +%T)"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  if [ "$2" = std ]; then
    for c in 1 4 24; do
      [ -f $W/ABORT ] && return 1
      bench $a sweep_c$c --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
    done
    bench $a long --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 1 --seed 2026100103
  else
    python3 $H/mtp_bench.py 8141 $M $W/${a}_mtp.json 2>&1 | grep mtp_bench | sed "s/^/[$a] /"
    bench $a mtp_c1_8k --conc 1 --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
    bench $a mtp_c4_8k --conc 4 --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
  fi
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
}

keylines() {
  docker logs $1 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished|speculative_config=|Auto-enabling|KV steady budget:|KV steady audit: (OK|SHORT|measured)" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | cut -c1-260 | head -20
}

arm() {  # arm <arm> <std|mtp>
  local a=$1 n=sx-cmp-$(echo $1 | tr A-Z a-z) m t0 st
  [ -f $W/ABORT ] && { echo "ABORT file present, skipping $a"; return 1; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
  m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  [ "$m" -lt 100 ] && { echo "[$a] MemAvailable ${m} GiB < 100, not starting"; return 1; }
  python3 $W/arm_compose_1003.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)  MemAvailable ${m} GiB  SX_OPT_KV_STEADY_BUDGET in compose: $(grep -c 'SX_OPT_KV_STEADY_BUDGET' $W/compose.$a.yaml) $(grep -o 'SX_OPT_KV_STEADY_BUDGET: "[01]"' $W/compose.$a.yaml)"
  docker compose -f $W/compose.$a.yaml up -d 2>&1 | tail -1
  watch_arm $n & local wp=$!
  t0=$(date +%s)
  until curl -sf -m 5 localhost:8141/health -o /dev/null; do
    st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null)
    if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 3600 ]; then
      echo "ARM_FAIL $a ($st)"; docker logs --tail 30 $n 2>&1 | cut -c1-250
      docker logs $n > $W/engine_$a.log 2>&1; docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1; docker rm -f $n >/dev/null 2>&1; kill $wp 2>/dev/null; return 1
    fi
    sleep 15
  done
  echo "[$a] healthy after $(( $(date +%s) - t0 ))s; version $(curl -s -m 5 localhost:8141/version)"
  keylines $n | sed "s/^/[$a] /"
  tests $a $2; local rc=$?
  echo "[$a] errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  docker rm -f $n >/dev/null 2>&1
  kill $wp 2>/dev/null
  return $rc
}

arm F1A std
arm F1Z std
arm FMA mtp
arm FMZ mtp
arm FMB mtp
arm F1B std
echo "VALIDATE_DONE $(date +%T)"
