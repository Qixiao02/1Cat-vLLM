#!/bin/bash
# probe_official.sh <commit> : does official 1Cat-vLLM at <commit> start on this model?
# Official main@e53d02171 hangs in warmup_kernels (GPU idle, all four workers in topk_topp_triton.py:1109 .all() sync).
# This builds a Python overlay of <commit>'s vllm/ on top of the e53d02171 image (csrc differs from v1.5.1 only by ADDED
# GGUF operators and an SM75 cmake file, nothing existing was changed, so the native part is equivalent), starts the O1 arm
# of arm_compose_1003.py on it and decides:
#   HEALTHY  /health answers
#   HANG     "Graph capturing finished" was logged and /health is still down 240 s later (normal: ~4 s), or 25 min overall
#   CRASH    the container stopped
# KEEP=1 leaves a healthy instance running (the O1 measurements can then continue on it); otherwise it is removed.
# Same memory / production-health watchdog as the main chain. One heavy thing at a time.
set -u
C=${1:-HEAD}
ARM=${ARM:-O1}
CN=sx-cmp-$(echo $ARM | tr A-Z a-z)
SRC=/mnt/2t/build/official-main/src
FULL=$(git -C $SRC rev-parse $C) || { echo "unknown commit $C"; exit 2; }
S=${FULL:0:9}
W=/mnt/2t/build/cmp1003
OV=/mnt/2t/build/official-overlay/$S
BASE=shixiang/1cat-vllm-v100:official-main-e53d02171-sm70main
TAG=shixiang/1cat-vllm-v100:official-ov-$S-sm70main
SUBJ=$(git -C $SRC log -1 --format=%s $FULL | cut -c1-90)
CT=$(git -C $SRC log -1 --format=%cI $FULL)
cd $W
if [ -n "${PROBE_IMAGE:-}" ]; then TAG=$PROBE_IMAGE; S=${PROBE_LABEL:-probe}; elif ! docker image inspect $TAG >/dev/null 2>&1; then
  rm -rf $OV && mkdir -p $OV
  git -C $SRC archive $FULL vllm | tar -x -C $OV
  cat > $OV/Dockerfile <<EOF
FROM $BASE
COPY vllm /opt/venv/lib/python3.12/site-packages/vllm
RUN sed -i "s/1\\.5\\.1\\.dev145+ge53d02171/1.5.1+python.overlay.of.$S.on.native.e53d02171/g" /opt/venv/lib/python3.12/site-packages/vllm/_version.py || true
LABEL org.opencontainers.image.revision="$FULL" org.opencontainers.image.source="https://github.com/1CatAI/1Cat-vLLM" org.shixiang.source.commit_time="$CT" org.shixiang.build.recipe="Python overlay of vllm/ at $S on the native build of main e53d02171 (csrc differs only by added GGUF operators and an SM75 cmake file)"
EOF
  docker build -q -t $TAG $OV > /dev/null || { echo "PROBE $S overlay build failed"; exit 2; }
fi
echo "PROBE $S ($CT) $SUBJ"
M=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
[ "$M" -lt 100 ] && { echo "PROBE $S MemAvailable ${M} GiB < 100, not starting"; exit 2; }
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)" -lt 1500 ] && break; sleep 10; done
docker rm -f $CN >/dev/null 2>&1
OFFICIAL_IMAGE=$TAG python3 $W/arm_compose_1003.py $ARM > /dev/null && docker compose -f $W/compose.$ARM.yaml config -q || { echo "PROBE $S bad compose"; exit 2; }
docker compose -f $W/compose.$ARM.yaml up -d 2>&1 | tail -1
( bad=0; n=0
  while docker inspect $CN >/dev/null 2>&1; do
    mm=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    if [ "$mm" -lt 25 ]; then echo "$(date +%T) STOP $CN: MemAvailable ${mm} GiB" >> $W/watch.log; docker stop -t 20 $CN >/dev/null 2>&1; break; fi
    if [ $((n % 6)) = 0 ]; then h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health); if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) probe $S mem_avail=${mm}GiB 8031=$h" >> $W/watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) STOP $CN: 8031 health failed" >> $W/watch.log; docker stop -t 20 $CN >/dev/null 2>&1; break; fi; fi
    n=$((n + 1)); sleep 10
  done ) &
t0=$(date +%s); cap=0; result=""
while :; do
  if curl -sf -m 5 localhost:8141/health -o /dev/null; then result="HEALTHY after $(( $(date +%s) - t0 ))s"; break; fi
  st=$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)
  if [ "$st" != running ]; then result="CRASH (container $st)"; break; fi
  if [ "$cap" = 0 ] && docker logs $CN 2>&1 | grep -q "Graph capturing finished"; then cap=$(date +%s); echo "PROBE $S graph capture finished at +$(( cap - t0 ))s"; fi
  if [ "$cap" != 0 ] && [ $(( $(date +%s) - cap )) -gt 240 ]; then result="HANG (no /health 240 s after graph capture, normal ~4 s)"; break; fi
  if [ $(( $(date +%s) - t0 )) -gt 1500 ]; then result="HANG (no /health after 25 min)"; break; fi
  sleep 10
done
echo "PROBE $S RESULT $result"
if [[ "$result" == HANG* ]]; then
  PY=/mnt/2t/build/tools/venv/bin/py-spy
  { for p in $(docker top $CN -eo pid,args 2>/dev/null | awk '/VLLM::Worker_TP/{print $1}'); do echo "===== worker pid $p"; timeout 40 $PY dump --pid $p 2>&1 | head -30; done; } > $W/probe_${S}_pyspy.txt 2>&1
  echo "PROBE $S innermost frame: $(sed -n '4,5p' $W/probe_${S}_pyspy.txt | tr '\n' ' ' | cut -c1-160)"
fi
docker logs $CN > $W/engine_probe_$S.log 2>&1
if [[ "$result" == HEALTHY* ]] && [ "${KEEP:-0}" = 1 ]; then
  echo "PROBE $S instance kept running (KEEP=1), image $TAG"
else
  docker compose -f $W/compose.$ARM.yaml down 2>&1 | tail -1
  docker rm -f $CN >/dev/null 2>&1
fi
echo "PROBE_DONE $S $(date +%T)"
