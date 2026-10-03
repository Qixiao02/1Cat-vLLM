#!/bin/bash
# build_1003.sh : the release wheel of the fork at commit e673bd168 (branch sx/mtp-integrate), built with the repository's own
# docker/Dockerfile exactly like the 1002 wheel (same build args, so the csrc layers come from the build cache: csrc, CMake and
# setup.py are identical between 8593319b2 and e673bd168). Only the wheel is built (target build). The only local change to the
# Dockerfile is ../dockerfile-local-gitmirror.patch (CMake third-party clones from verified local mirrors).
# Guarded: stops if MemAvailable < 30 GiB, docker root free < 25 GB, or production 8031 stops answering.
set -u
S=/mnt/2t/build/fork-wheel-1001
cd $S/src
FULL=e673bd16896880534f7a4cc5231650e7f6588da3
M=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); H=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health)
echo "preconditions: MemAvailable ${M} GiB, 8031 /health $H, docker root free $(df -BG --output=avail /var/lib/docker | tail -1 | tr -dc 0-9) GB"
if [ "$H" != 200 ] || [ "$M" -lt 100 ]; then echo "PRECONDITION_FAILED"; exit 1; fi

echo "== sync the source tree to $FULL"
git fetch $S/mtp-1003.bundle sx/mtp-integrate 2>&1 | tail -2
git checkout -- docker/Dockerfile
git checkout -q --detach $FULL || { echo "checkout failed"; exit 1; }
git apply --check ../dockerfile-local-gitmirror.patch && git apply ../dockerfile-local-gitmirror.patch || { echo "dockerfile patch does not apply"; exit 1; }
echo "HEAD $(git rev-parse HEAD) status: $(git status --porcelain | tr -d '\n')"
[ "$(git rev-parse HEAD)" = "$FULL" ] || { echo "wrong commit"; exit 1; }
[ "$(git status --porcelain | tr -d ' \n')" = "Mdocker/Dockerfile" ] || { echo "== FAILED host repo check"; exit 1; }

echo "== egress proxy (only used if a layer has to download from GitHub)"
docker rm -f sx-build-egress-proxy >/dev/null 2>&1
docker run -d --name sx-build-egress-proxy --network container:ubuntu-overseas-ssh -v /mnt/2t/build/official-d304698/connect_proxy.py:/p.py:ro python:3.10-slim-bookworm python3 -u /p.py 0.0.0.0 18080 >/dev/null && sleep 3 && docker logs sx-build-egress-proxy 2>&1 | tail -1

V=1.5.1+heavily.modified.v1.1003.torch2.10
BUILD_TAG=vllm-ci:build-image-sm70-fork-1003
P=http://10.200.4.2:18080
NP=localhost,127.0.0.1,10.200.4.2
COMMON=(--network host
  --build-arg HTTP_PROXY=$P --build-arg HTTPS_PROXY=$P --build-arg http_proxy=$P --build-arg https_proxy=$P
  --build-arg NO_PROXY=$NP --build-arg no_proxy=$NP
  --build-arg max_jobs=128 --build-arg GIT_REPO_CHECK=0 --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$V
  --build-arg CUDA_VERSION=12.8.1 --build-arg torch_cuda_arch_list=7.0
  --build-context gitmirror=$S/gitmirror
  --progress plain -f docker/Dockerfile)
echo "== START $(date '+%F %T') $FULL version $V"
( DOCKER_BUILDKIT=1 docker build "${COMMON[@]}" --target base . > ../build_1003_base.log 2>&1 && echo "== BASE DONE $(date '+%F %T')" > ../build_1003.log && DOCKER_BUILDKIT=1 docker build "${COMMON[@]}" --target build --tag $BUILD_TAG . >> ../build_1003.log 2>&1; echo "== BUILD_EXIT $?" >> ../build_1003.log ) < /dev/null > /dev/null 2>&1 &
sleep 3
BP=$(ps -eo pid,cmd | awk '$2=="docker" && $3=="build" {print $1}' | head -1)
echo "docker build pid: $BP"
(
  ROOT=$(docker info -f '{{.DockerRootDir}}'); bad=0; n=0
  while ps -eo cmd | awk '$1=="docker" && $2=="build"' | grep -q .; do
    m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); dsk=$(df -BG --output=avail $ROOT | tail -1 | tr -dc 0-9)
    if [ "$m" -lt 30 ] || [ "$dsk" -lt 25 ]; then echo "$(date +%T) ABORT mem=${m} disk=${dsk}" >> ../build_1003_watch.log; for p in $(ps -eo pid,cmd | awk '$2=="docker" && $3=="build" {print $1}'); do kill -INT $p; done; echo "== ABORTED" >> ../build_1003.log; break; fi
    if [ $((n % 6)) = 0 ]; then
      h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health); if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) mem_avail=${m}GiB disk=${dsk}GB load=$(cut -d' ' -f1 /proc/loadavg) 8031=$h" >> ../build_1003_watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) ABORT 8031 health failed" >> ../build_1003_watch.log; for p in $(ps -eo pid,cmd | awk '$2=="docker" && $3=="build" {print $1}'); do kill -INT $p; done; echo "== ABORTED" >> ../build_1003.log; break; fi
    fi
    n=$((n + 1)); sleep 10
  done
) > /dev/null 2>&1 < /dev/null &
sleep 20
date +%T; tail -3 ../build_1003_base.log 2>/dev/null | cut -c1-140; tail -2 ../build_1003.log 2>/dev/null | cut -c1-140
