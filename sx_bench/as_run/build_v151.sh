#!/bin/bash
# build_v151.sh : image with the OFFICIAL release wheel 1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl (sha256 757f2200...4959)
# on Ubuntu 24.04 (the wheel needs GLIBC >= 2.38; both official images are Ubuntu 22.04 / glibc 2.35).
# Dependencies, the CUDA 12.8 toolkit and NCCL come from the official main@e53d02171 image: its installed pins equal the
# pins the wheel declares (torch 2.10.0+cu128, flashinfer-python 0.6.11.post2, tilelang 0.1.10, nvidia-cutlass-dsl 4.7.0).
# Only the vllm package and the bundled flash_attn_v100 are replaced, installed from the wheel with --no-deps.
# Guarded like the other builds: stops if MemAvailable < 30 GiB, docker root free < 25 GB, or production 8031 stops answering.
set -u
D=/mnt/2t/build/official-v151
HEAD=shixiang/1cat-vllm-v100:official-main-e53d02171-sm70main
TAG=shixiang/1cat-vllm-v100:official-v1.5.1-ubuntu2404-sm70main
cd $D
M=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
H=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health)
echo "preconditions: MemAvailable ${M} GiB, 8031 /health $H, docker root free $(df -BG --output=avail /var/lib/docker | tail -1 | tr -dc 0-9) GB"
if [ "$H" != 200 ] || [ "$M" -lt 100 ]; then echo "PRECONDITION_FAILED"; exit 1; fi
( cd ctx && sha256sum -c SHA256SUMS ) || { echo "wheel checksum mismatch"; exit 1; }
echo "files the Dockerfile copies out of the official image:"
docker run --rm --entrypoint sh $HEAD -c 'ls /usr/lib/x86_64-linux-gnu/libnccl* /usr/include/nccl.h /usr/local/cuda-12.8/bin/nvcc 2>&1' | head -8
python3 $D/gen_dockerfile_v151.py || exit 1
rm -f build.log build_watch.log
( cd ctx && DOCKER_BUILDKIT=1 docker build --network host --progress plain -t $TAG . > ../build.log 2>&1; echo "== BUILD_EXIT $?" >> ../build.log ) &
sleep 3
BP=$(ps -eo pid,cmd | awk '$2=="docker" && $3=="build" {print $1}' | head -1)
echo "docker build pid: $BP"
(
  ROOT=$(docker info -f '{{.DockerRootDir}}'); bad=0; n=0
  while kill -0 $BP 2>/dev/null; do
    m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); dsk=$(df -BG --output=avail $ROOT | tail -1 | tr -dc 0-9)
    if [ "$m" -lt 30 ] || [ "$dsk" -lt 25 ]; then echo "$(date +%T) ABORT mem=${m} disk=${dsk}" >> build_watch.log; kill -INT $BP; echo "== ABORTED" >> build.log; break; fi
    if [ $((n % 6)) = 0 ]; then
      h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health); if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) mem_avail=${m}GiB disk=${dsk}GB load=$(cut -d' ' -f1 /proc/loadavg) 8031=$h" >> build_watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) ABORT 8031 health failed" >> build_watch.log; kill -INT $BP; echo "== ABORTED" >> build.log; break; fi
    fi
    n=$((n + 1)); sleep 10
  done
) > /dev/null 2>&1 < /dev/null &
sleep 25
date +%T; grep -E '^#[0-9]+ \[' build.log | tail -3 | cut -c1-150; tail -2 build.log | cut -c1-150
