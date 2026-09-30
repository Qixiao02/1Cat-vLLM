#!/bin/bash
# Wheel and image of the fork (github.com/Qixiao02/1Cat-vLLM, branch 1cat-vllm-heavily-modified-v1) built from source
# with the repository's own docker/Dockerfile, the same recipe as the official d30469863 build of 2026-09-30
# (/mnt/2t/build/official-d304698/build_official.sh, see its header for the reasons behind each argument):
# CUDA 12.8.1, torch_cuda_arch_list 7.0, max_jobs 128 (16 compile jobs), GIT_REPO_CHECK 0 with the check done here,
# GitHub traffic through the temporary egress proxy.
# Version: SETUPTOOLS_SCM_PRETEND_VERSION below; setup.py appends ".cu128" because the build CUDA is not the main one.
set -u
cd /mnt/2t/build/fork-wheel-1001/src
V=1.5.1+heavily.modified.v1.1001.torch2.10
TAG=shixiang/1cat-vllm-v100:heavily-modified-v1-1001-sm70main
BUILD_TAG=vllm-ci:build-image-sm70-fork-1001
P=http://10.200.4.2:18080
NP=localhost,127.0.0.1,10.200.4.2
COMMON=(--network host
  --build-arg HTTP_PROXY=$P --build-arg HTTPS_PROXY=$P --build-arg http_proxy=$P --build-arg https_proxy=$P
  --build-arg NO_PROXY=$NP --build-arg no_proxy=$NP
  --build-arg max_jobs=128 --build-arg GIT_REPO_CHECK=0 --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$V
  --build-arg CUDA_VERSION=12.8.1 --build-arg torch_cuda_arch_list=7.0
  --build-context gitmirror=/mnt/2t/build/fork-wheel-1001/gitmirror
  --progress plain -f docker/Dockerfile)
# The only local change allowed is ../dockerfile-local-gitmirror.patch: CMake clones cutlass, DeepGEMM, FlashMLA,
# qutlass, triton and flash-attention-v100 from local mirrors (gitmirror/, cloned through ghfast.top; the pinned commit
# hashes were checked in the mirrors, the two tags cutlass v4.4.2 and triton v3.5.1 against the GitHub API).
[ "$(git status --porcelain | tr -d " 
")" = "Mdocker/Dockerfile" ] || { echo "== FAILED host repo check"; exit 1; }
echo "== START $(date '+%F %T') $(git rev-parse HEAD) version $V"
DOCKER_BUILDKIT=1 docker build "${COMMON[@]}" --target base . || { echo "== FAILED base $(date '+%F %T')"; exit 1; }
echo "== BASE DONE $(date '+%F %T')"
# The wheel first: it is what gets released, and the image stage reuses this build.
DOCKER_BUILDKIT=1 docker build "${COMMON[@]}" --target build --tag $BUILD_TAG . || { echo "== FAILED wheel $(date '+%F %T')"; exit 1; }
mkdir -p ../artifacts
docker run --rm -v /mnt/2t/build/fork-wheel-1001/artifacts:/artifacts_host \
  -v /mnt/2t/build/official-d304698/src/tools/check_sm70_release_artifact.py:/check_sm70_release_artifact.py:ro $BUILD_TAG \
  bash -c 'cp -r dist /artifacts_host && chmod -R a+rw /artifacts_host; ls -la dist; python3 /check_sm70_release_artifact.py dist/*.whl; echo "upstream artifact check exit $? (informational: written for upstream main d30469863)"'
ls -la ../artifacts/dist 2>/dev/null
echo "== WHEEL DONE $(date '+%F %T')"
DOCKER_BUILDKIT=1 docker build "${COMMON[@]}" --target vllm-openai --tag $TAG . || { echo "== FAILED image $(date '+%F %T')"; exit 1; }
docker images $TAG
echo "== ALL DONE $(date '+%F %T')"
