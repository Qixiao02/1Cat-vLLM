"""gen_dockerfile_v151.py : writes /mnt/2t/build/official-v151/ctx/Dockerfile (see build_v151.sh)."""
import json
import subprocess

HEAD = "shixiang/1cat-vllm-v100:official-main-e53d02171-sm70main"
env = json.loads(subprocess.check_output(["docker", "image", "inspect", HEAD, "-f", "{{json .Config.Env}}"]))
keep = []
for e in env:
    k, _, v = e.partition("=")
    if k in ("PATH", "HOME", "HOSTNAME") or k.startswith(("NV_", "NVIDIA_REQUIRE", "NVARCH", "VLLM_BUILD_", "VLLM_IMAGE_TAG", "CUDA_VERSION")):
        continue
    keep.append((k, v))
envlines = "\n".join("ENV %s=%s" % (k, json.dumps(v)) for k, v in keep)

TEMPLATE = r"""# Official 1Cat-vLLM v1.5.1 release wheel on Ubuntu 24.04 (the wheel needs GLIBC >= 2.38; the official images are 22.04).
# Dependencies, CUDA 12.8 toolkit and NCCL come from the official main@e53d02171 image; their pins equal the wheel's
# declared pins. Only the vllm package (and the bundled flash_attn_v100) is replaced by the release wheel (--no-deps).
FROM nvidia/cuda:12.8.1-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /vllm-workspace
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && sed -i 's#http://archive.ubuntu.com/ubuntu#http://mirrors.aliyun.com/ubuntu#g; s#http://security.ubuntu.com/ubuntu#http://mirrors.aliyun.com/ubuntu#g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv g++ gcc make curl ca-certificates sudo libibverbs-dev libnuma-dev numactl libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 && ln -sf /usr/bin/python3.12-config /usr/bin/python3-config \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED
COPY --from=@HEAD@ /usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN ln -sfn /usr/local/cuda-12.8 /usr/local/cuda && ln -sfn /usr/local/cuda-12.8 /usr/local/cuda-12 \
    && printf '/usr/local/cuda/targets/x86_64-linux/lib\n/usr/local/cuda/lib64\n' > /etc/ld.so.conf.d/zz-cuda.conf
COPY --from=@HEAD@ /usr/lib/x86_64-linux-gnu/libnccl* /usr/lib/x86_64-linux-gnu/
COPY --from=@HEAD@ /usr/include/nccl.h /usr/include/nccl.h
RUN ldconfig
COPY --from=@HEAD@ /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=@HEAD@ /usr/local/bin /usr/local/bin
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
@ENVLINES@
COPY 1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl /tmp/
RUN python3 -m pip uninstall -y 1cat-vllm flash_attn_v100 flash-attn-v100 2>&1 | tail -3 \
    && uv pip install --system --no-deps /tmp/1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl 2>&1 | tail -4 \
    && rm -f /tmp/1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl \
    && python3 -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)"
LABEL org.opencontainers.image.title="1Cat-vLLM official release v1.5.1 wheel on Ubuntu 24.04, SM70 for V100" \
      org.opencontainers.image.source="https://github.com/1CatAI/1Cat-vLLM" \
      org.opencontainers.image.version="1.5.1" \
      org.shixiang.source.describe="official GitHub release v1.5.1 (tag commit 589a5d1e1, 2026-10-03T01:32:37+08:00), release published 2026-10-03T08:06:33+08:00" \
      org.shixiang.build.wheel="1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl sha256=757f2200ba6bd20f7e3e5c8fc68c7e05461df6d284baf676ac0c16ce19434959 (matches the release digest and SHA256SUMS)" \
      org.shixiang.build.recipe="nvidia/cuda:12.8.1-base-ubuntu24.04 (glibc 2.39, the wheel needs >= 2.38) + apt python3.12 g++ numactl; CUDA 12.8 toolkit, NCCL and the Python dependencies copied from the official main e53d02171 image (same pins as the wheel declares); vllm and flash_attn_v100 replaced by the release wheel, installed with --no-deps"
ENTRYPOINT ["vllm", "serve"]
"""
text = TEMPLATE.replace("@HEAD@", HEAD).replace("@ENVLINES@", envlines)
open("/mnt/2t/build/official-v151/ctx/Dockerfile", "w").write(text)
print("Dockerfile written, %d ENV lines carried over from the official image" % len(keep))
