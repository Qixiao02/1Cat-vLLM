"""arm_compose_1003.py <arm> : compose file for one arm of the comparison "our fork e673bd168 (2026-10-03 12:23) vs
official main@e53d02171 (2026-10-03 11:48)" on GPU 4-7, port 127.0.0.1:8141. Every arm serves
/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 with `vllm serve`. Volumes, security options and ulimits come from the 8031
compose; the environment and the serve flags are set here per arm. Compose files go to /mnt/2t/build/cmp1003/.

Official arms use the official project's own current recommendation (docs/design/sm70_qwen38_concurrency40.md,
sm70_flash_next_batch_defaults.md, scripts/serve_flash_next_nvfp4_v100.sh at e53d02171). Since b75d3cd0f the batch /
MTP acceleration routes are DEFAULT ON there, so they are not exported; /v1/sm70/acceleration is recorded instead.

  O1  official, no MTP, upstream's benchmark contract: prefix caching off, --mamba-cache-mode align, max-num-seqs 16,
      gpu util 0.94, OMP_NUM_THREADS=1, spawn workers
  O2  O1 with prefix caching on
  F1  our fork, production settings: prefix caching on, max-num-seqs 24, gpu util 0.90, production switches
  F2  F1 with prefix caching off
  OM  official + native MTP k=4, release launcher settings (align, prefix on, --no-async-scheduling) with the
      qualified batch settings (prefill budget 4096, max-num-seqs 16, 32K context), util 0.95
  OM2 OM at util 0.92 (fallback if OM does not start: upstream says the 1.25 GiB/rank packs need KV headroom)
  FM  our fork + native MTP k=4: prefix caching on, max-num-seqs 16, util 0.87 (the fork's MTP setting)
  FM2 FM with the steady-state KV budget (SX_OPT_KV_STEADY_BUDGET=1, reserve 1990 MiB) at util 0.93
The compile cache is disabled on every arm (official main turns it on by default and documents a wrong-kernel risk with
AOT=1 on unpatched torch 2.10; the fork's own cache is off by default). It does not change steady-state speed."""
import re
import sys

arm = sys.argv[1]
W = "/mnt/2t/build/cmp1003"
SRC = "/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml"
import os
OFFICIAL = os.environ.get("OFFICIAL_IMAGE", "shixiang/1cat-vllm-v100:official-main-e53d02171-sm70main")
FORK = "shixiang/1cat-vllm-v100:heavily-modified-v1-e673bd168-sm70main"

COMMON_ENV = {
    "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    "VLLM_QWEN4EXP_PLE_HOST_GIB": "12", "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
    "HOME": "/cache/home", "XDG_CACHE_HOME": "/cache", "HF_HOME": "/cache/huggingface",
    "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub", "TORCH_HOME": "/cache/torch",
    "TRITON_CACHE_DIR": "/cache/triton", "TILELANG_CACHE_DIR": "/cache/tilelang",
    "USER": "sx-inference", "LOGNAME": "sx-inference",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
}
OFFICIAL_ENV = {"OMP_NUM_THREADS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HF_HUB_OFFLINE": "1"}
FORK_ENV = {  # production 8031 switches
    "OMP_NUM_THREADS": "8", "VLLM_SM70_QWEN38_HYBRID_PLE": "0", "VLLM_PLE_CPU_OFFLOAD": "0",
    "VLLM_PLE_DISK_OFFLOAD": "0", "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "1",
    "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "240",
}
FORK_MTP_ENV = {**FORK_ENV, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "1200"}
MTP = '{"method":"mtp","num_speculative_tokens":4}'
ALIGN = ["--mamba-cache-mode=align"]
ARMS = {
    "O1": dict(image=OFFICIAL, prefix=False, seqs=16, util="0.94", maxlen=131072, mtp=False, batched=8192,
               env=OFFICIAL_ENV, extra=ALIGN),
    "O1old": dict(image=OFFICIAL, prefix=False, seqs=16, util="0.94", maxlen=131072, mtp=False, batched=8192,
                  env={"OMP_NUM_THREADS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                       "VLLM_SM70_QWEN38_BATCH_FASTPATH": "1"}, extra=[]),
    "O2": dict(image=OFFICIAL, prefix=True, seqs=16, util="0.94", maxlen=131072, mtp=False, batched=8192,
               env=OFFICIAL_ENV, extra=ALIGN),
    "F1": dict(image=FORK, prefix=True, seqs=24, util="0.90", maxlen=131072, mtp=False, batched=8192,
               env=FORK_ENV, extra=[]),
    "F2": dict(image=FORK, prefix=False, seqs=24, util="0.90", maxlen=131072, mtp=False, batched=8192,
               env=FORK_ENV, extra=[]),
    "OM": dict(image=OFFICIAL, prefix=True, seqs=16, util="0.95", maxlen=32768, mtp=True, batched=4096,
               env=OFFICIAL_ENV, extra=ALIGN + ["--no-async-scheduling"]),
    "OM2": dict(image=OFFICIAL, prefix=True, seqs=16, util="0.92", maxlen=32768, mtp=True, batched=4096,
                env=OFFICIAL_ENV, extra=ALIGN + ["--no-async-scheduling"]),
    "FM": dict(image=FORK, prefix=True, seqs=16, util="0.87", maxlen=32768, mtp=True, batched=8192,
               env=FORK_MTP_ENV, extra=[]),
    "FM2": dict(image=FORK, prefix=True, seqs=16, util="0.93", maxlen=32768, mtp=True, batched=8192,
                env={**FORK_MTP_ENV, "SX_OPT_KV_STEADY_BUDGET": "1", "SX_OPT_KV_STEADY_RESERVE_MIB": "1990"}, extra=[]),
}


V151_CACHE = "/opt/shixiang-inference/cache-official-v151-flashnext-tp4"
V151_BASE = {"OMP_NUM_THREADS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
V151_MTP_ENV = {
    "VLLM_SM70_QWEN38_GDN_INPUT_BATCH": "1", "VLLM_SM70_NVFP4_MOE_GROUPED_MTP5": "1",
    "VLLM_SM70_RMSNORM_GATED_EXACT": "1", "VLLM_SM70_MTP_MOE_FP16_EXACT": "1",
    "VLLM_SM70_MTP_HC_BATCH": "1", "VLLM_SM70_MTP_HC_COOPERATIVE": "1", "VLLM_SM70_MTP_ROUTER_BATCH": "1",
    "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV": "1", "VLLM_SM70_MTP_HC_FULL_UNROLL": "1", "VLLM_SM70_QSA_MTP_TOPK": "1",
    "VLLM_SM70_MTP_ROUTER_TOP16": "1", "VLLM_SM70_MTP_SHARED_BATCH": "1", "VLLM_SM70_MTP_PLE_CONV": "1",
}
ARMS["V1"] = dict(image=OFFICIAL, prefix=False, seqs=16, util="0.94", maxlen=131072, mtp=False, batched=8192,
                  env={**V151_BASE, "VLLM_SM70_QWEN38_BATCH_FASTPATH": "1"}, extra=[], cache=V151_CACHE)
ARMS["V2"] = dict(ARMS["V1"], prefix=True)
ARMS["V2b"] = dict(ARMS["V2"], util="0.90")
ARMS["VM"] = dict(image=OFFICIAL, prefix=True, seqs=16, util="0.95", maxlen=32768, mtp=True, batched=8192,
                  env={**V151_BASE, **V151_MTP_ENV}, extra=["--no-async-scheduling"], cache=V151_CACHE)
ARMS["VMa"] = dict(ARMS["VM"], util="0.92", batched=4096)
ARMS["VMb"] = dict(ARMS["VM"], util="0.90", batched=4096)
ARMS["V2c"] = dict(ARMS["V2"], util="0.92")
ARMS["VM2"] = dict(ARMS["VM"], util="0.92")
a = ARMS[arm]
cache = a.get("cache") or ("/opt/shixiang-inference/cache-official-flashnext-tp4" if a["image"] == OFFICIAL
         else "/opt/shixiang-inference/cache-forkwheel-flashnext-tp4")
s = open(SRC).read()


def sub(pattern, repl, count=1):
    global s
    s, n = re.subn(pattern, lambda m: repl, s, count=count, flags=re.M)
    assert n == count, (pattern, n)


s = "# Comparison arm %s (see %s/arm_compose_1003.py). Temporary.\n" % (arm, W) + s[s.index("\nname: ") + 1:]
sub(r"^name: .*$", "name: sx-cmp-%s" % arm.lower())
sub(r"^  [a-z0-9-]+:\n(?=(?:    #.*\n)*    image: )", "  cmp:\n")
s = re.sub(r"(  cmp:\n)(?:    #.*\n)+", r"\1", s, count=1)
sub(r"^    image: .*$", "    image: " + a["image"])
sub(r"^    container_name: .*$", "    container_name: sx-cmp-%s" % arm.lower())
sub(r"^    restart: .*$", '    restart: "no"')
sub(r'^    ports: \["0\.0\.0\.0:8031:8001"\]$', '    ports: ["127.0.0.1:8141:8001"]')
sub(r"^      - \{type: bind, source: /opt/shixiang-inference/cache-flashnext-tp4, target: /cache\}$",
    "      - {type: bind, source: %s, target: /cache}" % cache)
sub(r"^      - \{type: bind, source: /usr/local/cuda-12\.8, target: /usr/local/cuda, read_only: true\}\n", "")
sub(r"^      - \{type: bind, source: /opt/shixiang-aihub/deploy/inference/entrypoint\.sh, target: /app/entrypoint\.sh, read_only: true\}\n", "")
env_start = s.index("    environment:\n") + len("    environment:\n")
env_end = s.index("    deploy:\n")
env = {**COMMON_ENV, **a["env"]}
s = s[:env_start] + "".join('      %s: "%s"\n' % kv for kv in env.items()) + s[env_end:]
sub(r'^              device_ids: \["0","1","2","3"\]$', '              device_ids: ["4","5","6","7"]')
sub(r'^      test: \["CMD", "/app/healthcheck\.sh"\]$',
    '      test: ["CMD-SHELL", "curl -sf http://127.0.0.1:8001/health >/dev/null || exit 1"]')
flags = [
    "/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4", "--served-model-name=Swift-1.5-Qwen3.8-Flash-Next",
    "--host=0.0.0.0", "--port=8001", "--tensor-parallel-size=4", "--dtype=half",
    "--attention-backend=FLASH_ATTN_V100", "--max-model-len=%d" % a["maxlen"], "--max-num-seqs=%d" % a["seqs"],
    "--max-num-batched-tokens=%d" % a["batched"], "--gpu-memory-utilization=%s" % a["util"], "--kv-cache-dtype=auto",
    "--trust-remote-code", "--enable-prefix-caching" if a["prefix"] else "--no-enable-prefix-caching",
    "--enable-chunked-prefill", "--enable-auto-tool-choice", "--tool-call-parser=qwen3_coder",
    "--reasoning-parser=qwen3", "--language-model-only",
] + a["extra"]
for _pfx, _new in a.get("replace", {}).items():
    flags = [f for f in flags if not (f == _pfx or f.startswith(_pfx + "="))]
    if _new:
        flags.append(_new)
if a["mtp"]:
    flags.append("--speculative-config=" + MTP)
tail = s.index('    entrypoint: ["/app/entrypoint.sh"]')
s = s[:tail] + '    entrypoint: ["vllm", "serve"]\n    command:\n' + "".join(
    "      - '%s'\n" % f for f in flags)
out = "%s/compose.%s.yaml" % (W, arm)
open(out, "w").write(s)
print(out)
