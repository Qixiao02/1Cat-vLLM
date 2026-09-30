"""arm_compose.py <arm> : compose file for one arm of the comparison "fork 1001 vs official main@d30469863" on
GPU 4-7, port 127.0.0.1:8141. Every arm serves /models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 with `vllm serve` from an
image built with the upstream docker/Dockerfile recipe. Volumes, security options and ulimits come from the 8031
compose; the environment and the serve flags are set here per arm.

  O1  official, upstream's qualified no-MTP settings: prefix caching off, max-num-seqs 16, gpu util 0.94,
      VLLM_SM70_QWEN38_BATCH_FASTPATH=1, OMP_NUM_THREADS=1, spawn workers
  O2  O1 with prefix caching on
  F1  fork 1001, production settings: prefix caching on, max-num-seqs 24, gpu util 0.90, production switches
  F2  F1 with prefix caching off
  OM  official + native MTP (k=4), upstream's qualified MTP settings: prefix caching on, max-num-seqs 16, util 0.95,
      every MTP acceleration switch of upstream's best qualified run
  FM  fork 1001 + native MTP (k=4): prefix caching on, max-num-seqs 16, util 0.87 (fork's MTP setting)
The two MTP arms use max-model-len 32768 (upstream qualified MTP at that length); the others 131072."""
import re
import sys

arm = sys.argv[1]
SRC = "/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml"
OFFICIAL = "shixiang/1cat-vllm-v100:official-d304698-sm70main"
FORK = "shixiang/1cat-vllm-v100:heavily-modified-v1-1001-sm70main"

COMMON_ENV = {
    "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    "VLLM_QWEN4EXP_PLE_HOST_GIB": "12", "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
    "HOME": "/cache/home", "XDG_CACHE_HOME": "/cache", "HF_HOME": "/cache/huggingface",
    "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub", "TORCH_HOME": "/cache/torch",
    "TRITON_CACHE_DIR": "/cache/triton", "TILELANG_CACHE_DIR": "/cache/tilelang",
    "USER": "sx-inference", "LOGNAME": "sx-inference",
}
OFFICIAL_ENV = {"OMP_NUM_THREADS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
FORK_ENV = {  # production 8031 switches
    "OMP_NUM_THREADS": "8", "VLLM_SM70_QWEN38_HYBRID_PLE": "0", "VLLM_PLE_CPU_OFFLOAD": "0",
    "VLLM_PLE_DISK_OFFLOAD": "0", "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "1",
    "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "240",
}
OFFICIAL_MTP_ENV = {  # docs/design/sm70_flash_next_mtp4_batch_gdn.md: every admitted MTP acceleration
    "VLLM_SM70_QWEN38_GDN_INPUT_BATCH": "1", "VLLM_SM70_NVFP4_MOE_GROUPED_MTP5": "1",
    "VLLM_SM70_RMSNORM_GATED_EXACT": "1", "VLLM_SM70_MTP_MOE_FP16_EXACT": "1",
    "VLLM_SM70_MTP_HC_BATCH": "1", "VLLM_SM70_MTP_HC_COOPERATIVE": "1", "VLLM_SM70_MTP_ROUTER_BATCH": "1",
    "VLLM_SM70_FUSED_SIGMOID_MIXED_QKV": "1", "VLLM_SM70_MTP_HC_FULL_UNROLL": "1", "VLLM_SM70_QSA_MTP_TOPK": "1",
    "VLLM_SM70_MTP_ROUTER_TOP16": "1", "VLLM_SM70_MTP_SHARED_BATCH": "1", "VLLM_SM70_MTP_PLE_CONV": "1",
}
MTP = '{"method":"mtp","num_speculative_tokens":4}'
ARMS = {
    "O1": dict(image=OFFICIAL, prefix=False, seqs=16, util="0.94", maxlen=131072, mtp=False,
               env={**OFFICIAL_ENV, "VLLM_SM70_QWEN38_BATCH_FASTPATH": "1"}),
    "O2": dict(image=OFFICIAL, prefix=True, seqs=16, util="0.94", maxlen=131072, mtp=False,
               env={**OFFICIAL_ENV, "VLLM_SM70_QWEN38_BATCH_FASTPATH": "1"}),
    "F1": dict(image=FORK, prefix=True, seqs=24, util="0.90", maxlen=131072, mtp=False, env=FORK_ENV),
    "F2": dict(image=FORK, prefix=False, seqs=24, util="0.90", maxlen=131072, mtp=False, env=FORK_ENV),
    "OM": dict(image=OFFICIAL, prefix=True, seqs=16, util="0.95", maxlen=32768, mtp=True,
               env={**OFFICIAL_ENV, **OFFICIAL_MTP_ENV}),
    "FM": dict(image=FORK, prefix=True, seqs=16, util="0.87", maxlen=32768, mtp=True,
               env={**FORK_ENV, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "1200"}),
}
a = ARMS[arm]
cache = ("/opt/shixiang-inference/cache-official-flashnext-tp4" if a["image"] == OFFICIAL
         else "/opt/shixiang-inference/cache-forkwheel-flashnext-tp4")
s = open(SRC).read()


def sub(pattern, repl, count=1):
    global s
    s, n = re.subn(pattern, lambda m: repl, s, count=count, flags=re.M)
    assert n == count, (pattern, n)


s = "# Comparison arm %s (see /mnt/2t/build/cmp1001/arm_compose.py). Temporary.\n" % arm + s[s.index("\nname: ") + 1:]
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
    "--max-num-batched-tokens=8192", "--gpu-memory-utilization=%s" % a["util"], "--kv-cache-dtype=auto",
    "--trust-remote-code", "--enable-prefix-caching" if a["prefix"] else "--no-enable-prefix-caching",
    "--enable-chunked-prefill", "--enable-auto-tool-choice", "--tool-call-parser=qwen3_coder",
    "--reasoning-parser=qwen3", "--language-model-only",
]
if a["mtp"]:
    flags.append("--speculative-config=" + MTP)
tail = s.index('    entrypoint: ["/app/entrypoint.sh"]')
s = s[:tail] + '    entrypoint: ["vllm", "serve"]\n    command:\n' + "".join(
    "      - '%s'\n" % f for f in flags)
out = "/mnt/2t/build/cmp1001/compose.%s.yaml" % arm
open(out, "w").write(s)
print(out)
