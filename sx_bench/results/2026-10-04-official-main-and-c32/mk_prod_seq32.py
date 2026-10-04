"""mk_prod_seq32.py <old compose> <new compose> : production 8031 compose switched to 32 concurrent sequences.

The compose on disk is the 1004 one (image heavily-modified-v1-1004-sm70main). Only three things change:
`--max-num-seqs=24` -> `=32`, the MoE autotune cap `VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS` "240" -> "320"
(with its comment), and a header line recording the change. Everything else the 1003/1004 upgrades
introduced is asserted to be present (image line, 19 flags, no `--speculative-config`, health check,
entrypoint, compile-cache switch, PLE switches, util 0.90, max-model-len 131072) so a silent edit of the
production file is caught here instead of at container start-up.

Why the two knobs move together: `vllm/model_executor/warmup/awq_sm70_warmup.py:892` divides the cap by the
experts per token (top-k 10 here), so 240 only tunes up to 24 routed rows and 320 tunes up to 32. Measured
on the comparison host (README group 13): same engine 24 -> 32 = +25.4% aggregate, -5.9% per stream,
step +1.8 ms; 24-lane loads fall out of the captured graph set and are ~4% slower at that single point.
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub(pattern, repl, count=1):
    global s
    s, n = re.subn(pattern, lambda m: repl, s, count=count, flags=re.M)
    assert n == count, (pattern, n)


IMG = "shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main"

# --- preconditions: the 1004 production file must be intact ---
assert IMG in s, "the 1004 image line is gone; refusing to guess"
assert "/usr/local/cuda-12.8" not in s, "the host CUDA toolkit mount is back"
assert "entrypoint.sh, target: /app/entrypoint.sh" not in s, "the entrypoint.sh mount is back"
assert 'test: ["CMD-SHELL", "curl -sf http://127.0.0.1:8001/health >/dev/null || exit 1"]' in s, "health check changed"
assert 'VLLM_DISABLE_COMPILE_CACHE: "1"' in s, "the compile-cache switch is gone"
assert 'entrypoint: ["vllm", "serve"]' in s, "the entrypoint changed"
assert 'VLLM_SM70_QWEN38_HYBRID_PLE: "0"' in s, "the PLE hybrid switch is gone"
assert 'VLLM_QWEN4EXP_PLE_HOST_GIB: "12"' in s, "the PLE host budget is gone"

flags = [
    "/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4", "--served-model-name=Swift-1.5-Qwen3.8-Flash-Next",
    "--host=0.0.0.0", "--port=8001", "--tensor-parallel-size=4", "--dtype=half",
    "--attention-backend=FLASH_ATTN_V100", "--max-model-len=131072", "--max-num-seqs=24",
    "--max-num-batched-tokens=8192", "--gpu-memory-utilization=0.90", "--kv-cache-dtype=auto",
    "--trust-remote-code", "--enable-prefix-caching", "--enable-chunked-prefill", "--enable-auto-tool-choice",
    "--tool-call-parser=qwen3_coder", "--reasoning-parser=qwen3", "--language-model-only",
]
missing = [f for f in flags if ("'%s'" % f) not in s]
assert not missing, ("production flags missing from the compose", missing)
assert "      - '--speculative-config" not in s, "MTP must stay off in production"
assert "--max-num-seqs=32" not in s, "already switched to 32"
cmd_lines = re.findall(r"^      - '.*'$", s, flags=re.M)
assert len(cmd_lines) == len(flags), ("unexpected command list", len(cmd_lines), len(flags))
assert 'VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: "240"' in s, "the MoE autotune cap is not 240"

# --- 1. concurrency 24 -> 32 ---
sub(r"^      - '--max-num-seqs=24'$", "      - '--max-num-seqs=32'")

# --- 2. MoE autotune cap 240 -> 320 (it is divided by top-k 10: 240 covers 24 rows, 320 covers 32) ---
sub(r"^      # Autotune the TurboMind grouped GEMM up to 240 routed rows so the M24 decode width is tuned \(default 128\)$",
    "      # Autotune the TurboMind grouped GEMM up to 320 routed rows so the M32 decode width is tuned (default 128)")
sub(r'^      VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: "240"$',
    '      VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: "320"')

# --- 3. header: record the change, keep the 1004 upgrade history ---
old_header = (
    "# 2026-10-04: upgraded to the 1004 release wheel (image heavily-modified-v1-1004-sm70main). Previous version: image\n"
)
new_header = (
    "# 2026-10-04: --max-num-seqs 24 -> 32 and the MoE autotune cap 240 -> 320 (same 1004 image, no rebuild); the compose\n"
    "# before this change is kept next to this file as *.bak-seq32-pre-upgrade-*. Everything below is unchanged 1004.\n"
    "# 2026-10-04: upgraded to the 1004 release wheel (image heavily-modified-v1-1004-sm70main). Previous version: image\n"
)
assert s.count(old_header) == 1, "unexpected 1004 header block"
s = s.replace(old_header, new_header, 1)

assert "--max-num-seqs=24" not in s, "the old concurrency flag is still there"
assert 'VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: "320"' in s, "the new MoE cap is missing"
open(dst, "w").write(s)
print("written", dst)
