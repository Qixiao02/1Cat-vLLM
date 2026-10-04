"""mk_prod_cache.py <old compose> <new compose> : production 8031 compose with the startup compile-cache
reuse enabled (task-2 lever).  The compose on disk is the 1004 / 32-sequence one
(image heavily-modified-v1-1004-sm70main, --max-num-seqs=32, MoE cap 320).

Only two lines change: `VLLM_DISABLE_COMPILE_CACHE: "1"` -> `"0"` and a new
`SX_OPT_COMPILE_CACHE: "1"`.  Semantics (vllm/compilation/sx_compile_cache.py:186-206): the cache is
refused only when VLLM_DISABLE_COMPILE_CACHE is *explicitly* "1"; "0" or unset both let
plan_policy()/apply_policy() take over, which forces VLLM_USE_AOT_COMPILE=0 (subgraph mode, no AOT
FX-graph reload) and leaves the disable switch unset.  We deliberately do NOT add
VLLM_USE_AOT_COMPILE (explicit =1 in subgraph mode is the upstream parity-failure path,
sx_compile_cache.py:233-241) and do NOT add VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING.

Everything else is asserted to be present so a silent edit of the production file fails here instead of
at container start-up.  Dry run first, and only roll after sx_tests/compile-cache/cache_parity.sh has
passed on the new image (cold/warm token-level parity + control arm).
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

# --- preconditions: the 1004 / 32-sequence production file must be intact ---
assert IMG in s, "the 1004 image line is gone; refusing to guess"
assert "/usr/local/cuda-12.8" not in s, "the host CUDA toolkit mount is back"
assert "entrypoint.sh, target: /app/entrypoint.sh" not in s, "the entrypoint.sh mount is back"
assert 'test: ["CMD-SHELL", "curl -sf http://127.0.0.1:8001/health >/dev/null || exit 1"]' in s, "health check changed"
assert 'entrypoint: ["vllm", "serve"]' in s, "the entrypoint changed"
assert 'VLLM_SM70_QWEN38_HYBRID_PLE: "0"' in s, "the PLE hybrid switch is gone"
assert 'VLLM_QWEN4EXP_PLE_HOST_GIB: "12"' in s, "the PLE host budget is gone"
assert 'VLLM_DISABLE_COMPILE_CACHE: "1"' in s, "the compile-cache switch is not in its 1004 state"
assert "SX_OPT_COMPILE_CACHE" not in s, "the cache switch is already there"
assert "VLLM_USE_AOT_COMPILE" not in s, "an AOT switch appeared; do not add one here"
assert "VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING" not in s, "the profiling escape hatch appeared"

flags = [
    "/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4", "--served-model-name=Swift-1.5-Qwen3.8-Flash-Next",
    "--host=0.0.0.0", "--port=8001", "--tensor-parallel-size=4", "--dtype=half",
    "--attention-backend=FLASH_ATTN_V100", "--max-model-len=131072", "--max-num-seqs=32",
    "--max-num-batched-tokens=8192", "--gpu-memory-utilization=0.90", "--kv-cache-dtype=auto",
    "--trust-remote-code", "--enable-prefix-caching", "--enable-chunked-prefill", "--enable-auto-tool-choice",
    "--tool-call-parser=qwen3_coder", "--reasoning-parser=qwen3", "--language-model-only",
]
missing = [f for f in flags if ("'%s'" % f) not in s]
assert not missing, ("production flags missing from the compose", missing)
assert "      - '--speculative-config" not in s, "MTP must stay off in production"
cmd_lines = re.findall(r"^      - '.*'$", s, flags=re.M)
assert len(cmd_lines) == len(flags), ("unexpected command list", len(cmd_lines), len(flags))
assert 'VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: "320"' in s, "the MoE autotune cap is not 320"

# --- 1. compile-cache reuse on (disable switch 1 -> 0 plus the opt-in) ---
sub(r'^      VLLM_DISABLE_COMPILE_CACHE: "1"$',
    '      # SX_OPT_COMPILE_CACHE=1: reuse the compiled subgraphs across restarts (parity-verified by\n'
    '      # sx_tests/compile-cache/cache_parity.sh); VLLM_DISABLE_COMPILE_CACHE 1 -> 0. Only an explicit\n'
    '      # "1" refuses the cache, so "0" lets the switch take over (it forces VLLM_USE_AOT_COMPILE=0).\n'
    '      SX_OPT_COMPILE_CACHE: "1"\n'
    '      VLLM_DISABLE_COMPILE_CACHE: "0"')

# --- 2. header: record the change, keep the 32-sequence 1004 history ---
old_header = (
    "# 2026-10-04: --max-num-seqs 24 -> 32 and the MoE autotune cap 240 -> 320 (same 1004 image, no rebuild); the compose\n"
)
new_header = (
    "# 2026-10-04: startup compile-cache reuse on (SX_OPT_COMPILE_CACHE=1, VLLM_DISABLE_COMPILE_CACHE 1 -> 0): the\n"
    "# 148 s of Inductor codegen/compile on every restart is skipped; answers, KV size and throughput unchanged.\n"
    "# Cold/warm token parity gated by sx_tests/compile-cache/cache_parity.sh (README group 14). Everything below is\n"
    "# the unchanged 32-sequence 1004 file.\n"
    "# 2026-10-04: --max-num-seqs 24 -> 32 and the MoE autotune cap 240 -> 320 (same 1004 image, no rebuild); the compose\n"
)
assert s.count(old_header) == 1, "unexpected 32-sequence header block"
s = s.replace(old_header, new_header, 1)

assert 'VLLM_DISABLE_COMPILE_CACHE: "1"' not in s, "the old disable switch is still there"
assert 'SX_OPT_COMPILE_CACHE: "1"' in s, "the new switch is missing"
open(dst, "w").write(s)
print("written", dst)
