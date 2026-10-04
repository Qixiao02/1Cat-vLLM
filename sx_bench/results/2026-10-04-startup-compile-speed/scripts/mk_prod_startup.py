"""mk_prod_startup.py <old compose> <new compose> : production 8031 compose with BOTH startup levers of
README group 14 enabled, in one restart:

  1. compile-cache reuse across restarts: `VLLM_DISABLE_COMPILE_CACHE: "1"` -> `"0"` plus a new
     `SX_OPT_COMPILE_CACHE: "1"`.  Semantics (vllm/compilation/sx_compile_cache.py:186-206): the cache is
     refused only when VLLM_DISABLE_COMPILE_CACHE is *explicitly* "1"; "0" or unset both let
     plan_policy()/apply_policy() take over, which forces VLLM_USE_AOT_COMPILE=0 (subgraph mode, no AOT
     FX-graph reload).  We deliberately do NOT add VLLM_USE_AOT_COMPILE (explicit =1 in subgraph mode is
     the upstream parity-failure path, sx_compile_cache.py:233-241) and do NOT add
     VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING.  Measured: -30..45 s per restart (A-B-measured.md §2/§6).

  2. multithread safetensors loader: `--model-loader-extra-config {"enable_multithread_load": true,
     num_threads: 8}` (vllm/model_executor/model_loader/default_loader.py:257-302 via
     vllm/engine/arg_utils.py:913-924).  Measured on the same-shape arm: `Loading weights took`
     185.87 s (two control rounds 184.99 / 186.75) -> 117.78 s, `/health` 390.5 -> 318 s.

Everything else is asserted to be present so a silent edit of the production file fails here instead of at
container start-up.  Dry run first; only roll after sx_tests/compile-cache/cache_parity.sh has passed on
the new image (cold/warm token-level parity + control arm).
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
assert "model-loader-extra-config" not in s, "the multithread loader flag is already there"
assert "enable_multithread_load" not in s, "the multithread loader flag is already there"

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

# --- 2. multithread safetensors loader (two extra command tokens at the end of the list) ---
sub(r"^      - '--language-model-only'$",
    "      - '--language-model-only'\n"
    "      # enable_multithread_load: 8 threads instead of the single-threaded safetensors iterator\n"
    "      - '--model-loader-extra-config'\n"
    '      - \'{"enable_multithread_load": true, "num_threads": 8}\'')

# --- 3. header: record both changes, keep the 32-sequence 1004 history ---
old_header = (
    "# 2026-10-04: --max-num-seqs 24 -> 32 and the MoE autotune cap 240 -> 320 (same 1004 image, no rebuild); the compose\n"
)
new_header = (
    "# 2026-10-04: startup cut by ~100 s per restart (README group 14): SX_OPT_COMPILE_CACHE=1 with\n"
    "# VLLM_DISABLE_COMPILE_CACHE 1 -> 0 (the ~148 s of Inductor compile is skipped; cold/warm token parity gated by\n"
    "# sx_tests/compile-cache/cache_parity.sh) and --model-loader-extra-config enable_multithread_load=8 threads\n"
    "# (weight loading 186 -> 118 s in the same-shape A/B). Answers, KV pool and throughput unchanged.\n"
    "# 2026-10-04: --max-num-seqs 24 -> 32 and the MoE autotune cap 240 -> 320 (same 1004 image, no rebuild); the compose\n"
)
assert s.count(old_header) == 1, "unexpected 32-sequence header block"
s = s.replace(old_header, new_header, 1)

# --- post-conditions ---
assert 'VLLM_DISABLE_COMPILE_CACHE: "1"' not in s, "the old disable switch is still there"
assert 'SX_OPT_COMPILE_CACHE: "1"' in s, "the new switch is missing"
assert s.count("      - '--model-loader-extra-config'") == 1, "loader flag not inserted exactly once"
assert s.count('{"enable_multithread_load": true, "num_threads": 8}') == 1, "loader config not inserted exactly once"
new_lines = re.findall(r"^      - '.*'$", s, flags=re.M)
assert len(new_lines) == len(flags) + 2, ("unexpected command list after patch", len(new_lines))
open(dst, "w").write(s)
print("written", dst)
print("command lines: %d -> %d" % (len(cmd_lines), len(new_lines)))
