"""make_c32_arm.py : create the C32 arm (user-approved item 4) on the bench host.

Why: sx_bench/results/2026-10-04-decode-profile/LEVERS.md ranks "raise the concurrency cap from 24
to 32" first (--max-num-seqs 32 + VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS 240 -> 320). Its +17% total /
-12% per-stream numbers are an EXTRAPOLATION from 8->24 concurrency (+7.7 ms/step, ~0.5 ms per extra
stream, 11 ms fixed part at C1): "没有跑过 32 路". This arm measures C32 instead of extrapolating it,
on the same engine instance that serves C8/C16/C24, so the 24->32 delta is apples to apples with the
sweep_c24 numbers already in the 1004 results.

Arm: F1C32 = ARMS["F1"] (fork image, prefix caching on, util 0.90, maxlen 131072, TP4, no MTP,
batched tokens 8192) with seqs=32 and the MoE tuning cap raised 240 -> 320. Nothing else changes,
so F1 (24 seqs) remains the reference arm of the published 1004 comparison.

Checks that LEVERS.md asks for and this run answers from the engine log:
  * full-graph capture size: vllm/config/vllm.py:540-549 _sm70_nomtp_cudagraph_capture_sizes caps at
    32 (constant at :84 = (1,2,4,8,16,32)) and always adds max_num_seqs, so seqs=32 should capture
    [1,2,4,8,16,32] -- i.e. C32 stays inside validated full-graph concurrency, unlike 25..31.
  * KV headroom: F1 had GPU KV cache size 388,772 tokens at util 0.90; 32 x (8000 prompt + 256 gen)
    needs ~264k tokens, so C32 fits in one wave at 8K without KV pressure. The driver prints the
    "GPU KV cache size" / "Maximum concurrency" lines to prove it.
  * MoE tuning cap: 320 = 32 x (1 + 9) is the same arithmetic the 240 cap used for 24 seqs; no MTP
    here, so 320 is a plain ceiling.

Sweeps: c8, c16, c24, c32 on this one engine (same seed/lengths/passes as every other arm:
--lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101). long/l128k are skipped --
they are concurrency-independent and already measured on F1. kvq_check answers are still run.

This script writes two files next to the originals (nothing is overwritten; the O2P files stay put):
  arm_compose_c32.py : arm_compose_1003.py + ARMS["F1C32"]
  run_c32.sh         : run_all_1003.sh truncated after the helpers, running only `arm F1C32 std`
"""
BASE = "/mnt/2t/build/cmp1003"
gen = open(f"{BASE}/arm_compose_1003.py").read()

anchor = 'ARMS["V1"] = dict('
assert gen.count(anchor) == 1, "cannot find the insertion point in arm_compose_1003.py"
assert 'ARMS["F1C32"]' not in gen
arm = '''# 2026-10-04 C32 experiment (LEVERS.md rank 1, user-approved item 4). F1 with the scheduler
# capacity raised 24 -> 32 and the MoE tuning cap raised 240 -> 320. This is the arm that turns the
# "24 streams 831 token/s -> 32 streams ~970 token/s (+17%)" extrapolation into a measurement.
_C32_ENV = {**FORK_ENV, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "320"}
ARMS["F1C32"] = dict(ARMS["F1"], seqs=32, env=_C32_ENV)
'''
gen_c32 = gen.replace(anchor, arm + anchor, 1)
open(f"{BASE}/arm_compose_c32.py", "w").write(gen_c32)

drv = open(f"{BASE}/run_all_1003.sh").read()
cut = drv.index("arm F1 std")
head = drv[:cut]
assert "arm_compose_1003.py" in head
head = head.replace("arm_compose_1003.py", "arm_compose_c32.py")
assert head.count("-gt 3600") == 1, "expected exactly one 3600 s readiness timeout in arm()"
head = head.replace("-gt 3600", "-gt 1800")
# C32 needs the concurrency curve on one engine instance; long/l128k do not depend on max-num-seqs.
assert head.count("for c in 1 2 4 8 16 24; do") == 1
head = head.replace("for c in 1 2 4 8 16 24; do", "for c in 8 16 24 32; do")
for line in ("    bench $a long ", "    bench $a l128k "):
    keep = [l for l in head.splitlines(keepends=True) if not l.startswith(line)]
    assert len(keep) == len(head.splitlines(keepends=True)) - 1, line
    head = "".join(keep)
body = head + '''arm F1C32 std
echo "=== capture sizes and KV budget (engine_F1C32.log)"
grep -o "capture_sizes=[^ ]*" engine_F1C32.log | head -3
grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished" engine_F1C32.log | head -6
echo "RUN_C32_DONE $(date +%T)"
'''
open(f"{BASE}/run_c32.sh", "w").write(body)

print("written arm_compose_c32.py and run_c32.sh")
print("arm line:", [l for l in gen_c32.splitlines() if "F1C32" in l])
print("sweep line:", [l for l in body.splitlines() if "for c in" in l])
print("remaining bench lines:", [l.strip() for l in body.splitlines() if "bench $a" in l])
print("last three lines:", body.splitlines()[-3:])
