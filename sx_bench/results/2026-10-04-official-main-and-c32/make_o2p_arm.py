"""make_o2p_arm.py : create the official-main retest arm O2P on the bench host.

O2 (compose.O2.yaml) = official main@e53d02171, prefix caching on, max-num-seqs 16, util 0.94 -- the
arm that hung after graph capture in warmup_kernels, at the sampler's first host fence
(topk_topp_triton.py:1109), with all four ranks reporting an NCCL all-gather that never completed.
Diagnosis: official main auto-enables hybrid PLE (vllm/config/vllm.py:2140-2152) because OFFICIAL_ENV
in arm_compose_1003.py does not carry the three VLLM_*PLE* variables that FORK_ENV sets to 0.

v1 of this arm (= O2 + those three variables at 0, no --mamba-cache-mode=align, ready timeout 420 s)
never reached the hang: it died at KV allocation, engine_O2P.log:303
    Available KV cache memory: -0.82 GiB
    ValueError: No available memory for the cache blocks.
Reason (vllm/models/qwen4_exp/nvidia/ple_layer.py:739-790 + common/ple.py:340-364,448-462): with hybrid
PLE off, the disk cascade is still active, and the cascade branch measures the device and fills it
first -- plan_ple_placement(vram_budget=table-spill) put 4.69 GiB of the 11.92 GiB table on the device
(engine_O2P.log:249-252) and left only the rest on the host. The spill estimate subtracts
ple_vram_reserve_bytes(), whose default is min(8% of total, 4 GiB) = 2.54 GiB on a 32 GB V100; that
margin was tuned on 48 GB cards and is smaller than this engine's graph pool + activation peak
(1.24 GiB graph reserve alone, gpu_worker.py:592). v1 also overrode the arms' usual
VLLM_ENGINE_READY_TIMEOUT_S=1800 with 420.
Also note: dropping --mamba-cache-mode=align changed nothing -- the image applies it as a serving
default anyway (engine_O2P.log:20 "Applied 1Cat SM70 serving defaults: mamba_cache_mode=align").

v2 = v1 + VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB=10 (device room = 29.83 - 20.91 - 1.69 - 10 < 0, so the
whole table goes to the pinned host tier and the KV budget returns to the 3.88 GiB / ~307k tokens the
official v1.5.1 arms had) + the arms' usual ready timeout of 1800 s. This still tests the v1
hypothesis: with the hybrid lane off, does the first decode forward after graph capture finish?

This script writes two files next to the originals (nothing is overwritten):
  arm_compose_1004.py : arm_compose_1003.py + ARMS["O2P"]
  run_o2p.sh          : run_all_1003.sh truncated after the helpers, running only `arm O2P std`
"""
BASE = "/mnt/2t/build/cmp1003"
gen = open(f"{BASE}/arm_compose_1003.py").read()

anchor = 'ARMS["V1"] = dict('
assert gen.count(anchor) == 1, "cannot find the insertion point in arm_compose_1003.py"
assert 'ARMS["O2P"]' not in gen
o2p = '''# 2026-10-04 retest (v2): O2 with the startup differences the fork arms already have and that the
# official arms do not. (1) the three PLE variables: without them official main auto-enables hybrid
# PLE / PLE host-offload (vllm/config/vllm.py:2140-2152) and never finishes the first decode forward
# after graph capture (four-rank NCCL watchdog, 600 s). (2) VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB=10:
# with hybrid off the disk cascade measures the device and fills it first, and the default reserve
# (min(8% of total, 4 GiB) = 2.54 GiB here) is smaller than this engine's graph pool + activation
# peak, so v1 ended at "Available KV cache memory: -0.82 GiB". Reserving 10 GiB keeps the whole
# table in the pinned host tier and restores the ~3.88 GiB / ~307k token KV budget of the official
# v1.5.1 arms. (3) the arms' usual prepare timeout of 1800 s.
_O2P_ENV = dict(OFFICIAL_ENV)
_O2P_ENV.update({"VLLM_SM70_QWEN38_HYBRID_PLE": "0", "VLLM_PLE_CPU_OFFLOAD": "0",
                 "VLLM_PLE_DISK_OFFLOAD": "0", "VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB": "10",
                 "VLLM_ENGINE_READY_TIMEOUT_S": "1800"})
ARMS["O2P"] = dict(image=OFFICIAL, prefix=True, seqs=16, util="0.94", maxlen=131072, mtp=False,
                   batched=8192, env=_O2P_ENV, extra=[])
'''
gen1004 = gen.replace(anchor, o2p + anchor, 1)
open(f"{BASE}/arm_compose_1004.py", "w").write(gen1004)

drv = open(f"{BASE}/run_all_1003.sh").read()
cut = drv.index("arm F1 std")
head = drv[:cut]
assert "arm_compose_1003.py" in head
head = head.replace("arm_compose_1003.py", "arm_compose_1004.py")
# a hanging arm keeps its container running; 1800 s still covers a healthy official-main start
# (v1.5.1 needed ~15 min from container start to /health) without waiting the full hour.
assert head.count("-gt 3600") == 1, "expected exactly one 3600 s readiness timeout in arm()"
head = head.replace("-gt 3600", "-gt 1800")
body = head + 'arm O2P std\necho "RUN_O2P_DONE $(date +%T)"\n'
open(f"{BASE}/run_o2p.sh", "w").write(body)

print("written arm_compose_1004.py and run_o2p.sh")
print("compose generator line in run_o2p.sh:", [l for l in body.splitlines() if "arm_compose" in l])
print("readiness wait line:", [l for l in body.splitlines() if "-gt 1800" in l])
print("last three lines:", body.splitlines()[-3:])
