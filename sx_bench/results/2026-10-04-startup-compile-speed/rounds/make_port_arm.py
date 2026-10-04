"""make_port_arm.py : create the comparison arms of ONE port iteration on the bench host.

Every port iteration is: patch -> wheel (or Python-only overlay) -> image -> measure. This script covers the
last step. It never overwrites the earlier generators: it writes arm_compose_<TAG>.py and run_<TAG>.sh next to
arm_compose_1003.py / run_all_1003.sh.

Environment (all optional except TAG):
  TAG      file/image suffix, e.g. 1004port1   -> files arm_compose_1004port1.py, run_1004port1.sh
                                                 image shixiang/1cat-vllm-v100:heavily-modified-v1-1004port1-sm70main
  IMG      full image name (default: derived from TAG as above)
  CACHE    cache dir on the host (default /opt/shixiang-inference/cache-forkwheel-<TAG>-flashnext-tp4)
  EXTRA    python dict literal of extra env vars applied to every arm, e.g. {"VLLM_SM70_NVFP4_QPN2": "1"}
  ARMS     comma list of arm kinds to create: std (PA, no MTP, seqs 32 + MoE cap 320), mtp (PM, MTP k=4 util 0.87)
           default "std,mtp"
  CONCS    concurrency list for the std arm sweep (default "1 4 8 16 24 32")
  LONG     1 (default) keeps the 4-length long run and the 128K run in the std arm, 0 drops them

Printed at the end: the arm lines, the sweep line, and the run order, so the caller can assert before running.
"""
import os

BASE = "/mnt/2t/build/cmp1003"
TAG = os.environ["TAG"]
IMG = os.environ.get("IMG", f"shixiang/1cat-vllm-v100:heavily-modified-v1-{TAG}-sm70main")
CACHE = os.environ.get("CACHE", f"/opt/shixiang-inference/cache-forkwheel-{TAG}-flashnext-tp4")
EXTRA = os.environ.get("EXTRA", "{}")
KINDS = [k.strip() for k in os.environ.get("ARMS", "std,mtp").split(",") if k.strip()]
CONCS = os.environ.get("CONCS", "1 4 8 16 24 32")
LONG = os.environ.get("LONG", "1") == "1"

gen_path = f"{BASE}/arm_compose_1003.py"
drv_path = f"{BASE}/run_all_1003.sh"
gen = open(gen_path).read()
drv = open(drv_path).read()
assert "FORK_ENV" in gen and "FORK_MTP_ENV" in gen, "arm_compose_1003.py layout changed"

kinds = set()
for k in KINDS:
    assert k in ("std", "mtp"), f"unknown arm kind {k}"
    kinds.add(k)

anchor = 'ARMS["V1"] = dict('
assert gen.count(anchor) == 1, "cannot find the insertion point in arm_compose_1003.py"
out_gen = gen
arm_lines = []
if "std" in kinds:
    out_gen = out_gen.replace(anchor, (
        f'# port iteration {TAG}: production no-MTP flags, scheduler capacity 32 and the matching MoE cap (same\n'
        f'# shape as arm F1C32, so the delta is the port and nothing else). image/cache are the ported artifacts.\n'
        f'_PORT_ENV = {{**FORK_ENV, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "320", **{EXTRA}}}\n'
        f'ARMS["PA"] = dict(ARMS["F1"], image="{IMG}", cache="{CACHE}", seqs=32, env=_PORT_ENV)\n'
    ) + anchor, 1)
    arm_lines.append(f'ARMS["PA"] = image {IMG}, seqs 32, MoE cap 320, extra {EXTRA}')
if "mtp" in kinds:
    out_gen = out_gen.replace(anchor, (
        f'# port iteration {TAG}: native MTP k=4 at util 0.87, the arm where the decode batch is up to 5x the\n'
        f'# request count (M = seqs x (1+k)) and therefore the one a wide-M small-matrix port is expected to move.\n'
        f'_PORT_MTP_ENV = {{**FORK_MTP_ENV, **{EXTRA}}}\n'
        f'ARMS["PM"] = dict(ARMS["FM"], image="{IMG}", cache="{CACHE}", env=_PORT_MTP_ENV)\n'
    ) + anchor, 1)
    arm_lines.append(f'ARMS["PM"] = image {IMG}, MTP k=4, util 0.87, extra {EXTRA}')
assert out_gen.count(anchor) == 1
gen_out = f"{BASE}/arm_compose_{TAG}.py"
open(gen_out, "w").write(out_gen)

# ---- driver: helpers + one arm per kind ----
cut = drv.index("arm F1 std")
head = drv[:cut]
head = head.replace("arm_compose_1003.py", f"arm_compose_{TAG}.py")
assert head.count("-gt 3600") == 1, "expected exactly one 3600 s readiness timeout in arm()"
head = head.replace("-gt 3600", "-gt 1800")
assert head.count("for c in 1 2 4 8 16 24; do") == 1, "sweep line changed"
head = head.replace("for c in 1 2 4 8 16 24; do", f"for c in {CONCS}; do")
if not LONG:
    for line in ("    bench $a long ", "    bench $a l128k "):
        keep = [l for l in head.splitlines(keepends=True) if not l.startswith(line)]
        assert len(keep) == len(head.splitlines(keepends=True)) - 1, line
        head = "".join(keep)

order = []
if "std" in kinds:
    order.append("arm PA std")
if "mtp" in kinds:
    order.append("arm PM mtp")
body = head + "\n".join(order) + "\n" + (
    'echo "=== port facts (engine_PA/PM logs)"\n'
    'for a in PA PM; do [ -f engine_$a.log ] || continue; echo "--- $a"; '
    'grep -oE "capture_sizes=[^ ]*|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|'
    'Model loading took [0-9.]+ GiB and [0-9.]+ s|Graph capturing finished in [0-9]+ secs" engine_$a.log | sort -u | head -8; done\n'
    f'echo "RUN_PORT_{TAG}_DONE $(date +%T)"\n'
)
drv_out = f"{BASE}/run_{TAG}.sh"
open(drv_out, "w").write(body)

print(f"written {gen_out} and {drv_out}")
for l in arm_lines:
    print("arm:", l)
print("sweep line:", [l.strip() for l in body.splitlines() if "for c in" in l])
print("run order:", order)
print("cache:", CACHE)
print("last lines:", body.splitlines()[-3:])
