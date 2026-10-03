"""add_arms_1004.py : the arms of the 1004 wheel validation, added to arm_compose_1003.py (atomic replace, idempotent).
  F1A  production flags (arm F1), SX_OPT_KV_STEADY_BUDGET unset = the new default (the engine switches the budget on by itself)
  F1Z  the same with SX_OPT_KV_STEADY_BUDGET=0 (same image, same session: the control)
  FMA  native MTP k=4, util 0.87 (arm FM), switch unset            FMZ  the same with SX_OPT_KV_STEADY_BUDGET=0
  FMB  native MTP k=4, util 0.93, switch unset, no SX_OPT_KV_STEADY_RESERVE_MIB (arm FM2 without its explicit reserve)
  F1B  production flags at util 0.94, switch unset (above the util where the physical bound limits the no-MTP KV cache)
All use the validation cache directory of the 1004 wheel."""
import os

p = "/mnt/2t/build/cmp1003/arm_compose_1003.py"
s = open(p).read()
if '"F1A"' not in s:
    anchor = "a = ARMS[arm]\n"
    assert s.count(anchor) == 1
    add = (
        'C1004 = "/opt/shixiang-inference/cache-forkwheel-1004-flashnext-tp4"\n'
        'ARMS["F1A"] = dict(ARMS["F1"], cache=C1004)\n'
        'ARMS["F1Z"] = dict(ARMS["F1"], cache=C1004, env={**FORK_ENV, "SX_OPT_KV_STEADY_BUDGET": "0"})\n'
        'ARMS["FMA"] = dict(ARMS["FM"], cache=C1004)\n'
        'ARMS["FMZ"] = dict(ARMS["FM"], cache=C1004, env={**FORK_MTP_ENV, "SX_OPT_KV_STEADY_BUDGET": "0"})\n'
        'ARMS["FMB"] = dict(ARMS["FM"], cache=C1004, util="0.93")\n'
        'ARMS["F1B"] = dict(ARMS["F1"], cache=C1004, util="0.94")\n'
    )
    s = s.replace(anchor, add + anchor)
    open(p + ".tmp", "w").write(s)
    os.replace(p + ".tmp", p)
    print("arms F1A F1Z FMA FMZ FMB F1B added")
else:
    print("already there")
