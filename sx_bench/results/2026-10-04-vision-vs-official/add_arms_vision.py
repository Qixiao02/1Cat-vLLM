"""add_arms_vision.py : arms of the vision-on comparison "fork 1004 vs official v1.5.1, both with the vision tower loaded".
The same vision flags on both sides: --language-model-only is dropped, --limit-mm-per-prompt={"image":1,"video":0} and
--mm-processor-cache-gb=2 are added; everything else is the arm it is derived from (arm_compose_1003.py).
  fork (image from FORK_IMAGE)   FV1 = F1 (prefix on, util 0.90, 24 seqs)   FV2 = F2 (prefix off)   FVM = FM (MTP k=4, util 0.87)
  official v1.5.1 (OFFICIAL_IMAGE) VV1 = V1 (prefix off, util 0.94, 16 seqs)  VV2b = V2b (prefix on, util 0.90)  VVMa = VMa (MTP, util 0.92, prefill 4096)
Atomic replace, idempotent."""
import os

p = "/mnt/2t/build/cmp1003/arm_compose_1003.py"
s = open(p).read()
if '"FV1"' not in s:
    anchor = "a = ARMS[arm]\n"
    assert s.count(anchor) == 1
    add = (
        'VIS = ["--mm-processor-cache-gb=2", "--limit-mm-per-prompt={\\"image\\":1,\\"video\\":0}"]\n'
        'def _vis(base, **kw):\n'
        '    return dict(base, extra=list(base["extra"]) + VIS, replace={**base.get("replace", {}), "--language-model-only": None}, **kw)\n'
        'CV = "/opt/shixiang-inference/cache-forkwheel-1004-flashnext-tp4"\n'
        'ARMS["FV1"] = _vis(ARMS["F1"], cache=CV)\n'
        'ARMS["FV2"] = _vis(ARMS["F2"], cache=CV)\n'
        'ARMS["FVM"] = _vis(ARMS["FM"], cache=CV)\n'
        'ARMS["VV1"] = _vis(ARMS["V1"])\n'
        'ARMS["VV2b"] = _vis(ARMS["V2b"])\n'
        'ARMS["VVMa"] = _vis(ARMS["VMa"])\n'
    )
    s = s.replace(anchor, add + anchor)
    open(p + ".tmp", "w").write(s)
    os.replace(p + ".tmp", p)
    print("arms FV1 FV2 FVM VV1 VV2b VVMa added")
else:
    print("already there")
