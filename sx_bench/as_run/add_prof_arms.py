"""add_prof_arms.py : arms for the bounded decode profiles (3-second, decode-only windows; never a whole request).
  PROF_E  production flags (arm F1) + --enforce-eager + torch profiler with record_shapes: kernels can be tied to the op and the GEMM shape
  PROF_G  production flags (arm F1) + torch profiler, CUDA graphs on: real timings of the decode step at real context lengths
Both use the validation cache directory of the 1003 image; /cache/prof receives the traces. Atomic replace."""
import os

p = "/mnt/2t/build/cmp1003/arm_compose_1003.py"
s = open(p).read()
if '"PROF_E"' not in s:
    cfg = ("--profiler-config={\\\"profiler\\\":\\\"torch\\\",\\\"torch_profiler_dir\\\":\\\"/cache/prof\\\",\\\"torch_profiler_with_stack\\\":false,"
           "\\\"torch_profiler_record_shapes\\\":%s}")
    anchor = "a = ARMS[arm]\n"
    assert s.count(anchor) == 1
    add = (
        'PROFCACHE = "/opt/shixiang-inference/cache-forkwheel-1003-flashnext-tp4"\n'
        'ARMS["PROF_E"] = dict(ARMS["F1"], cache=PROFCACHE, extra=["--enforce-eager", "' + cfg % "true" + '"])\n'
        'ARMS["PROF_G"] = dict(ARMS["F1"], cache=PROFCACHE, extra=["' + cfg % "false" + '"])\n'
    )
    s = s.replace(anchor, add + anchor)
    open(p + ".tmp", "w").write(s)
    os.replace(p + ".tmp", p)
    print("PROF_E and PROF_G added")
else:
    print("already there")
