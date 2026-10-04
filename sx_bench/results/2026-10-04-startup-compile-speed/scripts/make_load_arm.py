#!/usr/bin/env python3
"""Generate a port arm whose only difference is the weight-loading path.

MODE:
  ctl -> no extra flags (control)
  mt  -> --model-loader-extra-config {"enable_multithread_load": true, "num_threads": 8}
  pf  -> --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8

Everything else comes from make_port_arm.py (TAG/IMG/CACHE/EXTRA/CONCS/LONG).
"""
import os
import subprocess
import sys

BASE = "/mnt/2t/build/cmp1003"

FLAG_SETS = {
    "ctl": [],
    "mt": [
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 8}',
    ],
    "pf": [
        "--safetensors-load-strategy=prefetch",
        "--safetensors-prefetch-num-threads=8",
    ],
}

mode = os.environ.get("MODE", "ctl")
if mode not in FLAG_SETS:
    sys.exit("unknown MODE %r (use %s)" % (mode, "/".join(sorted(FLAG_SETS))))
flags = FLAG_SETS[mode]

tag = os.environ["TAG"]
env = dict(os.environ)
env.update(
    TAG=tag,
    IMG=os.environ["IMG"],
    CACHE=os.environ["CACHE"],
    EXTRA=os.environ.get("EXTRA", "{}"),
    CONCS=os.environ.get("CONCS", "1"),
    LONG=os.environ.get("LONG", "0"),
)
r = subprocess.run(
    ["python3", "%s/make_port_arm.py" % BASE], env=env, capture_output=True, text=True
)
if r.returncode != 0:
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    sys.exit("make_port_arm.py failed for %s" % tag)

path = "%s/arm_compose_%s.py" % (BASE, tag)
src = open(path).read()
marker = 'ARMS["PA"] = dict(ARMS["F1"]'
assert src.count(marker) == 1, "expected exactly one PA definition in %s" % path
# The generated file is a script: it reads ARMS[arm] further down, so the patch has to be
# inserted right after the PA definition (appending at EOF would run after the compose is written).
i = src.index(marker)
j = src.index("\n", i) + 1
src = src[:j] + 'ARMS["PA"]["extra"] = %r\n' % (flags,) + src[j:]
open(path, "w").write(src)
print("load arm mode=%s tag=%s flags=%s" % (mode, tag, flags))
