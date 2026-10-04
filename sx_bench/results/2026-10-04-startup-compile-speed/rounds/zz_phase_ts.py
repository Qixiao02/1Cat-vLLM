#!/usr/bin/env python3
"""zz_phase_ts.py : wall-clock phase timeline from engine_PA.log timestamps (read-only).

Engine log lines carry `MM-DD HH:MM:SS`; unlike the 15 s health poll this gives second-precision
phase boundaries, so the compile-cache A/B can be judged on the compile+capture window instead of the
quantized health time.
"""
import os
import re
from datetime import datetime

W = "/mnt/2t/build/cmp1003"
TAGS = ["1004ctl", "1004su", "1004ctl2", "1004su2", "1004su3", "1004ctl3"]
PAT = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[([^\]]+)\] (.*)")


def sec(stamp):
    return datetime.strptime("2026-" + stamp, "%Y-%m-%d %H:%M:%S")


for tag in TAGS:
    path = os.path.join(W, "startup_ab", tag, "engine_PA.log")
    if not os.path.exists(path):
        print("%-10s missing" % tag)
        continue
    first = None
    loadw = []      # (stamp, seconds)
    modell = []
    comp = []
    cap = None
    kv = None
    for line in open(path, errors="ignore"):
        m = PAT.search(line)
        if not m:
            continue
        stamp, _src, msg = m.groups()
        if first is None:
            first = stamp
        if "Loading weights took" in msg:
            v = re.search(r"took ([0-9.]+) seconds", msg)
            loadw.append((stamp, float(v.group(1)) if v else -1))
        elif "Model loading took" in msg:
            v = re.search(r"and ([0-9.]+) seconds", msg)
            modell.append((stamp, float(v.group(1)) if v else -1))
        elif "torch.compile took" in msg:
            v = re.search(r"took ([0-9.]+) s", msg)
            comp.append((stamp, float(v.group(1)) if v else -1))
        elif "Graph capturing finished" in msg:
            v = re.search(r"finished in (\d+) secs", msg)
            cap = (stamp, float(v.group(1)) if v else -1)
        elif "GPU KV cache size" in msg and kv is None:
            kv = (stamp, msg.strip()[-40:])

    ml_end = max(s for s, _ in modell) if modell else None
    out = ["%-10s" % tag, "first=%s" % first]
    if loadw:
        out.append("loadw=%.1fs@%s" % (max(v for _, v in loadw), max(s for s, _ in loadw)))
    if ml_end:
        out.append("modelload_end=%s(%.1fs)" % (ml_end, max(v for _, v in modell)))
    if comp:
        out.append("compile=%s" % ",".join("%s/%.1fs" % (s, v) for s, v in comp))
    else:
        out.append("compile=none")
    if cap:
        span = (sec(cap[0]) - sec(ml_end)).total_seconds() if ml_end else -1
        out.append("cap_end=%s(%.0fs) span_from_load=%.0fs" % (cap[0], cap[1], span))
    if kv:
        out.append("kv@%s" % kv[0])
    print(" | ".join(out))
