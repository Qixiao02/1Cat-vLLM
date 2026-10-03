#!/usr/bin/env python3
"""analyze_eager.py <trace.json[.gz]> [label] [topn] : which op / shape each GPU kernel of an EAGER (record_shapes) decode window belongs to.

Kernel -> parent CPU op through args["External id"]; the parent's "Input Dims" give the GEMM shape. Steps are counted
from the QSA sparse-attention kernel (12 QSA layers => calls / 12). For every (parent op, dims, kernel family) prints
calls per step, mean us per call, ms per step, and for GEMM-like parents the achieved weight read rate
(largest input tensor in bytes / kernel time; fp16 assumed -> a LOWER BOUND of the real traffic only for dense
weights, not meaningful for non-GEMM ops)."""
import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 40
data = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
ev = [e for e in data.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
ops = {}
for e in ev:
    if e.get("cat") == "cpu_op":
        a = e.get("args", {})
        eid = a.get("External id")
        if eid is not None:
            ops[eid] = (e["name"], a.get("Input Dims"), a.get("Input type"))
kern = [e for e in ev if e.get("cat") == "kernel"]
qsa = sum(1 for e in kern if "_qsa_sparse_paged_gqa_splitk_kernel" in e["name"])
steps = max(1, round(qsa / 12))
busy = sum(e["dur"] for e in kern)
print("== %s : %d kernels, %d decode steps (QSA calls %d / 12), kernel time %.1f ms = %.2f ms per step" % (label, len(kern), steps, qsa, busy / 1000, busy / 1000 / steps))


def short(n):
    n = re.sub(r"\(.*", "", n)
    n = re.sub(r"void |\(anonymous namespace\)::|cutlass::Kernel2<|vllm::", "", n)
    return n[:70]


def numel(d):
    """element count of one "Input Dims" entry; a TensorList argument shows up as a list of dim lists -> its largest tensor"""
    if not isinstance(d, (list, tuple)):
        return int(d) if isinstance(d, int) else 0
    if any(isinstance(x, (list, tuple)) for x in d):
        return max((numel(x) for x in d), default=0)
    p = 1
    for x in d:
        p *= int(x)
    return p


groups = collections.defaultdict(lambda: [0, 0.0, 0.0])
for e in kern:
    eid = e.get("args", {}).get("External id")
    op = ops.get(eid)
    if op:
        name, dims, types = op
        dstr = str([d for d in (dims or []) if d][:3])
        wbytes = max((numel(d) for d in (dims or []) if d), default=0) * 2
    else:
        name, dstr, wbytes = "(no cpu op)", "", 0
    g = groups[(name, dstr, short(e["name"]))]
    g[0] += 1
    g[1] += e["dur"]
    g[2] = wbytes
print("%-34s %-34s %-44s %8s %8s %9s %9s" % ("parent op", "input dims", "kernel", "calls/st", "us/call", "ms/step", "GB/s"))
for (name, dstr, kn), (n, dur, wb) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:topn]:
    us = dur / n
    gbs = (wb / (us * 1e-6) / 1e9) if wb and name.startswith("aten::") else 0
    print("%-34s %-34s %-44s %8.1f %8.1f %9.3f %9s" % (name[:34], dstr[:34], kn[:44], n / steps, us, dur / 1000 / steps, ("%.0f" % gbs) if gbs else ""))

print()
fam = collections.defaultdict(float)
for (name, dstr, kn), (n, dur, wb) in groups.items():
    fam[kn] += dur
tot = sum(fam.values()) or 1
print("-- per kernel family (top 14, ms per step, share of kernel time)")
for kn, dur in sorted(fam.items(), key=lambda kv: -kv[1])[:14]:
    print("%-72s %9.3f %6.1f%%" % (kn, dur / 1000 / steps, 100 * dur / tot))
