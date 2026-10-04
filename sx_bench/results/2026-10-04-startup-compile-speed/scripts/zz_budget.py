#!/usr/bin/env python3
"""zz_budget.py : per-round startup budget from harness logs + engine log timestamps (read-only).

For every A/B round it prints: container start -> first engine line (pre-engine), weight loading,
post-load (compile + graph capture + KV profiling, overlapped), post-capture -> /health, and the
total health time.  Engine log lines carry MM-DD HH:MM:SS (server local, UTC+8), so the phase
boundaries are second-precision instead of the 15 s health poll.
"""
import os
import re
from datetime import datetime, timedelta, timezone

W = "/mnt/2t/build/cmp1003"
TAGS = ["1004ctl", "1004su", "1004ctl2", "1004su2", "1004su3", "1004ctl3", "1004top1"]
PAT = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[([^\]]+)\] (.*)")


def hms(s):
    """Harness stamp 'HH:MM:SS' in server local time (UTC+8)."""
    return datetime.strptime("2026-10-04 " + s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8)))


def ets(s):
    """Engine-log stamp 'MM-DD HH:MM:SS' in UTC (docker logs are UTC)."""
    return datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def runlog(tag):
    for name in ("run_%s.log" % tag, "startup_ab/%s/run_%s.log" % (tag, tag)):
        p = os.path.join(W, name)
        if os.path.exists(p):
            return p
    return None


print("%-10s %8s %8s %8s %8s %9s %7s %9s" % (
    "round", "pre_eng", "load", "post_ld", "post_cap", "total_h", "compile", "capture"))
for tag in TAGS:
    eng = os.path.join(W, "startup_ab", tag, "engine_PA.log")
    rl = runlog(tag)
    if not os.path.exists(eng) or rl is None:
        print("%-10s missing (engine=%s run=%s)" % (tag, os.path.exists(eng), rl is not None))
        continue
    text = open(rl, errors="ignore").read()
    m_start = re.search(r"=== start PA (\d\d:\d\d:\d\d)", text)
    m_health = re.findall(r"\[PA\] healthy after (\d+)s", text)
    first = None
    loadw = []
    modell = []
    comp = []
    cap = None
    for line in open(eng, errors="ignore"):
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
            comp.append(float(v.group(1)) if v else -1)
        elif "Graph capturing finished" in msg:
            v = re.search(r"finished in (\d+) secs", msg)
            cap = (stamp, float(v.group(1)) if v else -1)

    if not (m_start and m_health and modell):
        print("%-10s incomplete (start=%s health=%s)" % (tag, bool(m_start), m_health[-1:]))
        continue
    start = hms(m_start.group(1))
    health = int(m_health[-1])
    health_at = start.timestamp() + health
    pre = (ets(first).timestamp() - start.timestamp())
    ml_end = max(s for s, _ in modell)
    load = ets(ml_end).timestamp() - ets(first).timestamp()
    post_ld = ets(cap[0]).timestamp() - ets(ml_end).timestamp() if cap else -1
    post_cap = health_at - ets(cap[0]).timestamp() if cap else -1
    print("%-10s %8.0f %8.0f %8.0f %8.0f %9d %7s %9s" % (
        tag, pre, load, post_ld, post_cap, health,
        ("+".join("%.0f" % c for c in comp) if comp else "none"),
        "%.0fs" % cap[1] if cap else "-"))
