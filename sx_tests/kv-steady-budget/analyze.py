# SPDX-License-Identifier: Apache-2.0
"""Digest of one KV-steady-budget trial and the table over all trials.
Standard library only.

    analyze.py run   --samples S --events E --log L [--stress J] --util 0.93 \
                     --switch 1 --lane MTP --out result.json \
                     [--peak-limit-mib 32200] [--headroom-mib 500]
    analyze.py table result_*.json

``samples`` has one line per GPU per sample, ``epoch, gpu, used_mib, total_mib``
(``nvidia-smi --query-gpu=index,memory.used,memory.total``, every 2 s);
``events`` has ``epoch name`` lines (compose_up, ready, stress_begin,
stress_end); ``log`` is the engine's ``docker logs`` output.

The verdict is FAIL when any GPU's highest ``memory.used`` exceeds the peak
limit (default 32200 MiB), when less than ``--headroom-mib`` of CUDA-usable
memory would be left (the usable total is read from the engine's own log line,
because nvidia-smi's total includes memory CUDA never hands out), when a
request failed, when the engine log has an out-of-memory error or a traceback,
or when the engine's own post-warm-up audit said the reservation was SHORT.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

MiB = 1 << 20
GiB = 1 << 30

RANK_PREFIX = re.compile(r"^\([A-Za-z_0-9]+ pid=\d+\)\s*")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
KV_AVAILABLE = re.compile(r"Available KV cache memory:\s*(-?[\d.]+)\s*GiB")
MAX_CONC = re.compile(
    r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x"
)
GRAPH = re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB")
PLAN_KV = re.compile(r"KV steady budget \[(?P<kv>[^\]]*)\]")
AUDIT_MEASURED = re.compile(
    r"measured post-sizing growth (-?[\d.]+) MiB \(plan assumed (-?[\d.]+) MiB\)"
)
AUDIT_SHORT = re.compile(r"KV steady audit: SHORT by (-?[\d.]+) MiB")
AUDIT_OK = re.compile(r"KV steady audit: OK")
AUDIT_SUGGEST = re.compile(r"SX_OPT_KV_STEADY_RESERVE_MIB=(\d+)")
# Out-of-memory and crash signatures (any case), and vLLM's own error level: the
# log line starts with ERROR once the rank prefix is stripped.
OOM = re.compile(
    r"out of memory|OutOfMemoryError|CUDA error|Traceback \(most recent|"
    r"Engine core initialization failed",
    re.IGNORECASE,
)
ERROR_LEVEL = re.compile(r"^ERROR\b")
PHASE_END = re.compile(r"KV steady phase end_of_warmup: (\d+) MiB free")


def _clean(line: str) -> str:
    return RANK_PREFIX.sub("", ANSI.sub("", line.strip()))


def parse_samples(path: str) -> list[tuple[float, int, int, int]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 4:
                continue
            try:
                rows.append(
                    (float(parts[0]), int(parts[1]), int(float(parts[2])), int(float(parts[3])))
                )
            except ValueError:
                continue
    return rows


def parse_events(path: str) -> dict[str, float]:
    events: dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) >= 2:
                try:
                    events[parts[1]] = float(parts[0])
                except ValueError:
                    continue
    return events


def memory_phases(
    samples: list[tuple[float, int, int, int]], events: dict[str, float]
) -> dict:
    """Highest memory.used per phase over all GPUs (MiB) and per GPU."""
    t_up = events.get("compose_up", 0.0)
    t_ready = events.get("ready")
    t_begin = events.get("stress_begin")
    t_end = events.get("stress_end")

    def window(lo: float | None, hi: float | None) -> list[tuple[float, int, int, int]]:
        return [
            s
            for s in samples
            if (lo is None or s[0] >= lo) and (hi is None or s[0] <= hi)
        ]

    def peak(rows: list[tuple[float, int, int, int]]) -> int | None:
        return max((r[2] for r in rows), default=None)

    start_rows = window(t_up, t_ready)
    idle_rows: list[tuple[float, int, int, int]] = []
    if t_ready is not None:
        idle_rows = window(t_ready, t_begin)
    stress_rows: list[tuple[float, int, int, int]] = []
    if t_begin is not None:
        stress_rows = window(t_begin, (t_end + 4.0) if t_end is not None else None)
    per_gpu: dict[int, dict[str, int]] = defaultdict(dict)
    for name, rows in (("start", start_rows), ("idle", idle_rows), ("stress", stress_rows)):
        by_gpu: dict[int, int] = {}
        for _, gpu, used, _ in rows:
            by_gpu[gpu] = max(by_gpu.get(gpu, 0), used)
        for gpu, used in by_gpu.items():
            per_gpu[gpu][name] = used
    totals = {s[3] for s in samples}
    return {
        "start_peak_mib": peak(start_rows),
        "idle_mib": peak(idle_rows),
        "stress_peak_mib": peak(stress_rows),
        "peak_mib": peak(samples if not t_up else window(t_up, None)),
        "smi_total_mib": max(totals) if totals else None,
        "per_gpu": {str(g): v for g, v in sorted(per_gpu.items())},
    }


def parse_log(path: str) -> dict:
    kv_tokens: list[int] = []
    available: list[float] = []
    concurrency: list[float] = []
    graphs: list[tuple[int, float]] = []
    plan: dict[str, str] = {}
    audit_measured: list[tuple[float, float]] = []
    audit_short: list[float] = []
    audit_ok = 0
    audit_suggest: list[int] = []
    end_free: list[int] = []
    steady_lines: list[str] = []
    errors: list[str] = []
    auto_enabled = False
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _clean(raw)
            if not line:
                continue
            if m := KV_TOKENS.search(line):
                kv_tokens.append(int(m.group(1).replace(",", "")))
            if m := KV_AVAILABLE.search(line):
                available.append(float(m.group(1)))
            if m := MAX_CONC.search(line):
                concurrency.append(float(m.group(2)))
            if m := GRAPH.search(line):
                graphs.append((int(m.group(1)), float(m.group(2))))
            if m := PLAN_KV.search(line):
                for item in m.group("kv").split():
                    key, _, value = item.partition("=")
                    plan.setdefault(key, value)
            if m := AUDIT_MEASURED.search(line):
                audit_measured.append((float(m.group(1)), float(m.group(2))))
            if m := AUDIT_SHORT.search(line):
                audit_short.append(float(m.group(1)))
            if AUDIT_OK.search(line):
                audit_ok += 1
            if "KV steady audit" in line and (m := AUDIT_SUGGEST.search(line)):
                audit_suggest.append(int(m.group(1)))
            if "Auto-enabling the KV steady-state budget" in line:
                auto_enabled = True
            if "KV steady" in line and line not in steady_lines:
                steady_lines.append(line)
            if m := PHASE_END.search(line):
                end_free.append(int(m.group(1)))
            if (OOM.search(line) or ERROR_LEVEL.search(line)) and (
                "KV steady audit" not in line
            ):
                errors.append(line[:240])
    return {
        "kv_tokens": min(kv_tokens) if kv_tokens else None,
        "available_kv_gib": available,
        "max_concurrency": min(concurrency) if concurrency else None,
        "graph_capture": [{"secs": s, "gib": g} for s, g in graphs],
        "plan": plan,
        "audit_measured_mib": [m for m, _ in audit_measured],
        "audit_planned_mib": [p for _, p in audit_measured],
        "audit_short_mib": audit_short,
        "audit_ok_ranks": audit_ok,
        "audit_suggested_reserve_mib": audit_suggest,
        "end_of_warmup_free_mib": end_free,
        "steady_lines": steady_lines,
        "auto_enabled": auto_enabled,
        "error_lines": errors,
    }


def verdict(
    result: dict,
    *,
    peak_limit_mib: int,
    headroom_mib: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    mem = result["memory"]
    log = result["log"]
    peak = mem.get("peak_mib")
    if not result.get("ready"):
        reasons.append("engine never became healthy")
    if peak is not None and peak > peak_limit_mib:
        reasons.append(f"peak {peak} MiB > limit {peak_limit_mib} MiB")
    usable = log["plan"].get("total")
    if usable is not None and peak is not None:
        free = int(usable) // MiB - peak
        mem["cuda_free_at_peak_mib"] = free
        if free < headroom_mib:
            reasons.append(
                f"only {free} MiB of CUDA-usable memory free at the peak "
                f"(< {headroom_mib} MiB headroom)"
            )
    end_free = log["end_of_warmup_free_mib"]
    idle = mem.get("idle_mib")
    if usable is not None and end_free and idle is not None:
        # nvidia-smi's memory.used against what CUDA reports used (total - free) at
        # the end of the warm-up: if they differ, the peak limit (nvidia-smi) and the
        # headroom (CUDA) are not in the same unit.
        cuda_used = int(usable) // MiB - min(end_free)
        mem["smi_minus_cuda_idle_mib"] = idle - cuda_used
    stress = result.get("stress") or {}
    if stress.get("failed"):
        reasons.append(f"{stress['failed']} of {stress['total']} stress requests failed")
    if log["error_lines"]:
        reasons.append(f"{len(log['error_lines'])} error line(s) in the engine log")
    if log["audit_short_mib"]:
        reasons.append(
            "engine audit: reservation SHORT by up to %.0f MiB"
            % max(log["audit_short_mib"])
        )
    switch = result.get("switch")
    planned = bool(log["plan"])
    if switch == "auto":
        if not log["auto_enabled"]:
            reasons.append("switch unset but the engine never logged the auto-enable line")
        if not planned:
            reasons.append("switch unset but the engine never planned a steady budget")
    return not reasons, reasons


def cmd_run(args: argparse.Namespace) -> int:
    samples = parse_samples(args.samples)
    events = parse_events(args.events)
    result: dict = {
        "tag": args.tag,
        "util": args.util,
        "switch": args.switch,
        "lane": args.lane,
        "ready": "ready" in events,
        "memory": memory_phases(samples, events),
        "log": parse_log(args.log),
    }
    if args.stress:
        try:
            with open(args.stress, encoding="utf-8") as handle:
                result["stress"] = json.load(handle)
        except (OSError, ValueError):
            result["stress"] = {"failed": 1, "total": 1, "requests": []}
    ok, reasons = verdict(
        result, peak_limit_mib=args.peak_limit_mib, headroom_mib=args.headroom_mib
    )
    result["pass"] = ok
    result["fail_reasons"] = reasons
    result["peak_limit_mib"] = args.peak_limit_mib
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    mem = result["memory"]
    delta = mem.get("smi_minus_cuda_idle_mib")
    if delta is not None and abs(delta) > 128:
        print(
            "note: nvidia-smi memory.used is %+d MiB from CUDA's own used memory at "
            "the end of the warm-up; the peak limit and the headroom check use "
            "different units on this machine" % delta
        )
    print(
        "%s util %s switch %s: KV %s tokens, idle %s MiB, peak %s MiB -> %s%s"
        % (
            args.tag,
            args.util,
            args.switch,
            result["log"]["kv_tokens"],
            mem["idle_mib"],
            mem["peak_mib"],
            "PASS" if ok else "FAIL",
            "" if ok else " (" + "; ".join(reasons) + ")",
        )
    )
    return 0 if ok else 1


def cmd_table(args: argparse.Namespace) -> int:
    rows = []
    for path in args.results:
        with open(path, encoding="utf-8") as handle:
            rows.append(json.load(handle))
    header = (
        "switch | util | KV tokens | Available KV GiB | graph GiB | start peak | "
        "idle | stress peak | peak | free@peak | audit (suggested reserve MiB) | verdict"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        log, mem = r["log"], r["memory"]
        available = log["available_kv_gib"]
        graph = [g["gib"] for g in log["graph_capture"]]
        suggest = log["audit_suggested_reserve_mib"]
        tail = " (%d)" % max(suggest) if suggest else ""
        if log["audit_short_mib"]:
            audit = "SHORT %.0f%s" % (max(log["audit_short_mib"]), tail)
        elif log["audit_ok_ranks"]:
            audit = "ok" + tail
        else:
            audit = "-"
        print(
            "%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s"
            % (
                r["switch"],
                r["util"],
                log["kv_tokens"] if log["kv_tokens"] is not None else "-",
                "%.2f" % min(available) if available else "-",
                "%.2f" % max(graph) if graph else "-",
                mem["start_peak_mib"],
                mem["idle_mib"],
                mem["stress_peak_mib"],
                mem["peak_mib"],
                mem.get("cuda_free_at_peak_mib", "-"),
                audit,
                "PASS" if r["pass"] else "FAIL",
            )
        )
    return 0 if all(r["pass"] for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--samples", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--log", required=True)
    run.add_argument("--stress")
    run.add_argument("--tag", required=True)
    run.add_argument("--util", required=True)
    run.add_argument("--switch", required=True)
    run.add_argument("--lane", default="MTP")
    run.add_argument("--peak-limit-mib", type=int, default=32200)
    run.add_argument("--headroom-mib", type=int, default=500)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    table = sub.add_parser("table")
    table.add_argument("results", nargs="+")
    table.set_defaults(func=cmd_table)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
