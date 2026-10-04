#!/usr/bin/env python3
"""compare_answers.py : check that the needle/short answers of the benchmarked arms are textually identical.

Usage: python3 compare_answers.py [this_dir] [--refs=<group-5 result dir>]

Reads the `*_answers.json` files written by kvq_check.py for
  F1C32   fork 1004 with --max-num-seqs 32 (README group 13)            [this directory]
  O2P     official main@e53d02171, PLE auto channel disabled (group 12) [this directory]
  F1      fork 1004, published config (README group 5)                  [group-5 result directory]
  V2b     official v1.5.1 (README group 5)                              [group-5 result directory]
and prints the needle codes, the short answers, and which arms are identical.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
DIR = _args[0] if _args else HERE
REFS = (next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--refs=")), None)
        or os.path.join(os.path.dirname(os.path.abspath(DIR)), "2026-10-03-fork-vs-official-v1.5.1"))
NAMES = ["F1C32", "O2P", "F1", "V2b"]


def load(arm):
    name = f"{arm}_answers.json"
    for base in (DIR, REFS):
        path = os.path.join(base, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return None


loaded = {}
for arm in NAMES:
    value = load(arm)
    if value is None:
        print(f"{arm}: MISSING")
        continue
    loaded[arm] = value
    for item in value.get("needle", []):
        print(f"{arm}: needle words={item.get('words')} 正确 {item.get('correct')}/{item.get('of')} "
              f"耗时 {item.get('seconds')}s")
    for item in value.get("short", []):
        print(f"{arm}: short ok={item.get('ok')} answer={item.get('answer')!r}")

print()
print("== 逐字段比对（needle 的答案文本 + short 的答案，忽略耗时）==")
for i, a in enumerate(loaded):
    for b in list(loaded)[i + 1:]:
        ta, tb = loaded[a], loaded[b]
        na = [(x.get("words"), x.get("correct"), x.get("of"), x.get("answer")) for x in ta.get("needle", [])]
        nb = [(x.get("words"), x.get("correct"), x.get("of"), x.get("answer")) for x in tb.get("needle", [])]
        sa = [x.get("answer") for x in ta.get("short", [])]
        sb = [x.get("answer") for x in tb.get("short", [])]
        lines = [f"{a} vs {b}:",
                 f"  needle {'逐字相同' if na == nb else '不同'}"
                 f"（{a} {len(na)} 段 / {b} {len(nb)} 段，各段 words="
                 f"{[x[0] for x in na]}）",
                 f"  short {'逐字相同' if sa == sb else '有差异'}"
                 f"（{a} 对 {sum(1 for x in ta.get('short', []) if x.get('ok'))}/{len(sa)}，"
                 f"{b} 对 {sum(1 for x in tb.get('short', []) if x.get('ok'))}/{len(sb)}）"]
        for k, (x, y) in enumerate(zip(sa, sb)):
            if x != y:
                lines.append(f"    第 {k + 1} 题不同：{a}={x!r}  {b}={y!r}")
        print("\n".join(lines))
