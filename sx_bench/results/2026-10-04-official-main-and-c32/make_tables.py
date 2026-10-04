#!/usr/bin/env python3
"""make_tables.py : turn the raw pfx_bench logs of four arms into the markdown tables of RESULTS.md.

Usage:  python3 make_tables.py [log_dir] [out.md]
        (defaults: this script's directory and RESULTS.md next to it)

Arms (all on the same 4x V100-SXM2-32GB box, TP4, GPU 4-7, same model, same prompts/seed,
one instance at a time; 8K input, 256 greedy tokens, 2 passes averaged unless noted):

  F1      fork 1004 (image shixiang/1cat-vllm-v100:heavily-modified-v1-e673bd168-sm70main),
          prefix caching on, util 0.90, max-num-seqs 24, KV 410,988 tok          [published group 5]
  F1C32   same as F1 but --max-num-seqs 32 and VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS=320
  O2P     official main@e53d02171 (image ...:official-main-e53d02171-sm70main),
          prefix caching on, util 0.94, max-num-seqs 16, KV 299,169 tok,
          PLE auto channel disabled (3 PLE env vars = 0 + VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB=10)
  V2b     official v1.5.1 wheel in our own Ubuntu 24.04 image, prefix caching on,
          util 0.90, max-num-seqs 16, KV 202,161 tok                                [published group 5]

Only std (non-MTP) runs are read, and one token per step means aggregate tok/s = steps/s,
so step_ms = 1000 * conc / agg.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
DIR = _args[0] if _args else HERE
OUT = _args[1] if len(_args) > 1 else os.path.join(HERE, "RESULTS.md")
# The two reference arms (F1, V2b) were published with README group 5 and their logs stay in that
# result directory; only the two new arms (F1C32, O2P) live next to this script.
REFS = (next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--refs=")), None)
        or os.path.join(os.path.dirname(os.path.abspath(DIR)), "2026-10-03-fork-vs-official-v1.5.1"))

ARMS = {
    "F1": ("本分支 1004（seqs 24，util 0.90）", 410988, "已发布（README 第 5 组）"),
    "F1C32": ("本分支 1004（seqs 32，util 0.90）", 410988, "本次新测（README 第 13 组）"),
    "O2P": ("官方 main@e53d02171（seqs 16，util 0.94）", 299169, "本次新测（README 第 12 组）"),
    "V2b": ("官方 v1.5.1（seqs 16，util 0.90）", 202161, "已发布（README 第 5 组）"),
}
SWEEP_CONCS = (1, 2, 4, 8, 16, 24, 32)

LINE = re.compile(
    r"pass\s+(?P<pass>\d+)\s+C(?P<conc>\d+)\s+len\s+(?P<len>\d+)\s+prompt\s+(?P<prompt>\d+)\s+\|\s+"
    r"ttft first/last\s+(?P<ttft0>[\d.]+)/\s*(?P<ttft1>[\d.]+)\s*s\s+\|\s+"
    r"prefill\s+(?P<prefill>[\d.]+)\s*tok/s\s+\|\s+"
    r"decode\s+(?P<per>[\d.]+)\s*tok/s per stream,\s+(?P<agg>[\d.]+)\s*agg\s+\(window\s+(?P<win>[-.\d]+)s\)\s+\|\s+"
    r"e2e\s+(?P<e2e>[\d.]+)s\s+\|\s+kv peak\s+(?P<kv>[\d.]+)%\s+\|\s+wait\s+(?P<wait>\d+)\s+\|\s+"
    r"cache hits\s+(?P<hits>\d+)\s+\|\s+preempt\s+(?P<preempt>\d+)"
)
WAVE = re.compile(r"WAVE\s+(?P<live>\d+)/(?P<total>\d+):.*?decode\s+(?P<dec>[\d.]+)")


def parse(path):
    """one dict per pass line, in file order"""
    rows = []
    if not os.path.exists(path):
        return rows
    for raw in open(path, errors="replace"):
        m = LINE.search(raw)
        if not m:
            continue
        d = m.groupdict()
        row = {k: float(d[k]) for k in ("ttft0", "ttft1", "prefill", "per", "agg", "win", "e2e", "kv")}
        for k in ("pass", "conc", "len", "prompt", "wait", "hits", "preempt"):
            row[k] = int(d[k])
        w = WAVE.search(raw)
        row["wave_live"], row["wave_total"], row["wave_decode"] = (
            (int(w.group("live")), int(w.group("total")), float(w.group("dec"))) if w else (None, None, None)
        )
        rows.append(row)
    return rows


def avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def arm_path(name):
    """look in this directory first, then in the published group-5 directory (reference arms)"""
    path = os.path.join(DIR, name)
    return path if os.path.exists(path) else os.path.join(REFS, name)


def load_arm(arm):
    """{conc: [rows], 'long': [...], 'l128k': [...]}"""
    data = {}
    for c in SWEEP_CONCS:
        rows = parse(arm_path(f"{arm}_sweep_c{c}.log"))
        if rows:
            data[c] = rows
    data["long"] = parse(arm_path(f"{arm}_long.log"))
    data["l128k"] = parse(arm_path(f"{arm}_l128k.log"))
    return data


def kv_full(rows):
    """the run is not a usable speed datapoint: decode came out as 0 (measurement window collapsed)
    and/or the KV pool hit the ceiling with a preemption, so not all requests ran at once"""
    if not rows:
        return False
    return (avg(rows, "per") <= 0.01
            or avg(rows, "win") < 0
            or (avg(rows, "kv") > 95.0 and avg(rows, "preempt") >= 1))


def cell(rows):
    """per-stream decode, or a marker when the KV pool was full so decode is not comparable"""
    if not rows:
        return "—"
    if kv_full(rows):
        live = [r["wave_live"] for r in rows if r["wave_live"]]
        tot = [r["wave_total"] for r in rows if r["wave_total"]]
        return (f"**KV 满，decode 不可比**（只有 {int(mean(live))}/{int(mean(tot))} 条同时跑）"
                if live else "**KV 满，decode 不可比**")
    return f"{avg(rows, 'per'):.2f}"


def extras(rows, conc=None):
    """（合计、步时间、KV 峰值、抢占）尾注，KV 满时只报 KV 峰值与抢占"""
    if not rows:
        return ""
    parts = []
    if conc and not kv_full(rows) and avg(rows, "agg") > 0:
        parts.append(f"合计 {avg(rows, 'agg'):.0f}")
        parts.append(f"步 {1000.0 * conc / avg(rows, 'agg'):.1f} ms")
    parts.append(f"KV {avg(rows, 'kv'):.0f}%")
    if avg(rows, "preempt") > 0:
        parts.append(f"抢占 {int(avg(rows, 'preempt'))}")
    return "（" + "，".join(parts) + "）"


def diff(a, b):
    if not a or not b or kv_full(a) or kv_full(b):
        return "—"
    x, y = avg(a, "per"), avg(b, "per")
    return f"{100 * (x / y - 1):+.1f}%"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


data = {a: load_arm(a) for a in ARMS}
L = []
L.append("# 官方 main 对比（第 12 组）与 32 路并发实验（第 13 组）")
L.append("")
L.append("本文件的表格由 `make_tables.py` 从原始 `pfx_bench` 日志算出（读 `<arm>_sweep_c<C>.log`、"
         "`<arm>_long.log`、`<arm>_l128k.log`，不改一个数字）。本次新跑的两个臂（`O2P_*`、`F1C32_*`）"
         "的日志在本目录；两个参照臂（`F1_*`、`V2b_*`）是第 5 组已发布的数据，日志在 "
         "`../2026-10-03-fork-vs-official-v1.5.1/`（脚本会自动去那里找）。")
L.append("")
L.append("目录里同时收了每格的原始 JSON（`O2P_sweep_c*.json`、`O2P_long.json`、`O2P_l128k.json`、"
         "`F1C32_sweep_c*.json`、`*_answers.json`、`*_accel.json`），本文件的每个数字都能在它们里面对到"
         "（`*.log` 只是本地中间产物，仓库的 `.gitignore` 不收 `*.log`，第 5 组目录也是这样）。")
L.append("")
L.append("## 四组配置")
L.append("")
L.append(md_table(
    ["臂", "说明", "KV 容量（token）", "来源"],
    [[a, ARMS[a][0], f"{ARMS[a][1]:,}" if ARMS[a][1] else "见引擎日志", ARMS[a][2]] for a in ARMS],
))
L.append("")
L.append("## 1. 8K 输入、贪心 256 token 的并发扫描（每路 token/s，每格 2 遍平均）")
L.append("")
sweep_rows = []
for c in SWEEP_CONCS:
    row = [f"**{c}**"]
    for a in ARMS:
        rows = data[a].get(c, [])
        row.append(cell(rows) + extras(rows, c) if rows else "—")
    row.append(diff(data["F1"].get(c), data["O2P"].get(c)))
    row.append(diff(data["F1"].get(c), data["V2b"].get(c)))
    sweep_rows.append(row)
L.append(md_table(["并发", *ARMS.keys(), "F1 对 O2P", "F1 对 V2b"], sweep_rows))
if 24 in data["F1C32"] and 24 in data["F1"]:
    L.append("")
    L.append("注：F1C32 的 24 路一格没有对应的捕获尺寸（`--max-num-seqs 32` 把尺寸集合换成 "
             "`(1, 2, 4, 8, 16, 32)`），这一格是 eager 执行的，比 F1 的图形重放慢 4.2%；"
             "原因与影响见第 4 节。")
L.append("")
L.append("## 2. 4 条长 prompt 同时到达（默认采样，400 token，每路 token/s）")
L.append("")
long_rows = []
for want in (8000, 16000, 32000, 64000):
    row = [f"**{want // 1000}K**"]
    for a in ARMS:
        rows = [r for r in data[a]["long"] if abs(r["len"] - want) <= 40]
        row.append(cell(rows) + extras(rows) if rows else "—")
    long_rows.append(row)
L.append(md_table(["每条输入", *ARMS.keys()], long_rows))
L.append("")
L.append("## 3. 单条 128K")
L.append("")
L.append(md_table(
    ["臂", "prefill（token/s）", "首字（s）", "decode（token/s）", "KV 峰值"],
    [[a,
      f"{avg(data[a]['l128k'], 'prefill'):.0f}" if data[a]["l128k"] else "—",
      f"{avg(data[a]['l128k'], 'ttft0'):.2f}" if data[a]["l128k"] else "—",
      f"{avg(data[a]['l128k'], 'per'):.2f}" if data[a]["l128k"] else "—",
      f"{avg(data[a]['l128k'], 'kv'):.0f}%" if data[a]["l128k"] else "—"] for a in ARMS],
))
L.append("")
L.append("## 4. 并发上限 24 → 32（第 13 组）")
L.append("")
L.append("两个臂只差两个启动参数：`--max-num-seqs 32` 和 `VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS=320`"
         "（其余完全相同：同一镜像、前缀缓存开、util 0.90、batched 8192、无 MTP）。第二个参数的来历："
         "`vllm/model_executor/warmup/awq_sm70_warmup.py:892` 里 `tuned_max_tokens = "
         "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS // experts_per_token`（本模型 top-k = 10），"
         "生产值 240 只把 MoE 的调优热身上限推到 24 路，320 才覆盖到 32 路。")
L.append("")
c32 = data["F1C32"]
if 24 in c32 and 32 in c32:
    p24, p32 = avg(c32[24], "per"), avg(c32[32], "per")
    a24, a32 = avg(c32[24], "agg"), avg(c32[32], "agg")
    s24, s32 = 1000.0 * 24 / a24, 1000.0 * 32 / a32
    f1_24 = data["F1"].get(24, [])
    fp, fa = (avg(f1_24, "per"), avg(f1_24, "agg")) if f1_24 else (float("nan"), float("nan"))
    fs = 1000.0 * 24 / fa if fa else float("nan")
    fk = avg(f1_24, "kv") if f1_24 else float("nan")
    L.append(md_table(
        ["指标", "F1 @24（捕获含 24 槽）", "F1C32 @24（捕获无 24 槽）", "F1C32 @32（捕获 32 槽）",
         "同引擎 24→32", "生产对生产 F1@24 → F1C32@32", "LEVERS 外推"],
        [["每路（token/s）", f"{fp:.2f}", f"{p24:.2f}", f"{p32:.2f}",
          f"{100 * (p32 / p24 - 1):+.1f}%", f"{100 * (p32 / fp - 1):+.1f}%", "−12%"],
         ["合计（token/s）", f"{fa:.0f}", f"{a24:.0f}", f"{a32:.0f}",
          f"{100 * (a32 / a24 - 1):+.1f}%", f"{100 * (a32 / fa - 1):+.1f}%", "+17%"],
         ["步时间（ms）", f"{fs:.1f}", f"{s24:.1f}", f"{s32:.1f}",
          f"{s32 - s24:+.1f} ms", f"{s32 - fs:+.1f} ms", "+4.0 ms"],
         ["KV 峰值（%）", f"{fk:.0f}", f"{avg(c32[24], 'kv'):.0f}", f"{avg(c32[32], 'kv'):.0f}", "—", "—", "—"],
         ["抢占次数", f"{int(avg(f1_24, 'preempt')) if f1_24 else '—'}", f"{int(avg(c32[24], 'preempt'))}",
          f"{int(avg(c32[32], 'preempt'))}", "—", "—", "—"]],
    ))
    L.append("")
    L.append("**读这张表要注意捕获尺寸这个干扰项**：`engine_F1.log` 是 "
             "`capture_sizes=(1, 2, 4, 8, 16, 24)`，`engine_F1C32.log` 是 `capture_sizes=(1, 2, 4, 8, 16, 32)`"
             "（`vllm/config/vllm.py:540-549`）。`vllm/v1/worker/gpu/cudagraph_utils.py:566-583` 的 `dispatch()` "
             "按 `num_tokens` 精确查 `self._candidates[num_tokens]`，查不到就返回 `cg_mode=NONE`；"
             "所以 24 路这一波在 F1C32 上没有图可重放（走 eager），"
             f"同一个 24 路负载比 F1 慢 {abs(100 * (p24 / fp - 1)):.1f}%（合计也低 "
             f"{abs(100 * (a24 / fa - 1)):.1f}%）。"
             "C8、C16 两个配置都有对应图，实测差 −0.7% / −0.1%，在噪声内。")
    L.append("")
    L.append("- **C32 已经贴近 KV 上限**：32 × 8K 时 KV 峰值 95.1–95.3%（池 410,988 token）、wait 28、"
             "0 抢占；再长一点或再多一条请求就会开始抢占。")
    L.append("- **启动代价**：F1C32 511 s 健康（图捕获 93 s / 1.02 GiB），F1 457 s（55 s / 1.04 GiB）；"
             "多一个捕获尺寸多约 38 s。两者的 KV 容量相同（410,988 token）、引擎日志都是 0 错误、"
             "needle 都是 24/24。")
else:
    L.append("_F1C32 的 sweep 日志还没到（缺 24 或 32 那一格）。_")
L.append("")
L.append("同一台引擎实例上的完整曲线（F1C32 的 c8/c16/c24/c32 与 F1 的 c24 参照）：")
L.append("")
L.append(md_table(
    ["并发", "F1（seqs 24）", "F1C32（seqs 32）"],
    [[f"**{c}**",
      (cell(data["F1"].get(c, [])) + extras(data["F1"].get(c, []), c)) if data["F1"].get(c) else "—",
      (cell(c32.get(c, [])) + extras(c32.get(c, []), c)) if c32.get(c) else "—"]
     for c in (8, 16, 24, 32)],
))
L.append("")
text = "\n".join(L) + "\n"
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(text)
filled = {a: sorted(k for k in data[a] if isinstance(k, int)) for a in ARMS}
print(f"[written {OUT}]")
for a in ARMS:
    print(f"  {a}: concs {filled[a]}  long_rows={len(data[a]['long'])}  l128k_rows={len(data[a]['l128k'])}")
