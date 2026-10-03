"""make_profile_tables.py : PROFILE_TABLES.md from the downloaded analysis files of the 1003 decode profile.

Usage: make_profile_tables.py [OUT.md [RAWDIR]]. Inputs (RAWDIR, default this directory): G_<len>_c<conc>.txt = analyze_trace.py output of the CUDA-graph instance (PROF_G),
E_<len>_c<conc>.txt = analyze_eager2.py output of the eager + record_shapes instance (PROF_E).
analyze_trace.py prints window TOTALS under a "per step" heading; every number below divides by the graph launches of the window.
"""
import io
import os
import re
import sys

HERE = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))  # directory with the G_*/E_* analysis files
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "PROFILE_TABLES.md")
WINDOWS = [("8000_c1", "8K x 1"), ("8000_c4", "8K x 4"), ("8000_c8", "8K x 8"), ("4000_c16", "4K x 16"), ("2000_c24", "2K x 24"), ("32000_c1", "32K x 1")]
FAMS = [("gemm", "稠密 GEMM"), ("hc", "HC 专用内核"), ("gdn/fla", "GDN"), ("moe", "MoE"), ("qsa/attn", "QSA 注意力"), ("router/topk", "router / top-k"),
        ("allreduce", "all-reduce"), ("copy/elementwise", "拷贝 / 逐元素"), ("norm", "norm"), ("other", "其他"), ("ple", "PLE")]


def parse_g(key):
    t = io.open(os.path.join(HERE, "G_%s.txt" % key), encoding="utf-8", errors="replace").read()
    m = re.search(r"span ([\d.]+) ms \| GPU busy ([\d.]+) ms \((\d+)%\) \| kernels (\d+) \| graph launches (\d+)", t)
    span, busy, _, nk, steps = float(m.group(1)), float(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
    fam = {}
    for k, v in re.findall(r"([a-z/ ]+?) ([\d.]+)(?:,|$)", re.search(r"kernel time per step by family \(ms\): (.*)", t).group(1)):
        fam[k.strip()] = float(v) / steps
    kern = []
    for line in t.splitlines():
        mm = re.match(r"\s+([\d.]+) ms\s+([\d.]+)\s+(.*)", line)
        if mm:
            kern.append((float(mm.group(1)) / steps, float(mm.group(2)) / steps, mm.group(3).strip()))
    return dict(steps=steps, span=span / steps, busy=busy / steps, kernels=nk / steps, fam=fam, kern=kern)


G = {k: parse_g(k) for k, _ in WINDOWS}
o = []
o.append("# 1003 单步 decode 的时间花在哪（2026-10-04 剖析，4×V100，TP4，生产参数，不开 MTP，FP16 KV）\n")
o.append("来源：`prof_1003.sh` 在测试 GPU 4–7 上起两个实例，每个窗口只在所有请求都出了第一个 token 之后抓 3 秒 decode，所以每个 trace 是 100–200 个 graph 步（PROF_G）或 13–15 个 eager 步（PROF_E），不是整条请求。"
         "原始分析文件在本目录 `raw/`。**下面所有“每步”数字都是窗口总量除以该窗口的 graph 启动次数**（`analyze_trace.py` 的标题写的是 per step，实际是窗口总量）。"
         "profiler 开着，所以 span 比不开时略长；按 span 折算的吞吐和 README 的实测同量级（8K×8 每路 47 对 50.1 token/s；2K×24 合计 831 对 854；单路 90 对 99.1）。完整的 trace 文件（每个窗口约 35 MB × 4 张卡）没有收进仓库，只收分析输出。\n")
o.append("## 1. CUDA graph 模式：一步多长，kernel 数，间隙\n")
o.append("| 窗口（上下文 × 并发） | graph 步数 | 一步 span (ms) | GPU 忙 (ms) | kernel 间隙 (ms) | 间隙占比 | 每步 kernel 数 | 折合合计 token/s |")
o.append("|---|---|---|---|---|---|---|---|")
for k, name in WINDOWS:
    g = G[k]
    conc = int(k.split("_c")[1])
    gap = g["span"] - g["busy"]
    o.append("| %s | %d | %.2f | %.2f | %.2f | %.0f%% | %.0f | %.0f |" % (name, g["steps"], g["span"], g["busy"], gap, 100 * gap / g["span"], g["kernels"], conc * 1000 / g["span"]))
o.append("\n“折合合计 token/s”= 并发 ÷ 一步 span，是推算值，不是单独测的。\n")
o.append("## 2. 每步 GPU 忙的时间按 kernel 家族分（ms / 步）\n")
o.append("| 窗口 | " + " | ".join(n for _, n in FAMS) + " | 合计 |")
o.append("|---|" + "---|" * (len(FAMS) + 1))
for k, name in WINDOWS:
    g = G[k]
    o.append("| %s | " % name + " | ".join("%.2f" % g["fam"].get(f, 0.0) for f, _ in FAMS) + " | %.2f |" % sum(g["fam"].get(f, 0.0) for f, _ in FAMS))
o.append("\n家族合计比“GPU 忙”略大，是因为不同 stream 上的 kernel 在时间上有重叠。注意：1 并发时 HC 走专用的 FP16 融合内核（算在“HC 专用内核”里）；4 并发起 HC 的两个大矩阵乘落到通用 cutlass GEMM（算在“稠密 GEMM”里），所以 HC 列掉下来、GEMM 列涨上去；同样，GDN 输入投影和小批量 GEMV 在 8 并发以内是自写内核（GDN 列、GEMM 列里的自写部分），16 并发起也落到通用 cutlass GEMM，所以各列不能逐行直接比。看 GEMM + HC 的总成本要把两列加起来：")
o.append("")
o.append("| 窗口 | GEMM + HC (ms/步) | 占 GPU 忙 |")
o.append("|---|---|---|")
for k, name in WINDOWS:
    g = G[k]
    s = g["fam"].get("gemm", 0) + g["fam"].get("hc", 0)
    o.append("| %s | %.2f | %.0f%% |" % (name, s, 100 * s / g["busy"]))
o.append("")
o.append("## 3. 每步最耗时的 kernel（8K×8 和 2K×24，ms/步，调用次数/步）\n")
for k, name in (("8000_c8", "8K × 8"), ("2000_c24", "2K × 24")):
    o.append("**%s**\n" % name)
    o.append("| ms/步 | 调用/步 | kernel |")
    o.append("|---|---|---|")
    for ms, calls, kn in G[k]["kern"][:10]:
        o.append("| %.2f | %.0f | `%s` |" % (ms, calls, kn.replace("|", "/")[:80]))
    o.append("")
o.append("## 4. 稠密 GEMM 的带宽利用（eager + record_shapes，8 并发，200 token 提示）\n")
o.append("eager 模式下 all-reduce 的 kernel 时间是在等最慢的卡，不代表真实通信时间，所以只看计算 kernel。下表“GB/s”= 权重字节数 ÷ kernel 时间（FP16 估算，只对稠密权重有意义）；V100-SXM2 的 HBM2 峰值约 900 GB/s。\n")
t = io.open(os.path.join(HERE, "E_200_c8.txt"), encoding="utf-8", errors="replace").read().splitlines()
rows = [ln for ln in t if ln.startswith("aten::mm") and "cutlass" in ln]
o.append("| 形状（输入 × 权重） | 调用/步 | µs/次 | ms/步 | GB/s |")
o.append("|---|---|---|---|---|")
for ln in rows[:9]:
    shape = ln[34:68].strip()
    nums = ln[34 + 35 + 45:].split()
    # columns after the 3 text columns: calls/st us/call ms/step GB/s
    m = re.search(r"\s(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d{3})\s+(\d+)\s*$", ln)
    if m:
        o.append("| `%s` | %s | %s | %s | %s |" % (shape, m.group(1), m.group(2), m.group(3), m.group(4)))
o.append("")
io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(o) + "\n")
print("written", OUT)
