"""make_report.py : builds the comparison tables from the RAW logs of the run (D:/AI/pfx-bench-20260930/cmp1003/raw).
Every number printed comes from a log line; passes are averaged; deltas are computed here, not by hand."""
import glob
import os
import re
import statistics as st

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "2026-10-03-fork-vs-official-v1.5.1")
PASS = re.compile(
    r"pass (\d+) C(\d+)\s+len\s+(\d+) prompt (\d+) \| ttft first/last\s+([\d.]+)/\s*([\d.]+) s \| prefill\s+([\d.]+) tok/s \| "
    r"decode\s+([\d.]+) tok/s per stream,\s+([\d.]+) agg.*?kv peak ([\d.]+)%.*?wait (\d+).*?preempt (\d+)"
    r"(?:.*?WAVE (\d+)/(\d+): prefill ([\d.]+) decode ([\d.]+))?")


def cells(arm, name):
    """{(conc, len): list of dict} from <arm>_<name>.log"""
    p = os.path.join(RAW, "%s_%s.log" % (arm, name))
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8", errors="replace"):
        m = PASS.search(line)
        if not m:
            continue
        g = m.groups()
        d = dict(conc=int(g[1]), length=int(g[2]), ttft_last=float(g[5]), prefill=float(g[6]), dec=float(g[7]), agg=float(g[8]),
                 kv=float(g[9]), wait=int(g[10]), preempt=int(g[11]),
                 wave=(int(g[12]), int(g[13]), float(g[14]), float(g[15])) if g[12] else None)
        out.setdefault((d["conc"], d["length"]), []).append(d)
    return out


def avg(rows, key):
    v = [r[key] for r in rows]
    return st.mean(v) if v else None


def fmt(x, nd=1):
    return "–" if x is None else ("%.*f" % (nd, x))


def pct(a, b):
    return "–" if (a is None or b is None or b == 0) else "%+.1f%%" % (100 * (a - b) / b)


def read(path):
    return open(os.path.join(RAW, path), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(RAW, path)) else ""


ALL_LOGS = "\n".join(read(f) for f in ("run_all_1003.log", "run_main_final2.log", "run_v2b.log", "run_final2.log", "repro_theirs.log", "fm_answers_rerun.log"))


def engine(arm):
    t = read("engine_%s.log" % arm)
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
    w = re.search(r"Model loading took ([\d.]+) GiB and ([\d.]+) seconds", t)
    return (int(kv.group(1).replace(",", "")) if kv else None, float(w.group(1)) if w else None)


def healthy(arm):
    m = re.search(r"\[%s\] healthy after (\d+)s" % arm, ALL_LOGS)
    return int(m.group(1)) if m else None


def gpumem(arm):
    m = re.search(r"\[%s\] gpu mem \(MiB\): (\d+) (\d+) (\d+) (\d+)\s+errors in engine log: (\d+)" % arm, ALL_LOGS)
    if m:
        return max(int(m.group(i)) for i in range(1, 5)), int(m.group(5))
    m = re.search(r"\[%s\] gpu mem \(MiB\): (\d+) (\d+) (\d+) (\d+)" % arm, ALL_LOGS)
    e = re.search(r"\[%s\] errors in engine log: (\d+)" % arm, ALL_LOGS)
    return (max(int(m.group(i)) for i in range(1, 5)) if m else None, int(e.group(1)) if e else None)


out = []
P = out.append

# ---------------------------------------------------------------- no MTP, prefix OFF
P("## 1. 不开 MTP，前缀缓存关：我们 F2 对 官方 V1\n")
P("| 并发 | 解码单流 我们 | 解码单流 官方 | 差异 | 聚合 我们 | 聚合 官方 | 8K 预填充 我们 | 8K 预填充 官方 | 差异 |")
P("|---|---|---|---|---|---|---|---|---|")
f2, v1 = {}, {}
for c in (1, 2, 4, 8, 16, 24):
    f2[c] = cells("F2", "sweep_c%d" % c).get((c, 8000), [])
    v1[c] = cells("V1", "sweep_c%d" % c).get((c, 8000), [])
    if not f2[c]:
        continue
    P("| C%d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        c, fmt(avg(f2[c], "dec")), fmt(avg(v1[c], "dec")), pct(avg(f2[c], "dec"), avg(v1[c], "dec")) if v1[c] else "官方上限 16 路",
        fmt(avg(f2[c], "agg")), fmt(avg(v1[c], "agg")), fmt(avg(f2[c], "prefill"), 0), fmt(avg(v1[c], "prefill"), 0),
        pct(avg(f2[c], "prefill"), avg(v1[c], "prefill")) if v1[c] else "–"))

# ---------------------------------------------------------------- no MTP, prefix ON
P("\n## 2. 不开 MTP，前缀缓存开：我们 F1 对 官方 V2b（两者 util 0.90）\n")
P("| 并发 | 解码单流 我们 | 解码单流 官方 | 差异 | 聚合 我们 | 聚合 官方 | 8K 预填充 我们 | 8K 预填充 官方 | 差异 | 官方 KV 峰值 | 官方并发情况 |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for c in (1, 2, 4, 8, 16, 24):
    a = cells("F1", "sweep_c%d" % c).get((c, 8000), [])
    b = cells("V2b", "sweep_c%d" % c).get((c, 8000), [])
    if not a:
        continue
    note = "–"
    if b and b[0]["wave"]:
        w = b[0]["wave"]
        note = "只有 %d/%d 能同时跑，抢占 %d 次，同时跑的这批解码 %.1f tok/s/流" % (w[0], w[1], b[0]["preempt"], w[3])
        bdec = None
    else:
        bdec = avg(b, "dec") if b else None
    P("| C%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        c, fmt(avg(a, "dec")), fmt(bdec) if b and bdec else "（装不下）" if b else "官方上限 16 路", pct(avg(a, "dec"), bdec) if bdec else "–",
        fmt(avg(a, "agg")), fmt(avg(b, "agg")) if b and bdec else "–", fmt(avg(a, "prefill"), 0), fmt(avg(b, "prefill"), 0) if b else "–",
        pct(avg(a, "prefill"), avg(b, "prefill")) if b else "–", ("%.0f%%" % avg(b, "kv")) if b else "–", note))

# ---------------------------------------------------------------- long context
P("\n## 3. 4 并发长上下文（没传温度，走模型默认采样）\n")
P("| 输入 | 解码单流 F1 | F2 | 官方 V1(缓存关) | 官方 V2b(缓存开) | 预填充 F1 | F2 | V1 | V2b | KV 峰值 F1 / V2b |")
P("|---|---|---|---|---|---|---|---|---|---|")
lf1, lf2, lv1, lv2 = (cells(a, "long") for a in ("F1", "F2", "V1", "V2b"))
for L in (8000, 16000, 32000, 64000):
    key = (4, L)

    def d(x):
        r = x.get(key, [])
        if r and r[0]["wave"]:
            return "%s (仅 %d/%d 同时跑)" % (fmt(r[0]["wave"][3]), r[0]["wave"][0], r[0]["wave"][1])
        return fmt(avg(r, "dec")) if r else "–"
    P("| %dK | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s |" % (
        L // 1000, d(lf1), d(lf2), d(lv1), d(lv2), fmt(avg(lf1.get(key, []), "prefill"), 0), fmt(avg(lf2.get(key, []), "prefill"), 0),
        fmt(avg(lv1.get(key, []), "prefill"), 0), fmt(avg(lv2.get(key, []), "prefill"), 0),
        fmt(avg(lf1.get(key, []), "kv"), 0) + "%", fmt(avg(lv2.get(key, []), "kv"), 0) + "%"))
P("\n128K 单流：")
P("| | 预填充 tok/s | 首 token 秒 | 解码 tok/s | KV 峰值 |")
P("|---|---|---|---|---|")
for arm, label in (("F1", "我们 F1(缓存开)"), ("F2", "我们 F2(缓存关)"), ("V1", "官方 V1(缓存关)"), ("V2b", "官方 V2b(缓存开)")):
    r = cells(arm, "l128k").get((1, 128000), [])
    P("| %s | %s | %s | %s | %s%% |" % (label, fmt(avg(r, "prefill"), 0), fmt(avg(r, "ttft_last")), fmt(avg(r, "dec")), fmt(avg(r, "kv"), 0)))

# ---------------------------------------------------------------- MTP
P("\n## 4. MTP（k=4）：我们 FM / FM2 对 官方 VMa\n")
mt = {}
for m in re.finditer(r"\[(\w+)\] \[mtp_bench\] (code|chat) C(\d+): decode ([\d.]+) tok/s per request, aggregate ([\d.]+) tok/s, tokens per round ([\d.]+)", ALL_LOGS):
    mt[(m.group(1), m.group(2), int(m.group(3)))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
P("| 测试 | FM 每请求 | FM2 每请求 | 官方 VMa 每请求 | FM 相对官方 | FM2 相对官方 | 每轮 token FM / FM2 / VMa |")
P("|---|---|---|---|---|---|---|")
for kind, c in (("code", 1), ("code", 4), ("chat", 1), ("chat", 4)):
    a, b, o = mt.get(("FM", kind, c)), mt.get(("FM2", kind, c)), mt.get(("VMa", kind, c))
    if a and b and o:
        P("| %s prompt C%d（贪心/采样见脚本） | %.1f | %.1f | %.1f | %s | %s | %.2f / %.2f / %.2f |" % (
            "代码" if kind == "code" else "聊天", c, a[0], b[0], o[0], pct(a[0], o[0]), pct(b[0], o[0]), a[2], b[2], o[2]))
P("")
P("| 测试 | FM | FM2 | 官方 VMa | FM 相对官方 | FM2 相对官方 |")
P("|---|---|---|---|---|---|")
for kind, c in (("code", 4), ("chat", 4)):
    a, b, o = mt.get(("FM", kind, c)), mt.get(("FM2", kind, c)), mt.get(("VMa", kind, c))
    P("| %s C4 聚合 tok/s | %.1f | %.1f | %.1f | %s | %s |" % ("代码" if kind == "code" else "聊天", a[1], b[1], o[1], pct(a[1], o[1]), pct(b[1], o[1])))
for c in (1, 4):
    rows = {arm: cells(arm, "mtp_c%d_8k" % c).get((c, 8000), []) for arm in ("FM", "FM2", "VMa")}
    P("| 8K 输入 C%d 贪心解码（每流）| %s | %s | %s | %s | %s |" % (c, fmt(avg(rows["FM"], "dec")), fmt(avg(rows["FM2"], "dec")), fmt(avg(rows["VMa"], "dec")),
                                                      pct(avg(rows["FM"], "dec"), avg(rows["VMa"], "dec")), pct(avg(rows["FM2"], "dec"), avg(rows["VMa"], "dec"))))
    P("| 8K 输入 C%d 预填充 tok/s | %s | %s | %s | %s | %s |" % (c, fmt(avg(rows["FM"], "prefill"), 0), fmt(avg(rows["FM2"], "prefill"), 0), fmt(avg(rows["VMa"], "prefill"), 0),
                                                       pct(avg(rows["FM"], "prefill"), avg(rows["VMa"], "prefill")), pct(avg(rows["FM2"], "prefill"), avg(rows["VMa"], "prefill"))))

# ---------------------------------------------------------------- capacity / startup / memory
P("\n## 5. 容量、启动、显存、错误\n")
P("| 组 | 配置 | KV 容量 tokens | 权重 GiB/卡 | 启动秒 | 显存峰值 MiB | 引擎日志错误数 | 结果 |")
P("|---|---|---|---|---|---|---|---|")
meta = [
    ("F1", "我们，前缀缓存开，24 路，util 0.90", "完整"), ("F2", "我们，前缀缓存关，24 路，util 0.90", "完整"),
    ("V1", "官方 v1.5.1，缓存关，16 路，util 0.94", "完整"), ("V2", "官方 v1.5.1，缓存开，16 路，util 0.94", "**C4 起 OOM，作废**"),
    ("V2b", "官方 v1.5.1，缓存开，16 路，util 0.90", "完整（C16、4×64K 装不下）"), ("V2c", "官方 v1.5.1，缓存开，16 路，util 0.92", "**C8 第二遍 OOM**，仅 C1/2/4/8 前半有效"),
    ("FM", "我们 MTP，util 0.87，预填充 8192", "完整"), ("FM2", "我们 MTP + 稳态 KV 预算，util 0.93，预留 1990 MiB", "完整"),
    ("VM", "官方 v1.5.1 MTP，util 0.95，预填充 8192", "**首个 8K 请求 OOM，作废**"), ("VMa", "官方 v1.5.1 MTP，util 0.92，预填充 4096", "完整"),
]
for arm, cfg, res in meta:
    kv, w = engine(arm)
    mem, err = gpumem(arm)
    P("| %s | %s | %s | %s | %s | %s | %s | %s |" % (arm, cfg, ("{:,}".format(kv) if kv else "–"), fmt(w, 2), healthy(arm) or "–", mem or "–", err if err is not None else "–", res))

# ---------------------------------------------------------------- answers
P("\n## 6. 答案检查（needle 找 8 个验证码 + 4 道短题，贪心，关思考）\n")
P("| 组 | needle | short | 第 1 题（37×43+125，正确 1716）的回答 |")
P("|---|---|---|---|")
import json
for arm in ("F1", "F2", "FM", "FM2", "V1", "V2", "V2b", "V2c", "VMa"):
    pth = os.path.join(RAW, "%s_answers.json" % arm)
    if not os.path.exists(pth):
        P("| %s | 无文件 | – | – |" % arm)
        continue
    d = json.load(open(pth, encoding="utf-8"))
    P("| %s | %s | %d/4 | %s |" % (arm, "，".join("%dw:%d/%d" % (x["words"], x["correct"], x["of"]) for x in d["needle"]), sum(x["ok"] for x in d["short"]), repr(d["short"][0]["answer"].strip())))

text = "\n".join(out)
open(os.path.join(RAW, "REPORT_TABLES.md"), "w", encoding="utf-8").write(text)
print(text)
