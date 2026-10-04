"""make_vision_cmp_report.py DIR OLD_DIR : RESULTS.md of sx_bench/results/2026-10-04-vision-vs-official (both sides with the vision tower loaded)
from the raw logs. OLD_DIR = results/2026-10-03-fork-vs-official-v1.5.1 (the same tests with --language-model-only) for the "what vision costs each side" table.
Every number is computed here from the pfx_bench logs, the mtp_bench JSON, the image_bench JSON, the engine logs and vision_cmp.log."""
import io
import json
import os
import re
import sys

D, OLD = sys.argv[1], sys.argv[2]
VLOG = io.open(os.path.join(D, "vision_cmp.log"), encoding="utf-8", errors="replace").read()
PASS = re.compile(r"pass (\d+) C(\d+)\s+len\s+(\d+).*?prefill\s+([\d.]+) tok/s \| decode\s+([\d.]+) tok/s per stream,\s+([\d.]+) agg.*?kv peak ([\d.]+)%.*?wait (\d+)")


def passes(d, arm, name):
    rows = []
    path = os.path.join(d, "%s_%s.log" % (arm, name))
    if not os.path.exists(path):
        return rows
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = PASS.search(line)
        if m:
            rows.append(dict(p=int(m.group(1)), c=int(m.group(2)), n=int(m.group(3)), prefill=float(m.group(4)), dec=float(m.group(5)), agg=float(m.group(6)), kv=float(m.group(7)), wait=int(m.group(8))))
    return rows


def mean(v):
    return sum(v) / len(v)


def pct(a, b):
    return "%+.0f%%" % (100 * (a / b - 1))


def kv_tokens(d, arm):
    t = io.open(os.path.join(d, "engine_%s.log" % arm), encoding="utf-8", errors="replace").read()
    return int(re.search(r"GPU KV cache size: ([\d,]+) tokens", t).group(1).replace(",", ""))


def healthy(arm):
    return int(re.search(r"\[%s\] healthy after (\d+)s" % arm, VLOG).group(1))


def errs(arm):
    m = re.search(r"\[%s\] errors in engine log: (\d+)\s+OOM lines: (\d+)" % arm, VLOG)
    return int(m.group(1)), int(m.group(2))


def gpumem(arm):
    return int(re.search(r"\[%s\] gpu mem \(MiB\): (\d+)" % arm, VLOG).group(1))


def answers(d, arm):
    a = json.load(io.open(os.path.join(d, "%s_answers.json" % arm), encoding="utf-8"))
    return [x["answer"] for x in a["needle"]], [x["answer"] for x in a["short"]], (sum(x["correct"] for x in a["needle"]), sum(x["of"] for x in a["needle"]), sum(1 for x in a["short"] if x["ok"]))


ARMS = {  # arm: (label, util, max-num-seqs)
    "FV1": ("本分支 1004，前缀缓存开", "0.90", 24), "VV2b": ("官方 v1.5.1，前缀缓存开", "0.90", 16),
    "FV2": ("本分支 1004，前缀缓存关", "0.90", 24), "VV1": ("官方 v1.5.1，前缀缓存关", "0.94", 16),
    "FVM": ("本分支 1004，开 MTP", "0.87", 16), "VVMa": ("官方 v1.5.1，开 MTP（每步 prefill 4096）", "0.92", 16),
}
o = []
o.append("# 本分支 1004 对官方 v1.5.1，两边都开视觉（2026-10-04）\n")
o.append("同一台机器（4 张 V100-SXM2-32GB，TP4，GPU 4–7）、同一个模型、同一批 prompt（同一个随机种子），一次只起一个实例。**两边都去掉 `--language-model-only`**，加上同样的 "
         "`--limit-mm-per-prompt={\"image\":1,\"video\":0}` 和 `--mm-processor-cache-gb=2`；其余参数照 2026-10-03 那次对比里各自能稳定运行的配置（官方前缀缓存开用 0.90，关用 0.94，开 MTP 用 0.92 加每步 prefill 4096）。"
         "本分支用 1004 的验证镜像，官方用 v1.5.1 发布版的 wheel（装在 Ubuntu 24.04 镜像里，不是官方原样镜像，原因见 README 第 5 组）。官方 `main` 没有测（上次起不来）。\n")
o.append("**开视觉时两边都没有进各自的快速通道**：本分支启动日志里没有“quality-qualified … no-MTP path”那批自动设置和 dual-compile 通道，graph 捕获只有 6 秒、0.29 GiB（不开视觉是 1.04 GiB）；"
         "官方的加速报告里 profile 从 `qwen4exp_fp16_decode` 变成通用的 `qwen38_27b_nvfp4_dflash2`，graph 捕获 6 秒、0.21 GiB。所以这一轮比的是两边的通用路径。\n")
o.append("## 配置、容量和稳定性\n")
o.append("| 配置 | util | 并发上限 | KV 容量（token） | 启动到就绪（秒） | 引擎日志错误 / 显存溢出 | 测试后每卡显存（MiB） | needle / short |")
o.append("|---|---|---|---|---|---|---|---|")
ans = {}
for a, (label, util, seqs) in ARMS.items():
    n, s, ok = answers(D, a)
    ans[a] = (n, s)
    e = errs(a)
    o.append("| %s | %s | %d | {:,} | %d | %d / %d | {:,} | %d/%d，%d/4 |".format(kv_tokens(D, a), gpumem(a)) % (label, util, seqs, healthy(a), e[0], e[1], ok[0], ok[1], ok[2]))
same_std = all(ans[a] == ans["FV1"] for a in ("VV2b", "FV2", "VV1"))
same_mtp = ans["FVM"] == ans["VVMa"]
o.append("\n答案：前缀缓存开、关四组（不开 MTP）的 needle 和 short 答案**%s**；两个开 MTP 的组之间**%s**（needle 长度不同，只和同类比）。" % ("逐字相同" if same_std else "不完全相同", "逐字相同" if same_mtp else "不完全相同"))
try:
    for arm, old_arm in (("FV1", "F1"), ("FVM", "FM")):
        on, os_, _ = answers(D, arm)
        no, nos, _ = answers(OLD, old_arm)
        o.append("本分支 %s（开视觉）和 2026-10-03 的 %s（不开视觉）：needle 答案%s，short 答案%s。" % (arm, old_arm, "逐字相同" if on == no else "**不同**", "逐字相同" if os_ == nos else "**不同**"))
except Exception as e:  # noqa: BLE001
    o.append("（和不开视觉的答案没能比较：%s）" % e)
o.append("")


def sweep_table(f_arm, o_arm, title):
    o.append("## %s\n" % title)
    o.append("8K 输入、贪心，2 遍平均。decode 是每路速度，prefill 是这一组请求的合计。\n")
    o.append("| 并发 | 本分支 decode | 官方 decode | 本分支相对官方 | 本分支 prefill | 官方 prefill | 本分支相对官方 | 官方等待中的请求（最多） |")
    o.append("|---|---|---|---|---|---|---|---|")
    for c in (1, 2, 4, 8, 16, 24):
        f = passes(D, f_arm, "sweep_c%d" % c)
        r = passes(D, o_arm, "sweep_c%d" % c)
        if not f:
            continue
        fd, fp = mean([x["dec"] for x in f]), mean([x["prefill"] for x in f])
        if r:
            od, op = mean([x["dec"] for x in r]), mean([x["prefill"] for x in r])
            o.append("| %d | %.1f | %.1f | %s | %.0f | %.0f | %s | %d |" % (c, fd, od, pct(fd, od), fp, op, pct(fp, op), max(x["wait"] for x in r)))
        else:
            o.append("| %d | %.1f（合计 %.0f） | 官方最多 16 路 | — | %.0f | — | — | — |" % (c, fd, mean([x["agg"] for x in f]), fp))
    o.append("\n4 条同时到达，每条 8K / 16K / 32K / 64K（默认采样，2 遍平均）和单条 128K（1 遍）：\n")
    o.append("| 长度 | 本分支 prefill | 官方 prefill | 本分支相对官方 | 本分支 decode 每路 | 官方 decode 每路 | 本分支相对官方 | KV 占用峰值：本分支 / 官方 | 官方等待中的请求 |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for name, lens in (("long", (8000, 16000, 32000, 64000)), ("l128k", (128000,))):
        for n in lens:
            f = [x for x in passes(D, f_arm, name) if x["n"] == n]
            r = [x for x in passes(D, o_arm, name) if x["n"] == n]
            fd, fp = mean([x["dec"] for x in f]), mean([x["prefill"] for x in f])
            od, op = mean([x["dec"] for x in r]), mean([x["prefill"] for x in r])
            dec_cmp = pct(fd, od) if od > 0 else "—"
            odtxt = "%.1f" % od if od > 0 else "**没有**（4 条放不下，只有 2 条同时运行）"
            o.append("| %dK | %.0f | %.0f | %s | %.1f | %s | %s | %.0f%% / %.0f%% | %d |" % (n // 1000, fp, op, pct(fp, op), fd, odtxt, dec_cmp, max(x["kv"] for x in f), max(x["kv"] for x in r), max(x["wait"] for x in r)))
    o.append("")


sweep_table("FV1", "VV2b", "前缀缓存开（本分支 util 0.90，官方 util 0.90）")
sweep_table("FV2", "VV1", "前缀缓存关（本分支 util 0.90，官方 util 0.94）")

# ---- MTP
o.append("## 开 MTP（k=4，上下文 32,768，16 路；本分支 util 0.87，官方 util 0.92 加每步 prefill 4096）\n")
mt = {}
for arm in ("FVM", "VVMa"):
    d = json.load(io.open(os.path.join(D, "%s_mtp.json" % arm), encoding="utf-8"))
    mt[arm] = {(c["workload"], c["conc"]): c for c in d["cells"]}
o.append("自然 prompt（`mtp_bench.py`，每格 1 遍）每请求 decode（token/s）和每轮产出的 token：\n")
o.append("| 负载 | 本分支 | 官方 | 本分支相对官方 |")
o.append("|---|---|---|---|")
for (wl, c), label in ((("code", 1), "写代码，单请求"), (("code", 4), "写代码，4 并发每路"), (("chat", 1), "聊天，单请求"), (("chat", 4), "聊天，4 并发每路")):
    f, r = mt["FVM"][(wl, c)], mt["VVMa"][(wl, c)]
    o.append("| %s | %.1f（每轮 %.2f） | %.1f（每轮 %.2f） | %s |" % (label, f["decode_tok_s_mean"], f["tokens_per_round"], r["decode_tok_s_mean"], r["tokens_per_round"], pct(f["decode_tok_s_mean"], r["decode_tok_s_mean"])))
o.append("\n8K 输入、贪心，2 遍平均：\n")
o.append("| 并发 | 本分支 decode 每路 | 官方 decode 每路 | 本分支相对官方 | 本分支 prefill | 官方 prefill | 本分支相对官方 |")
o.append("|---|---|---|---|---|---|---|")
for c in (1, 4):
    f, r = passes(D, "FVM", "mtp_c%d_8k" % c), passes(D, "VVMa", "mtp_c%d_8k" % c)
    fd, od, fp, op = mean([x["dec"] for x in f]), mean([x["dec"] for x in r]), mean([x["prefill"] for x in f]), mean([x["prefill"] for x in r])
    o.append("| %d | %.1f | %.1f | %s | %.0f | %.0f | %s |" % (c, fd, od, pct(fd, od), fp, op, pct(fp, op)))
o.append("")

# ---- images
o.append("## 图片请求（`image_bench.py`）\n")
o.append("每条请求带一张自己的 448×448 噪声 PNG（多模态处理缓存不会命中）和一句提示，贪心，64 个新 token，流式。C1：6 条一条接一条；C4：8 条，每次 4 条。先发一条预热（不计）。"
         "“decode 每请求”靠流式的首字时刻算，**官方 C4 上这个值会被流式分块弄得不可信，所以只比首字和端到端时间**。\n")
o.append("| 配置 | C1 首字 (s) | C1 端到端 (s) | C4 首字 (s) | C4 端到端 (s) | C4 总耗时 8 条 (s) | 失败 |")
o.append("|---|---|---|---|---|---|---|")
for a, (label, _, _) in ARMS.items():
    j = json.load(io.open(os.path.join(D, "%s_img.json" % a), encoding="utf-8"))
    cells = []
    fail = 0
    for k in ("C1", "C4"):
        ok = [r for r in j[k]["requests"] if "error" not in r and r.get("ttft")]
        fail += j[k]["errors"]
        cells.append((mean([r["ttft"] for r in ok]), mean([r["e2e"] for r in ok]), j[k]["wall"]))
    o.append("| %s | %.2f | %.2f | %.2f | %.2f | %.1f | %d |" % (label, cells[0][0], cells[0][1], cells[1][0], cells[1][1], cells[1][2], fail))
o.append("\n另外每个实例还发了两张 224×224 的纯色图（红、蓝）问主色，12 次回答全部正确（`Red` / `Blue`，92 个 prompt token）。\n")

# ---- vision tax
o.append("## 开视觉对两边各慢多少（同一批测试，和不开视觉比）\n")
o.append("不开视觉的数据来自 2026-10-03 的对比（本分支 1003 的代码，官方同一个 v1.5.1 镜像），测法相同。1003 和 1004 的区别只有 KV 稳态预算默认开，不影响速度（README 第 8 组）。\n")
o.append("| 测试 | 本分支：不开视觉 → 开视觉 | 官方：不开视觉 → 开视觉 |")
o.append("|---|---|---|")


def avg(d, arm, name, key, n=None):
    r = [x for x in passes(d, arm, name) if n is None or x["n"] == n]
    return mean([x[key] for x in r])


for title, (f_on, o_on, f_off, o_off) in (("前缀缓存开", ("FV1", "VV2b", "F1", "V2b")), ("前缀缓存关", ("FV2", "VV1", "F2", "V1"))):
    for label, name, key, n in (("1 并发 decode 每路", "sweep_c1", "dec", None), ("16 并发 decode 每路", "sweep_c16", "dec", None), ("1 并发 prefill", "sweep_c1", "prefill", None),
                                ("16 并发 prefill", "sweep_c16", "prefill", None), ("4×64K prefill", "long", "prefill", 64000), ("单条 128K prefill", "l128k", "prefill", None)):
        try:
            a1, a2 = avg(OLD, f_off, name, key, n), avg(D, f_on, name, key, n)
            b1, b2 = avg(OLD, o_off, name, key, n), avg(D, o_on, name, key, n)
            o.append("| %s，%s | %.0f → %.0f（%s） | %.0f → %.0f（%s） |" % (title, label, a1, a2, pct(a2, a1), b1, b2, pct(b2, b1)))
        except Exception:  # noqa: BLE001
            pass
o.append("")
o.append("开 MTP（每请求 decode，token/s）：\n")
o.append("| 负载 | 本分支：不开视觉 → 开视觉 | 官方：不开视觉 → 开视觉 |")
o.append("|---|---|---|")
try:
    old = {}
    for arm in ("FM", "VMa"):
        d = json.load(io.open(os.path.join(OLD, "%s_mtp.json" % arm), encoding="utf-8"))
        old[arm] = {(c["workload"], c["conc"]): c for c in d["cells"]}
    for (wl, c), label in ((("code", 1), "写代码，单请求"), (("code", 4), "写代码，4 并发每路"), (("chat", 1), "聊天，单请求"), (("chat", 4), "聊天，4 并发每路")):
        a1, a2 = old["FM"][(wl, c)]["decode_tok_s_mean"], mt["FVM"][(wl, c)]["decode_tok_s_mean"]
        b1, b2 = old["VMa"][(wl, c)]["decode_tok_s_mean"], mt["VVMa"][(wl, c)]["decode_tok_s_mean"]
        o.append("| %s | %.1f → %.1f（%s） | %.1f → %.1f（%s） |" % (label, a1, a2, pct(a2, a1), b1, b2, pct(b2, b1)))
except Exception as e:  # noqa: BLE001
    o.append("| （没能读到 | 不开视觉的 MTP 数据：%s） | |" % e)
io.open(os.path.join(D, "RESULTS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(o) + "\n")
print("written")
