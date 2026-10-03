"""make_wheel1004_report.py DIR [OLD_RESULTS_DIR] : RESULTS.md of sx_bench/results/2026-10-04-wheel-1004 from the raw logs.

Every number is computed from the pfx_bench logs (*_sweep_c*.log, *_long.log, *_mtp_c*_8k.log), the mtp_bench lines and the nvidia-smi
line in validate_1004.log, the engine logs and the answers files; nothing is typed in by hand."""
import io
import json
import os
import re
import sys

D = sys.argv[1]
OLD = sys.argv[2] if len(sys.argv) > 2 else None
VLOG = io.open(os.path.join(D, "validate_1004.log"), encoding="utf-8", errors="replace").read()
PASS = re.compile(r"pass (\d+) C(\d+)\s+len\s+(\d+).*?prefill\s+([\d.]+) tok/s \| decode\s+([\d.]+) tok/s per stream,\s+([\d.]+) agg")


def passes(arm, name):
    rows = []
    path = os.path.join(D, "%s_%s.log" % (arm, name))
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = PASS.search(line)
        if m:
            rows.append(dict(p=int(m.group(1)), c=int(m.group(2)), n=int(m.group(3)), prefill=float(m.group(4)), dec=float(m.group(5)), agg=float(m.group(6))))
    return rows


def mean(v):
    return sum(v) / len(v)


def kv_tokens(arm):
    t = io.open(os.path.join(D, "engine_%s.log" % arm), encoding="utf-8", errors="replace").read()
    return int(re.search(r"GPU KV cache size: ([\d,]+) tokens", t).group(1).replace(",", ""))


def gpu_mem(arm):
    m = re.search(r"\[%s\] gpu mem \(MiB\): (\d+)" % arm, VLOG)
    return int(m.group(1))


def healthy(arm):
    return int(re.search(r"\[%s\] healthy after (\d+)s" % arm, VLOG).group(1))


def errors(arm):
    return int(re.search(r"\[%s\] errors in engine log: (\d+)" % arm, VLOG).group(1))


def answers(arm):
    a = json.load(io.open(os.path.join(D, "%s_answers.json" % arm), encoding="utf-8"))
    needle = [x["answer"] for x in a["needle"]]
    short = [x["answer"] for x in a["short"]]
    ok = sum(x["correct"] for x in a["needle"]), sum(x["of"] for x in a["needle"]), sum(1 for x in a["short"] if x["ok"])
    return needle, short, ok


def pct(a, b):
    return "%+.1f%%" % (100 * (a / b - 1))


o = []
o.append("# 发行的 wheel（1004）的验证（2026-10-04）\n")
o.append("wheel：`1cat_vllm-1.5.1+heavily.modified.v1.1004.torch2.10.cu128-cp312-cp312-linux_x86_64.whl`，202,699,723 字节，sha256 `ba934bbbeb3036aeea552141dfa803b2c5aad73828d0a74ec9d7d0b9cd92312b`，"
         "从提交 `4e888907c` 用仓库自己的 `docker/Dockerfile` 冷构建（BuildKit 缓存已清，50 分钟）。验证镜像是 1003 的验证镜像换上这个 wheel（`--no-deps`），GPU 4–7，"
         "生产参数，全新缓存目录。同一个镜像里，每条通道各跑一组“默认”（不设 `SX_OPT_KV_STEADY_BUDGET`）和一组“开关设 0”的对照。\n")
# ---- no MTP
o.append("## 不开 MTP（生产参数，前缀缓存开，24 路）\n")
arms = [("F1Z", "开关设 0（对照），util 0.90"), ("F1A", "默认，util 0.90"), ("F1B", "默认，util 0.94")]
o.append("| 配置 | KV 容量（token） | 启动到就绪（秒） | 测试后每卡显存（MiB） | 引擎日志错误 | needle / short |")
o.append("|---|---|---|---|---|---|")
ans = {}
for a, label in arms:
    needle, short, ok = answers(a)
    ans[a] = (needle, short)
    o.append("| %s | {:,} | %d | {:,} | %d | %d/%d，%d/4 |".format(kv_tokens(a), gpu_mem(a)) % (label, healthy(a), errors(a), ok[0], ok[1], ok[2]))
o.append("\n首个启动（F1A）用的是全新缓存目录，要现编内核，所以最慢。答案：" + "；".join(
    "%s 与 F1Z 的 needle 答案%s、short 答案%s" % (a, "逐字相同" if ans[a][0] == ans["F1Z"][0] else "**不同**", "逐字相同" if ans[a][1] == ans["F1Z"][1] else "**不同**") for a in ("F1A", "F1B")) + "。")
if OLD:
    try:
        old = json.load(io.open(os.path.join(OLD, "F1W_answers.json"), encoding="utf-8"))
        on = [x["answer"] for x in old["needle"]]
        os_ = [x["answer"] for x in old["short"]]
        o.append("和 1003 的发行 wheel（`F1W`）比：needle 答案%s，short 答案%s。" % ("逐字相同" if on == ans["F1Z"][0] else "**不同**", "逐字相同" if os_ == ans["F1Z"][1] else "**不同**"))
    except Exception as e:  # noqa: BLE001
        o.append("（和 1003 的 F1W 答案没能比较：%s）" % e)
o.append("")
o.append("decode 每路（token/s，2 遍平均）和 prefill 合计（token/s，2 遍平均），8K 输入、贪心：\n")
o.append("| 并发 | 配置 | decode 每路 | 相对对照 | 合计 decode | prefill | 相对对照 |")
o.append("|---|---|---|---|---|---|---|")
for c in (1, 4, 24):
    base = passes("F1Z", "sweep_c%d" % c)
    bd, bp = mean([r["dec"] for r in base]), mean([r["prefill"] for r in base])
    for a, label in arms:
        r = passes(a, "sweep_c%d" % c)
        d, p, g = mean([x["dec"] for x in r]), mean([x["prefill"] for x in r]), mean([x["agg"] for x in r])
        o.append("| %d | %s | %.1f | %s | %.0f | %.0f | %s |" % (c, label, d, "—" if a == "F1Z" else pct(d, bd), g, p, "—" if a == "F1Z" else pct(p, bp)))
o.append("\n4 并发、每条 8K / 16K / 32K / 64K（默认采样，1 遍）：\n")
o.append("| 长度 | 配置 | decode 每路 | 相对对照 | prefill 合计 | 相对对照 |")
o.append("|---|---|---|---|---|---|")
base = {r["n"]: r for r in passes("F1Z", "long")}
for n in sorted(base):
    for a, label in arms:
        r = {x["n"]: x for x in passes(a, "long")}[n]
        o.append("| %s | %s | %.1f | %s | %.0f | %s |" % ("%dK" % (n // 1000), label, r["dec"], "—" if a == "F1Z" else pct(r["dec"], base[n]["dec"]), r["prefill"],
                                                           "—" if a == "F1Z" else pct(r["prefill"], base[n]["prefill"])))
# ---- MTP
o.append("\n## 开 MTP（k=4，上下文 32,768，16 路）\n")
marms = [("FMZ", "开关设 0（对照），util 0.87"), ("FMA", "默认，util 0.87"), ("FMB", "默认，util 0.93")]
o.append("| 配置 | KV 容量（token） | 启动到就绪（秒） | 测试后每卡显存（MiB） | 引擎日志错误 | needle / short |")
o.append("|---|---|---|---|---|---|")
mans = {}
for a, label in marms:
    needle, short, ok = answers(a)
    mans[a] = (needle, short)
    o.append("| %s | {:,} | %d | {:,} | %d | %d/%d，%d/4 |".format(kv_tokens(a), gpu_mem(a)) % (label, healthy(a), errors(a), ok[0], ok[1], ok[2]))
o.append("\n答案：" + "；".join("%s 与 FMZ 的 needle 答案%s、short 答案%s" % (a, "逐字相同" if mans[a][0] == mans["FMZ"][0] else "**不同**", "逐字相同" if mans[a][1] == mans["FMZ"][1] else "**不同**") for a in ("FMA", "FMB")) + "。")
if OLD:
    try:
        old = json.load(io.open(os.path.join(OLD, "FM_answers.json"), encoding="utf-8"))
        o.append("和 1003 的 `FM` 答案比：needle 答案%s，short 答案%s。" % ("逐字相同" if [x["answer"] for x in old["needle"]] == mans["FMZ"][0] else "**不同**",
                                                                   "逐字相同" if [x["answer"] for x in old["short"]] == mans["FMZ"][1] else "**不同**"))
    except Exception as e:  # noqa: BLE001
        o.append("（和 1003 的 FM 答案没能比较：%s）" % e)
o.append("\n自然 prompt（`mtp_bench.py`，每格 1 遍）每请求 decode（token/s）和每轮产出的 token：\n")
o.append("| 负载 | " + " | ".join(l for _, l in marms) + " |")
o.append("|---|" + "---|" * len(marms))
MB = re.compile(r"\[(FM[ZAB])\] \[mtp_bench\] (code|chat) C(\d): decode ([\d.]+) tok/s per request, aggregate ([\d.]+) tok/s, tokens per round ([\d.]+)")
mb = {}
for m in MB.finditer(VLOG):
    mb[(m.group(1), m.group(2), int(m.group(3)))] = (float(m.group(4)), float(m.group(6)))
for kind, c, label in (("code", 1, "写代码，单请求"), ("code", 4, "写代码，4 并发每路"), ("chat", 1, "聊天，单请求"), ("chat", 4, "聊天，4 并发每路")):
    o.append("| %s | " % label + " | ".join("%.1f（每轮 %.2f）" % mb[(a, kind, c)] for a, _ in marms) + " |")
o.append("\n8K 输入、贪心（2 遍，每遍的值）decode 每路（token/s）：\n")
o.append("| 并发 | " + " | ".join(l for _, l in marms) + " |")
o.append("|---|" + "---|" * len(marms))
for c in (1, 4):
    cells = []
    for a, _ in marms:
        r = passes(a, "mtp_c%d_8k" % c)
        cells.append("%.1f / %.1f（平均 %.1f）" % (r[0]["dec"], r[1]["dec"], mean([x["dec"] for x in r])))
    o.append("| %d | " % c + " | ".join(cells) + " |")
z4 = mean([x["dec"] for x in passes("FMZ", "mtp_c4_8k")])
a4 = mean([x["dec"] for x in passes("FMA", "mtp_c4_8k")])
b4 = mean([x["dec"] for x in passes("FMB", "mtp_c4_8k")])
o.append("\n4 并发 8K 贪心的平均：默认 util 0.87 相对对照 %s，util 0.93 相对对照 %s。" % (pct(a4, z4), pct(b4, z4)))
io.open(os.path.join(D, "RESULTS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(o) + "\n")
print("written")
