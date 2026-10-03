"""make_vision_tables.py : RESULTS.md of sx_bench/results/2026-10-04-vision-fp8-probe from the raw pfx_bench logs (2 passes each)."""
import io
import os
import re
import sys

D = sys.argv[1]
ARMS = [("F1", "vp", "生产参数（不开视觉，FP16 KV）"), ("T_VISION", "vp", "只去掉 `--language-model-only`（开视觉）"),
        ("T_FP8", "vp", "只加 `--kv-cache-dtype fp8_e4m3`（不开视觉）")]
PASS = re.compile(r"pass (\d+) C(\d+).*?prefill +([\d.]+) tok/s \| decode +([\d.]+) tok/s per stream")


def read(path):
    rows = {}
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = PASS.search(line)
        if m:
            rows.setdefault((int(m.group(2)), int(m.group(1))), (float(m.group(3)), float(m.group(4))))
    return rows


data = {}
for arm, tag, _ in ARMS:
    r = {}
    for c in (1, 2):
        r.update(read(os.path.join(D, "%s_%s_c%d.log" % (arm, tag, c))))
    data[arm] = r
r = {}
for c in (1, 2):
    r.update(read(os.path.join(D, "theirs_command", "THEIRS_c%d.log" % c)))
data["THEIRS"] = r


def mean(arm, conc, idx):
    v = [data[arm][(conc, p)][idx] for p in (0, 1) if (conc, p) in data[arm]]
    return sum(v) / len(v)


def kv(arm):
    path = os.path.join(D, "engine_%s_vp.log" % arm) if arm != "THEIRS" else os.path.join(D, "theirs_command", "engine_THEIRS.log")
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", io.open(path, encoding="utf-8", errors="replace").read())
    return int(m.group(1).replace(",", ""))


def pct(a, b):
    return "%+.0f%%" % (100 * (a / b - 1))


o = []
o.append("# 开视觉和 FP8 KV 各自的代价（2026-10-04，1003 镜像，4×V100，TP4，GPU 4–7）\n")
o.append("同一台机器、同一个镜像（`heavily-modified-v1-1003-sm70main`）、同一套压测（8K 输入、256 个贪心 token、种子 2026100101、每格 2 遍取平均），一次只改一个启动参数；"
         "“他们的命令”那一行是更早一次副线测试（同样的镜像和压测，脚本和原始数据在 `theirs_command/`），他们的命令同时带了下面两项，还有 `--max-num-seqs 2`、`--max-model-len 262144`、`--quantization modelopt_fp4`、`--mamba-cache-mode align`、"
         "`--default-chat-template-kwargs`（思考默认开）和 `--compilation-config` 只用 piecewise graph。\n")
o.append("| 配置 | 1 并发 prefill (token/s) | 相对生产 | 1 并发 decode 每路 | 相对生产 | 2 并发 prefill | 相对生产 | 2 并发 decode 每路 | 相对生产 | KV 容量 (token) |")
o.append("|---|---|---|---|---|---|---|---|---|---|")
base = "F1"
names = {"F1": "生产参数（不开视觉，FP16 KV）", "T_VISION": "只开视觉（去掉 `--language-model-only`）", "T_FP8": "只用 FP8 KV（`--kv-cache-dtype fp8_e4m3`）",
         "THEIRS": "他们的完整命令（开视觉 + FP8 KV + 其他）"}
for arm in ("F1", "T_VISION", "T_FP8", "THEIRS"):
    p1, d1, p2, d2 = mean(arm, 1, 0), mean(arm, 1, 1), mean(arm, 2, 0), mean(arm, 2, 1)
    o.append("| %s | %.0f | %s | %.1f | %s | %.0f | %s | %.1f | %s | %s |" % (
        names[arm], p1, "—" if arm == base else pct(p1, mean(base, 1, 0)), d1, "—" if arm == base else pct(d1, mean(base, 1, 1)),
        p2, "—" if arm == base else pct(p2, mean(base, 2, 0)), d2, "—" if arm == base else pct(d2, mean(base, 2, 1)), "{:,}".format(kv(arm))))
o.append("\n两遍的原始值（prefill / decode）：\n")
o.append("| 配置 | 1 并发 第 1 遍 | 1 并发 第 2 遍 | 2 并发 第 1 遍 | 2 并发 第 2 遍 |")
o.append("|---|---|---|---|---|")
for arm in ("F1", "T_VISION", "T_FP8", "THEIRS"):
    cells = ["%.0f / %.1f" % data[arm][(c, p)] for c in (1, 2) for p in (0, 1)]
    o.append("| %s | %s |" % (names[arm], " | ".join(cells)))
o.append("")
o.append("## 读法\n")
o.append("- **开视觉和 FP8 KV 各自都会让 decode 慢约四分之一，两个一起也不会更慢**（单项 −23% / −24%，两项合起来 −25%）：日志里能看到原因。不开视觉、FP16 KV 的生产参数启动时有 6 条“quality-qualified … no-MTP path”自动设置和一条“Auto-enabling the SM70 Qwen3.8 dual-compile lane”；"
         "只开视觉或只用 FP8 KV 时这 7 条都没有出现（自动设置的总数从 18 条降到 12 条），快速通道整体没进。所以只改一项还不够，两项都要去掉。")
o.append("- **开视觉还多损失 prefill**（1 并发 −36%；双编译通道没进，推测大块 prefill 因此走了另一条路径，没有单独验证）；FP8 KV 单独用时 1 并发 prefill 只低约 7%，但 KV 容量多 81%（744,150 对 410,988 token）。开视觉的 KV 容量少 5%（388,772 token；推测是视觉塔和编码器缓存占了显存，没有单独验证）。")
o.append("- **输出**：三组的 needle（8 个验证码）答案逐字相同，4 道短题答案也逐字相同（第 1 题 37×43+125 三组都答 1721，是模型自己的错）。FP8 KV 启动时日志有“QSA E4M3 scale overlay is incomplete: 0/24 local K/V scales loaded”类的警告（没有标定的 scale，按 1 运行），这次测的几个问题没出现差异，不等于没有数值风险，见 README“E4M3 KV”一节。")
o.append("- **视觉功能本身能用**：开视觉的实例上发了两张 224×224 的纯色图（红、蓝）问主色，回答 `Red` 和 `Blue`，每张 92 个 prompt token。")
o.append("- 第 2 遍和第 1 遍偏差最大的格子是 FP8 2 并发的 prefill（4,056 对 5,974）和他们命令的 2 并发 prefill（2,957 对 5,727），都是第 1 遍偏低；这类格子没有单独验证原因，上面的“相对生产”按两遍平均算，prefill 的这两格因此不太可靠，decode 的数字两遍之间相差 1% 以内（开视觉 1 并发相差 4%）。")
io.open(os.path.join(D, "RESULTS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(o) + "\n")
print("written")
