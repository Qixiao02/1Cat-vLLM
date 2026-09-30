# 1Cat-vLLM-heavily-modified

基于 1Cat 官方代码的魔改分支，面向 4 张 V100 上的 Swift 1.5 Qwen3.8-Flash-Next，重点优化并发吞吐和 prefill。官方原版请看 [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。

This is a heavily modified fork of 1Cat-vLLM (branch `1cat-vllm-heavily-modified-v1`, latest tag `1cat-vllm-heavily-modified-v1-1001`, previous `1cat-vllm-heavily-modified-v1-0930`, based on upstream `main@02c87ab89`) tuned for concurrency and prefill of Swift 1.5 Qwen3.8-Flash-Next on 4x V100. See [HEAVILY_MODIFIED.md](HEAVILY_MODIFIED.md) for the changes, switches and measurements.

| 项 | 值 |
|---|---|
| 默认分支 | `1cat-vllm-heavily-modified-v1` |
| 最新版本（tag） | `1cat-vllm-heavily-modified-v1-1001` |
| 上一个版本（tag） | `1cat-vllm-heavily-modified-v1-0930` |
| 基于 | 官方 `main@02c87ab89`（v1.5.0 之后第 670 个提交） |
| 验证过的组合 | 4 张 V100-SXM2-32GB，TP4，Swift 1.5 Qwen3.8-Flash-Next NVFP4，不开 MTP |
| 下载 wheel | [Release 1cat-vllm-heavily-modified-v1-1001](https://github.com/Qixiao02/1Cat-vLLM/releases/tag/1cat-vllm-heavily-modified-v1-1001)：Linux x86_64，Python 3.12，CUDA 12.8，torch 2.10，只支持 V100（sm_70） |

目录：[效果一览](#效果一览) · [改动一览](#改动一览) · [提示：E4M3 KV 默认不生效](#提示e4m3-kv官方-pr-664默认不生效) · [实测测试对比](#实测测试对比) · [注意和已知限制](#注意和已知限制)

## 效果一览

4 张 V100，Swift 1.5 Qwen3.8-Flash-Next，不开 MTP。数据和测法见[实测测试对比](#实测测试对比)。

| 对比 | 结果 |
|---|---|
| 对官方 `main@d30469863`：4 条互不相同的长 prompt 同时到达（每条 8K–32K） | prefill 速度 **1.57–1.73 倍**，每路 decode 速度 **1.41–1.56 倍** |
| 对官方 `main@d30469863`：4 条 64K 同时到达 | 官方的 KV 缓存放不下，只能同时跑 3 条；本分支 4 条同时跑 |
| 本分支开着前缀缓存对关掉 | prefill 速度是关掉时的 **97%–100%** |
| 对起点（官方 `main@02c87ab89` 加改动第 1 项） | prefill 1.4–1.7 倍；采样 decode 每步快 18%–37%；业务 JSON 24 并发从 1.84 到 2.89 请求/秒 |

## 改动一览

“来源”一列写明每项是本分支自己做的，还是从官方移植的。本分支自己的改动都有 `SX_OPT_*` 环境变量开关，默认开，设为 `0` 就回到官方原路径。

| # | 改动 | 解决什么 | 来源 | 默认 | 版本 |
|---|---|---|---|---|---|
| 1 | Flash-V100 分组验证和 DFlash2 speculator | DFlash2 验证步把多条请求合并成一次调用 | **本分支** | 开 | 0930 |
| 2 | 第一批（Python/Triton）：去掉主机同步；开着前缀缓存时一步 prefill 跨多个状态块；多行内核；PLE 分组打包 | prefill 速度，并发 decode | **本分支** | 开 | 0930 |
| 3 | 第二批（CUDA 内核和 CUDA graph）：分组 decode 内核到 32 行；TP4 all-reduce；混合步 PIECEWISE graph | 并发 decode | **本分支** | 开 | 0930 |
| 4 | 第三批：MTP 通道优化；prefill 保护；短 prompt 不多拆一步 | MTP 通道，短 prompt 的 prefill | **本分支** | 开（MTP 本身默认不开） | 0930 |
| 5 | gated RMSNorm 精确算子，并让它真正生效 | 输出质量：单请求和并发用同一套算术 | 官方 PR #704 + **本分支**（修正放行条件） | 开（不开 MTP 时） | 0930 |
| 6 | PLE 短卷积 prefill 的缓冲从 6 块减到 2 块 | 开 MTP 时 4 条 8K 并发显存溢出 | 官方 PR #707 + **本分支**（MTP 路径分组打包） | 开 | 1001 |
| 6 | 开 MTP 时用标定过的 E4M3（8 位）存 KV 缓存 | 开 MTP 时 KV 缓存容量小 | 官方 PR #664 + **本分支**（标定打包工具） | **关**，见[提示](#提示e4m3-kv官方-pr-664默认不生效) | 1001 |
| 7 | 开着前缀缓存时及时释放换下来的状态块 | 长 prompt 的 KV 占用偏高 | **本分支** | 开 | 1001 |
| — | V100（SM70）wheel 的打包修复 | 从源码打出可用的 wheel | 官方提交 `4ab186009`、`c0e0ee66f`、`b3c9ce45f`、`6f6fc4c52`（Dockerfile 和 setup.py 的部分） | — | 1001 |

关于从官方移植的几项：

- 本分支基于官方 `main@02c87ab89`，**没有整体合并官方之后的代码**（官方最新是 `main@d30469863`，2026-09-29）。方向只有一个：从官方仓库到本分支。本分支没有向官方仓库提交过任何东西。
- 截至 2026-10-01，PR #704 和那 4 个打包提交官方自己已经合并进 main；PR #707 和 PR #664 官方还没有合并，PR 开着。
- 官方 `02c87ab89` 之后的其他改动没有移植，例如 `VLLM_SM70_QWEN38_BATCH_FASTPATH` 和前缀缓存的稀疏保留。

### 各项细节

点开看每一项具体改了什么。

<details>
<summary>名词</summary>

- **prefill**：引擎计算整段输入 prompt 的阶段。**decode**：之后逐个生成输出 token 的阶段。
- **步、行**：引擎按“步”推进，每一步对一批 token 做一次前向计算；“行”指一步里一起计算的 token 数，纯 decode 时每条请求占一行。**混合步**：同一步里既有 prefill 又有 decode。
- **主机同步**：CPU 停下来等 GPU 的结果。
- **前缀缓存、状态块**：开头相同的 prompt 复用已经算过的部分。Flash-Next 有带循环状态的层，前缀缓存以 mamba `align` 模式按 784 token 一块存放循环状态，这一块叫状态块。
- **QSA**：模型的稀疏注意力算子。**GEMV**：矩阵乘向量。**HC**：HyperConnection 残差。**MoE**：混合专家层。**PLE**：模型的 n-gram 嵌入表。
- **NVFP4**：权重 4 bit 的量化格式。**TP4**：张量并行，模型分摊在 4 张卡上。
- **CUDA graph**：把一步的 GPU 调用按固定行数录制下来重放，请求数不足时用空行补齐。FULL 是整步录制，PIECEWISE 是分段录制。
- **MTP**：用模型自带的多 token 预测头做推测解码（先草拟几个 token，再由主模型一次验证）。k 是每步草拟的 token 数。**DFlash2**：另一种推测解码方式。

</details>

<details>
<summary>1. Flash-V100 分组验证和 DFlash2 speculator（提交 <code>d3400c869</code>）</summary>

V100 注意力后端 `FLASH_ATTN_V100` 在 DFlash2 推测解码的验证步里，把多条请求合并成一次调用，并配套修改了 DFlash2 speculator。

官方 `main@02c87ab89` 加上这一项，是后面各项改动和收益对比的起点。

</details>

<details>
<summary>2. 第一批（Python/Triton）</summary>

- 采样器的 top-k/top-p 去掉每步一次的主机同步。
- 开着前缀缓存时，一步 prefill 可以跨多个状态块。官方的一步 prefill 最多推进一个状态块。
- QSA：去掉主机同步，双 warp 内核覆盖到 32 行，decode 的地址解析，混合步的内核路由。
- 稠密层的 GEMV 和 HC 增加多行内核，单请求路径的结果逐位不变。
- MoE 的路由和持久缓冲区扩到 32 行。
- PLE 在 prefill 时分组打包。

</details>

<details>
<summary>3. 第二批（CUDA 内核和 CUDA graph）</summary>

- NVFP4 MoE 的分组 decode 内核扩到 17–32 行，并跳过补齐行。
- TP4 的 push all-reduce 覆盖 10–160 KiB 的数据。
- 共享专家的门控支持多行。
- QSA 的 decode top-k 覆盖所有行。
- 25–1024 token 的混合步使用 PIECEWISE CUDA graph。

</details>

<details>
<summary>4. 第三批（3a）</summary>

- MTP 通道的优化。只有启动时开了 MTP 才会用到；MTP 默认不开，未达到生产条件，见“已知限制”。
- prefill 保护：只有 1 个 token（不开 MTP）或 k+1 个 token（开 MTP）的 prefill 分块不再重放 FULL CUDA graph。
- 短 prompt 不再为了在末尾留缓存检查点而多拆一步。

</details>

<details>
<summary>5. 移植官方 PR #704 的质量修复，并让它真正生效</summary>

- Qwen3.8 的 gated RMSNorm 改用精确的原生 CUDA 算子（`_C::sm70_rmsnorm_gated_exact_out`），结果与 PyTorch eager 的 FP32 计算逐位相同。
- 目的是让单请求和批量、decode 和混合步都用同一套算术，避免微小的舍入差异改变 MoE 路由、翻转 EOS（结束符）。
- 不开 MTP 时默认启用，由 `VLLM_SM70_RMSNORM_GATED_EXACT` 控制，设为 `0` 关闭。
- 与官方的区别：官方在 Python 里只对 1–192 行的输入放行这个算子。vLLM 对每个编译范围只追踪一次，按最大尺寸追踪，而且丢弃形状守卫，所以这个行数条件对整个范围只判断一次。Flash-Next 的 decode 图按 24 并发（这个算子的输入是 288 行）追踪，主编译按 8192 token 追踪，官方的条件在这样的部署上从来不满足，算子一次也没有运行（官方 `main@357d07bcb` 同样如此）。本分支只检查编译期不变的条件（2 维、宽 128、FP16、连续），内核接受任意行数，所以 decode、混合步和 prefill 图的每一行结果都相同。

</details>

<details>
<summary>6. MTP 通道的显存：移植官方 PR #707 和 PR #664（版本 1001 新增）</summary>

- **PR #707**：PLE 的短卷积在 prefill 时原来同时占 6 块和整批输入一样大的缓冲，现在只占 2 块，结果逐位不变。开 MTP 时这一步另外按请求长度分组打包，不再把短请求补齐到最长请求的长度。改之前，开 MTP、4 条 8K prompt 并发会在这一步显存溢出；改之后不再溢出，每步 prefill 仍是 8192 token。
- **PR #664**：开 MTP 时 KV 缓存可以用 8 位的 E4M3 格式存放（FP8 的一种：1 位符号、4 位指数、3 位尾数）。**默认不启用，KV 缓存仍是 FP16。** 怎样才生效、有哪些已知问题，见[提示](#提示e4m3-kv官方-pr-664默认不生效)；标定和启动的步骤见 [KV 缓存的格式](#kv-缓存的格式fp16-和-e4m3)；实测见第 5 组。
- 官方只为它自己发布的模型提供 scale。本分支补了从自己的标定结果打包 scale 的工具（`tools/qwen4_exp/sx_scale_pack.py`），整套流程见 [`sx_bench/as_run/e4m3_chain.sh`](sx_bench/as_run/e4m3_chain.sh)。

</details>

<details>
<summary>7. 开着前缀缓存时及时释放换下来的状态块（版本 1001 新增）</summary>

一步 prefill 跨多个状态块时，换下来的状态块原来要到请求结束才释放，现在处理完对应的 token 就释放。只改 KV 管理器，不改变分块方案、前缀命中和输出。

V100 实测（4 条冷 prompt 同时发出，KV 占用峰值，见实测第 6 组）：

| 每条输入 | 0930，开前缀缓存 | 1001，开前缀缓存 | 0930，关前缀缓存 |
|---|---|---|---|
| 8K | 13.6% | 12.9% | 11.6% |
| 16K | 23.9% | 20.8% | 18.8% |
| 32K | 44.5% | 34.5% | 34.0% |
| 64K | 86.1% | **64.4%** | 63.6% |

开着前缀缓存的 KV 占用降到了和关掉时基本一样，prefill 和 decode 速度不变。

</details>

## 提示：E4M3 KV（官方 PR #664）默认不生效

> [!IMPORTANT]
> **E4M3 KV（官方 PR #664 的移植）默认不生效。生产环境建议保持默认的 FP16。**
>
> **怎样才生效**，下面三项缺一不可：
>
> 1. 启动参数 `--kv-cache-dtype fp8_e4m3`。不写就是 FP16。
> 2. 开着 MTP 时设置环境变量 `VLLM_QWEN4EXP_QSA_E4M3_MTP=1`。不设会在启动时报错 `Qwen4Exp QSA E4M3 phase 1 requires MTP0`。
> 3. 模型目录里带有标定出来的 scale：24 个目标层 scale，开 MTP 时再加 2 个 MTP 层 scale。缺 MTP 层 scale 会拒绝启动。缺目标层 scale 时，默认只打一条警告并按 scale = 1 运行，数值可能被截断；设置 `VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` 后改为拒绝启动。
>
> 生效时启动日志里有 `Using fp8_e4m3 data type to store kv cache` 和 `QSA E4M3 calibrated scale gate passed: loaded 24/24 K/V scales`（开 MTP 时还有一条 `loaded 2/2`）。标定和启动的步骤见“注意和已知限制”里的[KV 缓存的格式](#kv-缓存的格式fp16-和-e4m3)一节。
>
> **已知问题**（数据见实测第 5 组）：
>
> - 4 并发的 decode 速度只有 FP16 KV 的一半左右（每路 30–39 tok/s，FP16 是 61–71），单请求基本不变。
> - KV 缓存容量多 65%，但开 MTP 时每条在跑的请求固定占约 13% 的缓存池，4 条 32K 并发仍然放不下。
> - 每张卡显存峰值 32,267 MiB，整卡 32,768 MiB，余量约 500 MiB。
> - scale 只覆盖标定时见过的数值范围，超出的会被截断。我们的标定只用了 18 条请求，prompt 最长 36K token。
> - 不开 MTP 时用 E4M3，本分支的各项优化会回到官方路径，这个组合没有测过。

## 实测测试对比

| 组 | 比什么 | 负载 | 结论 |
|---|---|---|---|
| [1](#1-本分支对官方-maind30469863并排2026-09-30) | 本分支 对 官方 `main@d30469863` | 4 条冷 prompt 并发，每条 8K–64K | prefill 1.57–1.73 倍，每路 decode 1.41–1.56 倍；64K 官方放不下 4 条 |
| [2](#2-本分支开关前缀缓存2026-09-30) | 本分支开 对 关前缀缓存 | 同上 | 开着时 prefill 是关掉时的 97%–100% |
| [3](#3-本分支对官方-main357d07bcb2026-09-28) | 本分支 对 官方 `main@357d07bcb` | 并发 1–24，业务负载，长文 | 24 并发总吞吐 439 对 313 token/s |
| [4](#4-本分支对起点) | 本分支 对 起点 | 同第 3 组 | prefill 1.4–1.7 倍，采样 decode 每步快 18%–37% |
| [5](#5-开-mtpfp16-kv-对-e4m3-kv2026-09-30) | 开 MTP 时 FP16 KV 对 E4M3 KV | 单请求和 4 并发，2K–32K | E4M3 的 KV 容量多 65%，4 并发 decode 减半 |
| [6](#6-发行的-wheel1001对-0930-的镜像2026-10-01) | 发行的 wheel（1001）对 0930 的镜像 | 同第 1 组，外加答题对照 | 答案逐字相同，速度相同，64K 的 KV 占用峰值从 86.1% 降到 64.4% |

共用条件：

- 硬件是 4 张 Tesla V100-SXM2-32GB，TP4。模型是 Swift 1.5 Qwen3.8-Flash-Next 的 NVFP4 量化版（PLE 表以 FP8 存储）。
- 第 1–4 组不开 MTP，除特别说明外都开着前缀缓存；第 5 组开 MTP。启动参数和环境变量见“注意”。
- 第 1、2、5、6 组的脚本和原始结果在 [`sx_bench/`](sx_bench/)，用法和字段说明见 [`sx_bench/README.md`](sx_bench/README.md)。第 3、4 组的脚本和原始记录没有收录。

<details>
<summary>第 1、2、5、6 组的指标怎么算</summary>

- **prefill 速度**：同时发出的几条请求的 prompt token 总数 ÷ 最后一条拿到首字的时间。
- **首字延迟（TTFT）**：从发出请求到收到第一个输出 token 的时间。表里的“第 1 条”“第 4 条”按拿到首字的先后排序。
- **decode 速度**：只统计几条请求都在生成的那段时间。“每路”是各条请求速度的平均，“合计”是各路之和。
- **KV 占用**：KV 缓存（存放各请求已算状态的显存池）已用的比例，每秒采样一次，表里是峰值。KV 缓存容量指它能放下的 token 数。
- **抢占**：KV 缓存用满时，引擎中止一条正在跑的请求并收回它的缓存，之后重算。
- **冷 prompt**：每条 prompt 是不同的随机文本，没有共同前缀，前缀缓存命中为 0。

</details>

### 1. 本分支对官方 `main@d30469863`，并排（2026-09-30）

4 条冷 prompt 同时发出，每条生成 400 token，2 遍平均。

速度（token/s）：

| 每条输入 | prefill 本分支 | prefill 官方 | 倍数 | decode 每路 本分支 | decode 每路 官方 | 倍数 |
|---|---|---|---|---|---|---|
| 8K | 6544 | 4162 | **1.57** | 61.9 | 43.8 | **1.41** |
| 16K | 6427 | 3972 | **1.62** | 60.4 | 38.7 | **1.56** |
| 32K | 6130 | 3551 | **1.73** | 59.3 | 40.4 | **1.47** |
| 64K | 5764 | 1870 | — | 57.1 | 40.5（只有 3 路） | — |

首字延迟和 decode 合计（本分支 / 官方）：

| 每条输入 | 第 1 条首字 | 第 4 条首字 | decode 合计（token/s） |
|---|---|---|---|
| 8K | 2.4 / 7.0 s | 4.9 / 7.7 s | 248 / 175 |
| 16K | 3.7 / 15.1 s | 10.0 / 16.1 s | 242 / 155 |
| 32K | 6.7 / 34.6 s | 20.9 / 36.1 s | 237 / 162 |
| 64K | 13.2 / 77.0 s | 44.4 / 136.9 s | 228 / 122（3 路） |

64K 不列倍数：官方的 KV 缓存放不下 4 条，只有 3 条同时运行，两边的负载不一样。KV 缓存容量是本分支 410,247 token，官方 201,421 token。

<details>
<summary>测法</summary>

- 两条线在同一台服务器上，本分支用 GPU 0–3，官方用 GPU 4–7。模型文件、启动参数和 prompt（同一个随机种子）都相同，一次只压测一条线。
- 官方这条线用官方自己的 Dockerfile 从 `main@d30469863` 干净构建，使用官方默认设置，并打开官方的可选开关 `VLLM_SM70_QWEN38_BATCH_FASTPATH=1`（批量快速路径，每张卡约占 1 GiB 显存）。本分支的代码早于这个开关，它自己的多行 decode 路径默认就是开的。
- 8K、16K、32K、64K 是每条 prompt 的 token 数（8,000 到 64,000）。
- 每个输入长度跑 2 遍，表里是 2 遍的平均值。

</details>

<details>
<summary>其他观察</summary>

- 每条 64K 时，官方的 KV 占用峰值到 99.6%，2 遍各发生 2 次抢占，第 4 条在 137 秒后才拿到首字。本分支的峰值是 86.1%，没有抢占。
- 2 遍之间的差：本分支的 prefill 和 decode 速度在各个长度上都不超过 2%。官方 8K 的 2 遍是 prefill 3905 和 4420 token/s，decode 每路 47.4 和 40.2 token/s。
- 首字的分布：本分支一条接一条地做 prefill，第 1 条的首字来得早；官方 4 条一起推进，4 个首字都来得晚，而且挨得很近。
- 本分支的代价：别的请求在 prefill 时，已经在生成的请求会停顿。相邻两个 token 的最长间隔，本分支是 1.2–1.5 秒，官方是 0.5–0.8 秒；稳定 decode 时的间隔中位数，本分支是 16–18 ms，官方是 22–24 ms。
- 各次测试期间的 GPU 平均利用率：本分支 92%–96%，官方 69%–84%。

</details>

### 2. 本分支开、关前缀缓存（2026-09-30）

开着前缀缓存时，prefill 速度是关掉时的 97%–100%，decode 速度最多相差约 3%。表里每格是“开 / 关”。

| 每条输入 | prefill（token/s） | 开 ÷ 关 | decode 每路（token/s） | 第 1 条首字 | KV 占用峰值 |
|---|---|---|---|---|---|
| 8K | 6329 / 6377 | 0.99 | 59.9 / 61.8 | 2.0 / 1.5 s | 13.6% / 11.6% |
| 16K | 6261 / 6452 | 0.97 | 60.8 / 60.2 | 4.0 / 2.7 s | 23.9% / 18.8% |
| 32K | 6111 / 6105 | 1.00 | 59.0 / 59.3 | 6.9 / 5.9 s | 44.5% / 34.0% |
| 64K | 5757 / 5841 | 0.99 | 56.5 / 57.2 | 13.2 / 11.9 s | 86.1% / 63.6% |

作为对照，官方开着前缀缓存时一步 prefill 最多推进一个状态块，一条 110K 的 prompt 要 141 个调度步，本分支是 14 个（改动第 2 项）。开着前缀缓存的两个代价是第 1 条首字晚约 1 秒、KV 占用更高，见“已知限制”；这组数据是版本 0930 的，KV 占用偏高在版本 1001 里已修（改动第 7 项）。

<details>
<summary>测法</summary>

- 方法同第 1 组：4 条冷 prompt 同时发出，每条生成 400 token，每个长度跑 2 遍取平均。
- “开”在作者的线上实例上测，测试时实例空闲；“关”用同一个镜像和同一套配置，另加 `--no-enable-prefix-caching`。
- 2 遍之间 prefill 速度最多差 7%（关、8K），其余都在 2% 以内。

</details>

### 3. 本分支对官方 `main@357d07bcb`（2026-09-28）

| 指标 | 本分支 | 官方 `main@357d07bcb` |
|---|---|---|
| 每路 token/s，并发 1 / 4 / 8 / 24 | 91 / 62 / 41 / 25 | 90 / 45 / 39 / 18 |
| 24 并发总吞吐 | 439 token/s | 313 token/s |
| 业务 JSON，8 / 24 并发 | 1.43 / 2.90 请求/秒 | 0.96 / 1.91 请求/秒 |
| 真实请求回放（1.5 请求/秒）：首字延迟 p95 / 端到端 p95 | 0.39 s / 8.4 s | 1.2 s / 19.3 s |
| 长文请求耗时，32K / 64K / 110K | 5.2 / 10.3 / 17.8 s | 11.3 / 23.8 / 39.6 s |
| 长文检索（9 处） | 9/9 | 9/9 |
| KV 缓存容量 | 410K token | 201K token |

<details>
<summary>测法和配置</summary>

- 官方 `main@357d07bcb` 是 2026-09-28 当时的最新提交，干净构建，全部原生库和 FlashAttention 都重新编译过。
- 两边的模型、压测和真实请求回放相同。每项只跑了一次，单次结果的波动估计在 ±10% 左右。
- 两列各用各的最佳配置。本分支用作者线上服务的配置，即“注意”里的环境变量：关闭 hybrid PLE 通道，打开 MoE 分组 decode。官方用它默认的 PLE 和 MoE 设置，并打开批量快速路径 `VLLM_SM70_QWEN38_BATCH_FASTPATH=1`。把本分支的线上配置直接套到官方代码上会更慢（24 并发总吞吐 210 token/s），所以不用它做对比。
- “业务 JSON”和“真实请求回放”是作者自己的业务负载。p95 是第 95 百分位，“端到端”指从发出请求到整条回答结束。
- 长文请求耗时是非流式请求的总时间，两列都开着前缀缓存。64K 和 110K 两条各有大约 10% 的 token 命中了缓存，两列相同。

</details>

### 4. 本分支对起点

起点是官方 `main@02c87ab89` 加改动第 1 项。

| 对比项 | 起点 → 本分支 |
|---|---|
| prefill 速度 | 1.4–1.7 倍 |
| 采样 decode（开着 top-k/top-p 采样）每步 | 快 18%–37% |
| 业务 JSON，24 并发 | 1.84 → 2.89 请求/秒 |

精确 gated RMSNorm（改动第 5 项）生效前后，吞吐、首字延迟和长文首字延迟都没有可测的变化。同一组 24 条请求单独跑和 8 并发跑，输出逐 token 相同的条数是 21/24 → 21/24；这个指标同一份代码测两次会在 17–21 之间波动，24 条样本分辨不出差异。

### 5. 开 MTP：FP16 KV 对 E4M3 KV（2026-09-30）

开 MTP（k=4），`--gpu-memory-utilization 0.87`，其余启动参数同“注意”。代码是版本 1001（当时还没有改动第 7 项）。

容量和显存：

| 项目 | FP16 KV | E4M3 KV |
|---|---|---|
| KV 缓存容量（token） | 131,072 | **216,820** |
| 每张卡显存峰值（MiB，整卡 32,768） | 32,019 | 32,267 |
| 每轮 token 数（一次草拟加验证平均产出的 token 数） | 2.1–2.4 | 2.1–2.4 |

decode 速度（每路，token/s）：

| 负载 | FP16 KV | E4M3 KV |
|---|---|---|
| 单请求 8K，采样 | 85.0 | 80.1 |
| 单请求 8K，贪心 | 87.5 | 84.0 |
| 4 并发 2K，采样（两遍） | 61.2、62.1 | **33.4、34.4** |
| 4 并发 2K，贪心（两遍） | 70.9、67.9 | **38.0、39.1** |
| 4 并发 8K | 62.5 | **30.8** |
| 4 并发 16K | —（放不下 4 条） | 30.2 |
| 4 并发 32K | —（放不下 4 条） | —（放不下 4 条） |

KV 占用峰值和抢占次数（4 并发）：

| 每条输入 | FP16 KV | E4M3 KV |
|---|---|---|
| 2K | 51.9%，0 次 | 52.3%，0 次 |
| 8K | 73.5%，0 次 | 67.6%，0 次 |
| 16K | 98.9%，1 次 | 88.6%，0 次 |
| 32K | 100%，3 次 | 100%，2 次 |

结论：

- E4M3 KV 下 4 并发的 decode 减半，单请求基本不变。E4M3 的验证步超过 16 行时走另一条路径，4 并发 × (k+1) 是 20 行，单请求是 5 行。这是按代码和日志做的判断，没有逐段计时确认。
- 两种 KV 下，开 MTP 时每条在跑的请求都占约 13% 的缓存池，和 prompt 长度无关（4 并发 2K 就占 52%）。这是从 KV 占用峰值反推的，原因没有查明。
- 输出对照：把 8 个验证码藏在 8.7K、34K、68K token 的文本里让模型找出来，再加 4 道短题，贪心解码。开 MTP、E4M3 KV 的实例和不开 MTP、FP16 KV 的实例结果相同：验证码 24/24，短题 3/4，短题答案逐字相同。

<details>
<summary>测法和补充数据</summary>

- prompt 和测法同第 1 组。2K 的格子跑两遍，其余每格一遍。
- prefill 速度（token/s，4 并发 8K / 16K / 32K）：FP16 KV 是 2862 / 3422 / 4164，E4M3 KV 是 4525 / 5321 / 3611。每格只有一遍，波动大，不适合用来比较两种 KV。
- E4M3 的 scale 用 18 条请求标定（代码、中英文文档、日志、JSON、对话、工具调用），prompt 最长 36K token。超出标定范围的数值会被截断。

</details>

### 6. 发行的 wheel（1001）对 0930 的镜像（2026-10-01）

wheel 用仓库自己的 `docker/Dockerfile` 从源码构建（CUDA 12.8.1，只编译 sm_70），装进干净镜像，按生产的启动参数和环境变量启动，不开 MTP。对照的是版本 0930 的镜像：答题对照在同一时间测，速度数字取第 1 组。prompt、随机种子和测法同第 1 组，2 遍平均。

| 项目 | 0930 的镜像 | 1001 的 wheel |
|---|---|---|
| KV 缓存容量 | 410,247 token | 410,247 token |
| 答题对照（贪心，7 道） | — | 答案与 0930 **逐字相同 7/7**；长文找验证码 24/24，短题 3/4，两边一样 |
| 模型加载 | 约 450 s | 280 s |

| 每条输入 | prefill（token/s）0930 / 1001 | decode 每路（token/s）0930 / 1001 | 第 1 条首字 0930 / 1001 | 第 4 条首字 0930 / 1001 | KV 占用峰值 0930 / 1001 |
|---|---|---|---|---|---|
| 8K | 6544 / 6433 | 61.9 / 62.2 | 2.4 / 2.5 s | 4.9 / 5.0 s | 13.6% / 12.9% |
| 16K | 6427 / 6414 | 60.4 / 60.9 | 3.7 / 3.8 s | 10.0 / 10.0 s | 23.9% / 20.8% |
| 32K | 6130 / 6110 | 59.3 / 59.5 | 6.7 / 6.8 s | 20.9 / 21.0 s | 44.5% / 34.5% |
| 64K | 5764 / 5799 | 57.1 / 57.4 | 13.2 / 13.0 s | 44.4 / 44.1 s | 86.1% / **64.4%** |

- 速度差都在 2% 以内，属于测量波动。两次测试用的显卡不同（0930 在 GPU 0–3，1001 在 GPU 4–7）。
- KV 占用峰值下降来自改动第 7 项。
- 答题题目见 [`sx_bench/as_run/kvq_check.py`](sx_bench/as_run/kvq_check.py)。

## 注意和已知限制

### 注意

- **适用范围**：只在 4 张 V100-SXM2-32GB、TP4、Swift 1.5 Qwen3.8-Flash-Next NVFP4（PLE 表以 FP8 存储）、不开 MTP 这一种组合上验证过。各项优化按这套硬件和模型的形状判断是否启用；其他组合会回到官方路径，能运行，但没有加速。没有针对 Qwen3.8-27B 加 DFlash2 做调优或验证。
- **构建和运行环境**：原生内核只为 sm_70（V100 的 CUDA 架构）编译。需要 Python 3.12 和 torch 2.10.0+cu128；运行时还需要 CUDA 12.8 toolkit，因为部分内核在首次启动时编译，冷启动要 8–25 分钟。PLE 表需要约 48 GiB 可锁定的主机内存。
- **安装 wheel**：从 [Release 页面](https://github.com/Qixiao02/1Cat-vLLM/releases/tag/1cat-vllm-heavily-modified-v1-1001) 下载后 `pip install <wheel 文件>`。wheel 的元数据里写明了 torch 2.10.0+cu128 的下载地址（download.pytorch.org），pip 会自动装上；flashinfer 0.6.11.post2 从 PyPI 装。建议装在单独的虚拟环境里。
- **版本标记**：默认分支是 `1cat-vllm-heavily-modified-v1`，版本用 git tag 标记。
  - `1cat-vllm-heavily-modified-v1-1001`（最新）：改动第 1–7 项。比 0930 多出的第 6、7 项只改 Python 文件；不开 MTP 时输出不变（PLE prefill 的结果逐位相同，有 CPU 测试）。第 5 组实测用的是它的代码，当时还没有第 7 项。
  - `1cat-vllm-heavily-modified-v1-0930`：改动第 1–5 项，引擎代码同 2026-09-28。第 1–4 组实测用的是它。
  - 运行中的引擎在 `/version` 返回的是编译进去的包版本。1001 的 wheel 包版本是 `1.5.1+heavily.modified.v1.1001.torch2.10.cu128`，`/version` 返回 `1.5.1+heavily.modified.v1.1001.torch2.10`（构建时自动加的 `.cu128` 只在 wheel 元数据里）；0930 的构建是 `1.5.1+heavily.modified.v1`。

启动参数（所有测量、两条线都用这一组）：

```bash
vllm serve <模型目录> \
  --tensor-parallel-size 4 --dtype half --attention-backend FLASH_ATTN_V100 \
  --max-model-len 131072 --max-num-seqs 24 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype auto --trust-remote-code \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only
```

本分支这条线的环境变量如下，`SX_OPT_*` 开关全部保持默认（开）：

| 环境变量 | 值 | 作用 |
|---|---|---|
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | `12` | 每个 TP 进程为 PLE 表锁定的主机内存（GiB），4 个进程合计约 48 GiB |
| `OMP_NUM_THREADS` | `8` | 每个进程的 OpenMP 线程数 |
| `VLLM_SM70_QWEN38_HYBRID_PLE` | `0` | 关闭 hybrid PLE 通道 |
| `VLLM_PLE_CPU_OFFLOAD` | `0` | 不把 PLE 查表交给单独的 CPU 进程 |
| `VLLM_PLE_DISK_OFFLOAD` | `0` | 不用磁盘文件映射存放 PLE 表 |
| `VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` | `1` | 打开 NVFP4 MoE 的分组 decode 内核，改动第 3 项的分组 decode 依赖它 |
| `VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS` | `240` | 预热时 MoE 内核调优覆盖的上限，240 对应 24 行（每个 token 选 10 个专家） |

### KV 缓存的格式（FP16 和 E4M3）

怎样才生效、有哪些已知问题，见前面的[提示](#提示e4m3-kv官方-pr-664默认不生效)。

Flash-Next 的注意力（QSA）只支持下面两种 KV 缓存格式，没有介于两者之间的位宽。默认是 FP16，不加任何开关就是它。

| | FP16（默认） | E4M3（可选） |
|---|---|---|
| 每个数占用 | 16 位 | 8 位：1 位符号、4 位指数、3 位尾数，最大能表示 448 |
| 有效数字 | 约 3 位 | 约 1 位 |
| scale | 不需要 | 每层 K、V 各一个，缓存里存的是“数值 ÷ scale”，要在自己的模型上标定 |
| KV 缓存容量（开 MTP，实测第 5 组） | 131,072 token | 216,820 token |
| 本分支的快速通道 | 开不开 MTP 都有 | 只有开 MTP 并设置 `VLLM_QWEN4EXP_QSA_E4M3_MTP=1` 时有 |

- E4M3 只压缩主 K/V，QSA 的索引缓存仍是 FP16，所以容量是多 65%，不是翻倍。
- 不开 MTP 时也可以用 `--kv-cache-dtype fp8_e4m3` 启动（需要 24 个目标层 scale），但本分支的通道检查在这个组合下只认 FP16，各项优化会回到官方路径。这个组合没有测过。
- 官方只为它自己发布的模型提供 scale。后训练过或重新转换过的模型（例如 Swift 1.5）要自己标定。

启用 E4M3 KV 的步骤（我们跑的全流程脚本是 [`sx_bench/as_run/e4m3_chain.sh`](sx_bench/as_run/e4m3_chain.sh)，工具在 `tools/qwen4_exp/`）：

1. 启动一个标定实例：FP16 KV，开 MTP，加 `--enforce-eager`，设置 `VLLM_QSA_KV_CALIBRATION_DIR=<目录>`。eager 模式下启动时的显存测算占用更多，留给 KV 缓存的更少，我们用的是 `--max-model-len 40960 --max-num-batched-tokens 2048`。
2. 实例就绪后，在该目录下建一个名为 `COLLECTING` 的文件，然后发送有代表性的请求。引擎把每个 QSA 层写进缓存的 K、V 的最大绝对值记到这个目录。
3. 停掉标定实例，依次运行 `qsa_kv_calibration.py summarize --expected-layers 13`、`sx_scale_pack.py target-report`、`qsa_kv_calibration.py overlay`、`sx_scale_pack.py manifest`、`materialize_qsa_scale_overlay.py`。得到一个新的模型目录：原模型文件的软链接，加两个 scale 文件和合并后的索引，原模型目录不变。scale = 最大绝对值 ÷ 448。
4. 把 scale 文件 `model-kvscales.safetensors` 的权限改成引擎用户可读（工具写出来是 0600）。
5. 用新目录启动：`--kv-cache-dtype fp8_e4m3`，`VLLM_QWEN4EXP_QSA_E4M3_MTP=1`，建议加 `VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1`（缺 scale 时直接报错）。

### 已知限制

- **MTP 通道**：生产环境请保持关闭。开 MTP 时 KV 缓存只有约 131K token（FP16）或 217K token（E4M3），每条在跑的请求还固定占约 13% 的缓存池，4 条 32K 并发放不下；每张卡显存峰值 32,019–32,267 MiB，整卡 32,768 MiB；E4M3 KV 下 4 并发的 decode 速度减半。数据见实测第 5 组。
- **只有 1 个 token 的 prompt**：全新请求的整个 prompt 只有 1 个 token 时，会读到没有清零的状态槽。官方也有同样的问题。聊天接口的 prompt 带模板，不会触发。
- **开着前缀缓存时长 prompt 的 KV 占用偏高（版本 0930 的问题，版本 1001 已修，见改动第 7 项）**：版本 0930 在 4 条 64K 并发时 KV 占用峰值是 86%，关掉前缀缓存是 64%；版本 1001 实测 64.4%。以下是原因。原因是一步 prefill 跨多个状态块时，换下来的状态块要到请求结束才释放，每个状态组多占“步数 − 2”个块（Flash-Next 有 4 个状态组）。一条冷的 110K 请求因此多占约 9% 的缓存池，这是用本分支的调度器和 KV 管理器模拟得到的估算值，不是实测。这只影响容量，不影响输出；长 prompt 并发多时，缓存池会更早用满，出现排队或抢占。版本 1001 已包含修复。
- **别的请求 prefill 时 decode 会停顿**：相邻两个 token 的间隔最长到 1.2–1.5 秒，数据见实测第 1 组。
- **开着前缀缓存时第 1 条请求的首字晚约 1 秒**：prompt 末尾要多跑一步，用来留下缓存检查点。
- **测试**：`sx_tests/` 下的测试需要 V100 和对应的镜像，每个文件里写了运行方法。
