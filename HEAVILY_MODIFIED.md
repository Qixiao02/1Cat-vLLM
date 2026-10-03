# 1cat-vllm-heavily-modified-v1-1001

Heavily modified fork of [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) for serving
**Swift 1.5 Qwen3.8-Flash-Next (NVFP4)** on **4× V100-SXM2-32GB (TP4, SM70)**. It focuses on
concurrency throughput and prefill. Every change sits behind an `SX_OPT_*` environment switch.
All switches default to on, and setting one to `0` restores the upstream code path.

本分支基于 1Cat 官方代码修改，目标是在 4 张 V100（TP4）上跑 Swift 1.5 Qwen3.8-Flash-Next（NVFP4），重点优化并发吞吐和 prefill。每项改动都有 `SX_OPT_*` 环境变量开关，默认开，设为 `0` 就回到官方原路径。

## 版本

| 项 | 值 |
|---|---|
| 版本名 | 1cat-vllm-heavily-modified-v1-1001 |
| git tag | `1cat-vllm-heavily-modified-v1-1001`（最新，改动第 1–7 项）；上一个版本 `1cat-vllm-heavily-modified-v1-0930`（改动第 1–5 项） |
| 默认分支 | `1cat-vllm-heavily-modified-v1` |
| Python 包版本（PEP 440） | 1001 的 wheel：`1.5.1+heavily.modified.v1.1001.torch2.10.cu128`，运行中的引擎在 `/version` 返回 `1.5.1+heavily.modified.v1.1001.torch2.10`。0930：`1.5.1+heavily.modified.v1`。从源码构建时用 `SETUPTOOLS_SCM_PRETEND_VERSION` 设置，见 `sx_bench/as_run/build_fork.sh` |
| 官方基线 | `main@02c87ab89`（2026-09-14），即官方 v1.5.0 之后第 670 个提交 |

## 改动（按提交顺序）

1. **Flash-V100 分组验证和 DFlash2 speculator**（提交 `d3400c869`）：Flash-V100 opt27-port v2 分组验证，以及配套的 DFlash2 speculator 改动。官方 `main@02c87ab89` 加上这一项，是后面各项改动和收益对比的起点。
2. **第一批（Python/Triton）**
   - 采样器 top-k/top-p 去掉每步的同步
   - mamba align 模式的 prefill 可以跨多块分块
   - QSA：去掉 host 同步，双 warp 覆盖到 32 行，地址解析，混合步路由
   - 稠密层多行 GEMV/HC（单请求路径逐位不变）
   - 路由和持久缓冲扩到 32
   - PLE prefill 打包
3. **第二批（CUDA + CUDA graph）**
   - NVFP4 分组 decode 扩到 M17–M32，并跳过补齐行
   - TP4 push all-reduce 覆盖 10–160 KiB
   - 共享专家门控支持多行
   - QSA decode top-k 覆盖所有行
   - 25–1024 token 的混合步使用 PIECEWISE CUDA graph
4. **第三批 3a**
   - MTP 通道优化，默认不启用，未达到生产条件
   - prefill 保护：1 token（不开 MTP）或 k+1 token（开 MTP）的 prefill 块不再重放 FULL 图
   - PF4：短 prompt 不再为尾部检查点多拆一步
5. **移植官方 #704 质量修复，并让它真正生效**
   - Qwen3.8 的 gated RMSNorm 改用精确的原生 CUDA 算子（`_C::sm70_rmsnorm_gated_exact_out`），结果与 PyTorch eager 的 FP32 计算逐位相同
   - 目的：单请求和批量、decode 和混合步都用同一套算术，避免微小的舍入差异改变 MoE 路由、翻转 EOS
   - 不开 MTP 时默认启用（`VLLM_SM70_RMSNORM_GATED_EXACT`，设为 `0` 关闭）
   - **与官方的区别**：官方在 Python 里只放行 1–192 行。vLLM 每个编译范围只追踪一次，按最大尺寸追踪，而且丢弃形状守卫，所以这个行数条件对整个范围只判断一次。Flash-Next 的 decode 图按 24 并发（288 行）追踪，主编译按 8192 token 追踪，官方条件在我们的部署上从来不满足，算子一次也没运行（官方 `main@357d07bcb` 同样如此）。本分支只检查编译期不变的条件（2 维、宽 128、FP16、连续），内核接受任意行数，所以 decode、混合和 prefill 图的每一行结果都相同
6. **MTP 通道的显存（2026-09-30，版本 1001 新增）**
   - 移植官方 PR #707：PLE 短卷积 prefill 的缓冲从 6 块减到 2 块，结果逐位不变；开 MTP 时按请求长度分组打包
   - 移植官方 PR #664：开 MTP 时 KV 缓存可以用 E4M3（8 位）存放。默认不启用，KV 仍是 FP16；启用需要 `--kv-cache-dtype fp8_e4m3`、`VLLM_QWEN4EXP_QSA_E4M3_MTP=1` 和标定出来的 26 个 scale
   - 怎样才生效、已知问题和实测见 [README.md](README.md) 的“KV 缓存的格式”一节和实测第 5 组
7. **开着前缀缓存时及时释放换下来的状态块（2026-10-01，版本 1001 新增）**：修掉“开前缀缓存时长 prompt 的 KV 占用偏高”，只改 KV 管理器，见 README.md 改动第 7 项

## 实测

对官方最新 `main@d30469863`，双方各用自己的最佳参数（2026-10-01，4 张 V100，同一个模型、同一批 prompt）。完整数据、配置和测法见 [README.md](README.md) 的实测第 1 组。

| 场景 | 本分支 1001 | 官方 | 结果 |
|---|---|---|---|
| prefill：4 条长 prompt 同时到达，每条 8K / 64K（token/s） | 6,578 / 5,796 | 5,587 / 3,669 | 本分支快 1.18 / 1.58 倍 |
| decode 每路，默认采样，4 并发（token/s） | 57–62 | 47–50 | 本分支快约 20% |
| decode 每路，贪心，1 / 2 / 4 / 8 / 16 并发 | 97.9 / 75.8 / 62.9 / 49.7 / 41.8 | 98.2 / 75.4 / 68.0 / 55.2 / 39.9 | 1–2 并发持平，4–8 并发官方快 8–11%，16 并发本分支快 5% |
| 24 并发 decode 合计 | 854 token/s | 官方最多 16 路 | 只有本分支能跑 |
| KV 缓存容量 | 410,247 token | 307,602 token | 本分支多 33% |
| 开 MTP，4 并发，每路：写代码 / 聊天 | 120 / 77 | 81 / 56 | 本分支快 36–45% |
| 开 MTP，单请求：写代码 / 聊天 | 169 / 103 | 191 / 108 | 官方快 4–13% |

2026-09-30 版里“decode 每路快 1.41–1.56 倍”的说法是给官方用了非最佳参数（开前缀缓存、最多 24 路、显存利用率 0.90、`OMP_NUM_THREADS=8`）测出来的，已作废。那次和 2026-09-28 的原始表格保留在 README.md 的“历史记录”里。

### 分支内部对比

精确 gated RMSNorm 生效前后（第 5 项）：吞吐、首字和长文首字都没有可测的变化。同一组请求单独跑和 8 并发跑，逐 token 相同的是 21/24，和生效前一样。这个指标同一份代码两次测量在 17–21 之间波动，24 条样本分辨不出差异。

相对起点（官方 `main@02c87ab89` 加第 1 项），本分支的主要收益：

- prefill 快 1.4–1.7 倍
- 采样 decode 每步快 18–37%
- 业务 JSON 24 并发：1.84 → 2.89 请求/秒

## 实验开关（默认关）

上面各项开关默认开，下面这个默认关，不在改动列表里。

- **`SX_OPT_COMPILE_CACHE`**（默认 `0`，行为和之前逐字节相同）：重启时复用 torch.compile 的缓存，冷启动里约 185 秒的编译预计能省下 100–170 秒（预期值，还没有在 V100 上测过）。`1` 复用编译好的子图，`aot` 复用整个 AOT 产物。打开后缓存键会带上构建指纹（torch/CUDA 版本、源码内容、原生库、检查点文件），换了镜像或改了任何 `VLLM_*`/`SX_OPT_*` 开关都不会命中旧缓存。官方 1.5.1 默认打开缓存，但它自己的 27B 测试里 AOT 重载的输出和冷编译不一致，所以这里在 `sx_tests/compile-cache/cache_parity.sh` 通过之前不要在生产打开。分析、移植了官方哪些提交、风险见 [docs/design/sx_compile_cache.md](docs/design/sx_compile_cache.md)。

## 已知限制

- **MTP 通道**：请在生产上保持关闭。KV 只有约 131K token（FP16）或 217K token（E4M3），每条在跑的请求固定占约 13% 的缓存池；E4M3 KV 下 4 并发的 decode 速度减半。数据见 README.md 实测第 3 组。
- **1 token prompt**：全新请求的 prompt 只有 1 个 token 时，状态槽没有清零。官方也有同样问题。聊天接口的 prompt 带模板，不会触发。
- **开前缀缓存时长 prompt 的 KV 占用偏高**：4 条 64K prompt 并发时，KV 占用峰值是 86%，关前缀缓存是 64%。原因是 prefill 一步跨多个状态块时，换下来的状态块要到请求结束才释放，每个状态组多占“步数 − 2”个块（Flash-Next 有 4 个状态组）。一条冷的 110K 请求因此多占约 9% 的缓存池（用本分支的调度器和 KV 管理器模拟得到的估算值，不是实测）。只影响容量，不影响输出；长 prompt 并发多时，缓存池会更早用满，出现排队或抢占。修复已有原型，还没有合入这个版本。
- **别的请求 prefill 时 decode 会停顿**：一条请求在 prefill 时，已经在生成的请求会停顿。4 并发冷 prompt 的测量里，相邻两个 token 的最长间隔是 1.2–1.5 秒，官方 `main@d30469863` 是 0.5–0.8 秒，见“和官方 `main@d30469863` 并排对比”。
- **测试环境**：`sx_tests/` 下的测试需要 V100 和对应镜像，每个文件里写了运行方法。
- **压测脚本和原始结果**：在 `sx_bench/`，说明见 `sx_bench/README.md`。目前收录 2026-09-30 的三组测量（和官方并排对比、前缀缓存开和关、开 MTP 时 FP16 KV 和 E4M3 KV）。
