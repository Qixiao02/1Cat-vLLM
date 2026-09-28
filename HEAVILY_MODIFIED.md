# 1Cat-vLLM 1.5.1-heavily-modified-v1

Heavily modified fork of [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) for serving
**Swift 1.5 Qwen3.8-Flash-Next (NVFP4)** on **4× V100-SXM2-32GB (TP4, SM70)**. It focuses on
concurrency throughput and prefill. Every change sits behind an `SX_OPT_*` environment switch.
All switches default to on, and setting one to `0` restores the upstream code path.

基于 1Cat 官方代码的魔改分支，目标是在 4 张 V100（TP4）上跑 Swift 1.5 Qwen3.8-Flash-Next（NVFP4），重点优化并发吞吐和 prefill。每项改动都有 `SX_OPT_*` 环境变量开关，默认开，设为 `0` 就回到官方原路径。

## 版本

| 项 | 值 |
|---|---|
| 版本名 | 1.5.1-heavily-modified-v1 |
| git tag | `v1.5.1-heavily-modified-v1` |
| Python 包版本（PEP 440） | `1.5.1+heavily.modified.v1`。从源码构建时设置 `SETUPTOOLS_SCM_PRETEND_VERSION=1.5.1+heavily.modified.v1` |
| Docker 镜像 tag | `1.5.1-heavily-modified-v1-sm70main` |
| 官方基线 | `main@02c87ab89`（2026-09-14），即官方 v1.5.0 之后第 670 个提交 |

## 改动（按提交顺序）

1. **1.7.2 镜像覆盖**：Flash-V100 opt27-port v2 分组验证和 DFlash2 speculator。与内部 1.7.2 镜像一致。
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
5. **移植官方 #704 质量修复**
   - Qwen3.8 的 gated RMSNorm 改用精确的原生 CUDA 算子（`_C::sm70_rmsnorm_gated_exact_out`）
   - 保证单请求和批量的编译图计算结果一致
   - 不开 MTP 时默认启用（`VLLM_SM70_RMSNORM_GATED_EXACT`）

## 实测：Flash-Next，4× V100，不开 MTP

对比对象是官方最新 `main@357d07bcb`（2026-09-28，干净构建，全部原生库和 FA2 都重编过）。两边的配置、压测和真实请求回放完全相同。

| 指标 | 本分支 | 官方最新 |
|---|---|---|
| 每路 token/s：C1 / C4 / C8 / C24 | 93 / 62 / 47 / 23 | 81 / 17 / 13 / 11 |
| 24 并发总吞吐 | 400 token/s | 210 token/s |
| 业务 JSON 8 / 24 并发 | 1.46 / 2.61 请求/秒 | 0.69 / 1.74 请求/秒 |
| 真实请求 1.5 请求/秒：首字 p95 / 端到端 p95 | 0.67 s / 11.4 s | 33 s / 78 s |
| 长文首字 32K / 64K / 110K | 6.1 / 11.6 / 19.8 s | 53 / 33 / 90 s |
| 长文检索（9 处） | 9/9 | 9/9 |

相对官方 1.7.2 镜像，本分支的主要收益：

- prefill 快 1.4–1.7 倍
- 采样 decode 每步快 18–37%
- 业务 JSON 24 并发：1.84 → 2.89 请求/秒

## 已知限制

- **MTP 通道**：KV 只有约 131K token，负载下还可能显存溢出。请在生产上保持关闭。
- **1 token prompt**：全新请求的 prompt 只有 1 个 token 时，状态槽没有清零。官方也有同样问题。聊天接口的 prompt 带模板，不会触发。
- **测试环境**：`sx_tests/` 下的测试需要 V100 和对应镜像，每个文件里写了运行方法。
