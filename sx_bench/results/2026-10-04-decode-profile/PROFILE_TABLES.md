# 1003 单步 decode 的时间花在哪（2026-10-04 剖析，4×V100，TP4，生产参数，不开 MTP，FP16 KV）

来源：`prof_1003.sh` 在测试 GPU 4–7 上起两个实例，每个窗口只在所有请求都出了第一个 token 之后抓 3 秒 decode，所以每个 trace 是 100–200 个 graph 步（PROF_G）或 13–15 个 eager 步（PROF_E），不是整条请求。原始分析文件在本目录 `raw/`。**下面所有“每步”数字都是窗口总量除以该窗口的 graph 启动次数**（`analyze_trace.py` 的标题写的是 per step，实际是窗口总量）。profiler 开着，所以 span 比不开时略长；按 span 折算的吞吐和 README 的实测同量级（8K×8 每路 47 对 50.1 token/s；2K×24 合计 831 对 854；单路 90 对 99.1）。完整的 trace 文件（每个窗口约 35 MB × 4 张卡）没有收进仓库，只收分析输出。

## 1. CUDA graph 模式：一步多长，kernel 数，间隙

| 窗口（上下文 × 并发） | graph 步数 | 一步 span (ms) | GPU 忙 (ms) | kernel 间隙 (ms) | 间隙占比 | 每步 kernel 数 | 折合合计 token/s |
|---|---|---|---|---|---|---|---|
| 8K x 1 | 156 | 11.06 | 9.66 | 1.40 | 13% | 1465 | 90 |
| 8K x 4 | 179 | 16.90 | 15.30 | 1.61 | 10% | 1724 | 237 |
| 8K x 8 | 143 | 21.22 | 19.39 | 1.83 | 9% | 1868 | 377 |
| 4K x 16 | 121 | 25.09 | 22.93 | 2.16 | 9% | 2036 | 638 |
| 2K x 24 | 105 | 28.89 | 26.93 | 1.96 | 7% | 1987 | 831 |
| 32K x 1 | 157 | 11.62 | 10.07 | 1.55 | 13% | 1465 | 86 |

“折合合计 token/s”= 并发 ÷ 一步 span，是推算值，不是单独测的。

## 2. 每步 GPU 忙的时间按 kernel 家族分（ms / 步）

| 窗口 | 稠密 GEMM | HC 专用内核 | GDN | MoE | QSA 注意力 | router / top-k | all-reduce | 拷贝 / 逐元素 | norm | 其他 | PLE | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8K x 1 | 2.98 | 2.15 | 1.44 | 1.15 | 0.94 | 0.54 | 0.62 | 0.33 | 0.12 | 0.63 | 0.03 | 10.93 |
| 8K x 4 | 6.79 | 0.97 | 1.87 | 2.47 | 1.39 | 0.70 | 1.02 | 0.55 | 0.12 | 0.75 | 0.20 | 16.82 |
| 8K x 8 | 7.88 | 0.96 | 2.66 | 3.40 | 2.14 | 0.73 | 1.11 | 0.72 | 0.13 | 0.84 | 0.28 | 20.87 |
| 4K x 16 | 9.63 | 1.02 | 1.59 | 4.55 | 3.02 | 0.70 | 1.53 | 0.83 | 0.13 | 0.84 | 0.31 | 24.16 |
| 2K x 24 | 10.50 | 1.08 | 2.23 | 5.48 | 3.88 | 0.68 | 2.18 | 0.92 | 0.14 | 0.88 | 0.34 | 28.31 |
| 32K x 1 | 2.99 | 2.10 | 1.44 | 1.15 | 0.98 | 0.84 | 0.74 | 0.33 | 0.12 | 0.63 | 0.03 | 11.35 |

家族合计比“GPU 忙”略大，是因为不同 stream 上的 kernel 在时间上有重叠。注意：1 并发时 HC 走专用的 FP16 融合内核（算在“HC 专用内核”里）；4 并发起 HC 的两个大矩阵乘落到通用 cutlass GEMM（算在“稠密 GEMM”里），所以 HC 列掉下来、GEMM 列涨上去；同样，GDN 输入投影和小批量 GEMV 在 8 并发以内是自写内核（GDN 列、GEMM 列里的自写部分），16 并发起也落到通用 cutlass GEMM，所以各列不能逐行直接比。看 GEMM + HC 的总成本要把两列加起来：

| 窗口 | GEMM + HC (ms/步) | 占 GPU 忙 |
|---|---|---|
| 8K x 1 | 5.13 | 53% |
| 8K x 4 | 7.76 | 51% |
| 8K x 8 | 8.84 | 46% |
| 4K x 16 | 10.65 | 46% |
| 2K x 24 | 11.57 | 43% |
| 32K x 1 | 5.08 | 50% |

## 3. 每步最耗时的 kernel（8K×8 和 2K×24，ms/步，调用次数/步）

**8K × 8**

| ms/步 | 调用/步 | kernel |
|---|---|---|
| 3.48 | 196 | `void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_16x16_64x2_tn` |
| 2.18 | 48 | `void (anonymous namespace)::w13_kernel<4, true>(__half const*, unsigned int cons` |
| 1.81 | 36 | `_qwen38_fp16_gdn_input_rows_kernel` |
| 1.65 | 72 | `_qwen38_fp16_rows_gemv_kernel` |
| 1.20 | 48 | `(anonymous namespace)::w2_kernel(__half const*, unsigned int const*, __half cons` |
| 1.14 | 12 | `_qsa_sparse_paged_gqa_splitk_kernel` |
| 1.06 | 49 | `void cutlass::Kernel2<cutlass_70_wmma_tensorop_s161616gemm_f16_32x32_64x2_tn_ali` |
| 1.06 | 96 | `void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn` |
| 0.72 | 36 | `void (anonymous namespace)::gdn_decode_mixed_qkv_global_state_kernel<c10::Half, ` |
| 0.64 | 12 | `_qsa_mqa_paged_kernel` |

**2K × 24**

| ms/步 | 调用/步 | kernel |
|---|---|---|
| 7.33 | 340 | `void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_32x32_64x2_tn` |
| 3.58 | 48 | `void (anonymous namespace)::w13_kernel<8, true>(__half const*, unsigned int cons` |
| 2.69 | 12 | `_qsa_sparse_paged_gqa_splitk_kernel` |
| 1.98 | 36 | `void (anonymous namespace)::gdn_decode_mixed_qkv_global_state_kernel<c10::Half, ` |
| 1.88 | 48 | `(anonymous namespace)::w2_kernel(__half const*, unsigned int const*, __half cons` |
| 1.85 | 97 | `volta_fp16_s884gemm_fp16_128x64_ldg8_f2f_tn` |
| 1.11 | 50 | `void vllm::sm70_cross_device_reduce_1stage_push<4>(vllm::sm70_tile_runtime::Rank` |
| 0.92 | 48 | `void vllm::sm70_cross_device_reduce_sum2_1stage_push<4>(vllm::sm70_tile_runtime:` |
| 0.83 | 12 | `_qsa_mqa_paged_kernel` |
| 0.78 | 241 | `void cublasLt::splitKreduce_kernel<32, 16, int, __half, __half, float, __half, f` |

## 4. 稠密 GEMM 的带宽利用（eager + record_shapes，8 并发，200 token 提示）

eager 模式下 all-reduce 的 kernel 时间是在等最慢的卡，不代表真实通信时间，所以只看计算 kernel。下表“GB/s”= 权重字节数 ÷ kernel 时间（FP16 估算，只对稠密权重有意义）；V100-SXM2 的 HBM2 峰值约 900 GB/s。

| 形状（输入 × 权重） | 调用/步 | µs/次 | ms/步 | GB/s |
|---|---|---|---|---|
| `[[8, 10240], [10240, 336]]` | 96.0 | 16.2 | 1.554 | 425 |
| `[[8, 2560], [2560, 4096]]` | 36.0 | 38.4 | 1.382 | 546 |
| `[[8, 320], [320, 10240]]` | 97.0 | 13.1 | 1.272 | 500 |
| `[[8, 1536], [1536, 2560]]` | 48.0 | 18.4 | 0.885 | 427 |
| `[[8, 2560], [2560, 512]]` | 48.0 | 9.8 | 0.471 | 267 |
| `[[8, 2560], [2560, 62080]]` | 1.0 | 459.9 | 0.460 | 691 |
| `[[8, 2560], [2560, 320]]` | 48.0 | 9.1 | 0.435 | 181 |
| `[[8, 2560], [2560, 3584]]` | 12.0 | 35.1 | 0.421 | 523 |
| `[[8, 2560], [2560, 24]]` | 36.0 | 10.8 | 0.390 | 11 |

