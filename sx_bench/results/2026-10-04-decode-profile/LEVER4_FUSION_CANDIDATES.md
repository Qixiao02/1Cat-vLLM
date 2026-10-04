# LEVER 4 — which small kernels are worth fusing (feasibility, read-only analysis)

Sources: `raw\G_8000_c8.txt` (8x), `raw\G_2000_c24.txt` (24x), `raw\E_8000_c8.txt` (8x eager, 13 decode steps, has
input dims), `raw\E_200_c24.txt` (24x eager, 13 steps), `PROFILE_TABLES.md`, `LEVERS.md`, the fork source under
`D:\AI\wt-int\vllm\models\qwen4_exp\nvidia\` and `D:\AI\wt-int\csrc\`. Scripts I wrote and ran (both under
`analysis\`): `lever4_kernels.py` (per-window kernel census), `lever4_side_by_side.py` (Table A),
`lever4_savings.py` (every arithmetic line in sections C and E). Every number below is a read value or a script
output, except where marked **estimate**.

## 0. The one measured constant everything hangs on

All raw PROF_G numbers are window totals; per step = value ÷ graph launches (`PROFILE_TABLES.md:3`).
Intra-graph gap (span − GPU busy) and gap per launch, from all six windows (`lever4_savings.py`):

| window | graph launches | span/step ms | GPU busy/step | kernels/step | gap/step ms | **gap per launch** |
|---|---|---|---|---|---|---|
| 8K x 1 | 156 | 11.06 | 9.66 | 1465 | 1.40 | 0.956 us |
| 8K x 4 | 179 | 16.90 | 15.30 | 1724 | 1.60 | 0.928 us |
| **8K x 8** | **143** | **21.22** | **19.39** | **1868** | **1.83** | **0.980 us** |
| 4K x 16 | 121 | 25.09 | 22.93 | 2036 | 2.16 | 1.061 us |
| **2K x 24** | **105** | **28.89** | **26.93** | **1987** | **1.96** | **0.986 us** |
| 32K x 1 | 157 | 11.62 | 10.07 | 1465 | 1.55 | 1.058 us |

So **one launch ≈ 1.0 us of step time** (0.93–1.06 us across all six windows), and it is flat: the gap is not a few
big stalls but ~1 us attached to each of ~1,900 launches. The raw files' `gaps > 0.1 ms` lists confirm this — at 4Kx16
only 31.9 ms of 121 steps (0.26 ms/step) is in gaps > 0.1 ms, and those are CPU graph-boundary stalls
(`cudaGraphLaunch`, `aten::new_full`), not the 2.16 ms/step intra-graph gap. **Consequence: a fusion is worth
(launches removed) × 1.0 us, plus the time of any kernel that disappears entirely.** Removing 100 launches = 0.1 ms =
0.47% of an 8x step. Anything under ~50 launches/step is not worth reviewing.

## A. Ranked small-kernel table at both profile points

`ms/step`, `calls/step`, `us/call` and `%of step` = window value ÷ graph launches (143 at 8x, 105 at 24x), computed by
`lever4_side_by_side.py`. "-" = not in that window's top-40 list. The top-40 list covers 205777/267111 calls
(1439/1868 = 77%) and 19.34 of 19.39 ms/step at 8x; 167475/208610 (1595/1987 = 80%) and 26.71 of 26.93 at 24x — i.e.
**the unlisted ~22% of launches are 1–3-call/step kernels worth ~2% of the time**, so nothing big hides below the cut.

| # | kernel | 8x calls | 8x ms | 8x us/call | 8x %step | 24x calls | 24x ms | 24x us/call | 24x %step |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `cublasLt::splitKreduce_kernel<32,16,int,__half,__half,float,__half,...>` | 145 | 0.441 | 3.04 | 2.08 | 241 | 0.781 | 3.24 | 2.70 |
| 2 | `_hc_silu_kernel` (Triton) | 97 | 0.205 | 2.11 | 0.97 | 97 | 0.201 | 2.07 | 0.70 |
| 3 | `_hc_gate_mix_kernel` (Triton) | 97 | 0.310 | 3.20 | 1.46 | 97 | 0.357 | 3.68 | 1.24 |
| 4 | `_hc_combine_norm_kernel` (Triton) | 95 | 0.442 | 4.66 | 2.08 | 95 | 0.515 | 5.42 | 1.78 |
| 5 | `plan_v2_kernel` | 48 | 0.242 | 5.05 | 1.14 | 48 | 0.267 | 5.57 | 0.93 |
| 6 | `qwen38_shared_gate_exact_kernel` | 48 | 0.196 | 4.09 | 0.92 | 48 | 0.202 | 4.20 | 0.70 |
| 7 | `act_and_mul_kernel<c10::Half,__half2,&vllm::silu_kernel>` | 48 | 0.192 | 4.00 | 0.91 | 48 | 0.191 | 3.99 | 0.66 |
| 8 | `cublasLt::splitKreduce_kernel<32,16,int,float,__half,float,__half,...>` | 49 | 0.190 | 3.88 | 0.90 | - | - | - | - |
| 9 | `reduce_v2_kernel` (NVFP4 w2 second pass) | - | - | - | - | 48 | 0.156 | 3.24 | 0.54 |
| 10 | `_causal_conv1d_update_kernel` | 36 | 0.137 | 3.81 | 0.65 | 36 | 0.156 | 4.33 | 0.54 |
| 11 | `rmsnorm_gated_exact_kernel<false>` | - | - | - | - | 36 | 0.136 | 3.79 | 0.47 |
| 12 | `_qsa_merge_splitk_kernel` | 12 | 0.146 | 12.19 | 0.69 | 12 | 0.141 | 11.73 | 0.49 |
| 13 | `qsa::qsa_lexicographic_decode_topk_rows_kernel<512>` | 12 | 0.153 | 12.77 | 0.72 | - | - | - | - |
| 14 | `_qsa_pre_indexer_kernel` (eager 8x row) | 12 | 0.072 | 6.0 | 0.34 | 12 | 0.069 | 5.7 | 0.24 |
| 15 | `sm70_cross_device_reduce_1stage_push<4>` (all-reduce) | 50 | 0.566 | 11.31 | 2.67 | 50 | 1.106 | 22.12 | 3.83 |
| 16 | `sm70_cross_device_reduce_sum2_1stage_push<4>` | 48 | 0.452 | 9.41 | 2.13 | 48 | 0.915 | 19.06 | 3.17 |
| 17 | eager-only `aten::copy_ [[256],[256]]` / `aten::arange [[0]]` / `aten::add [[8,2560],[8,2560]]` | 24/78/48 | 0.105/0.101/0.091 | 4.4/1.3/1.9 | 1.4 (all three) | 24/78/48 | 0.105/0.102/0.097 | 4.4/1.3/2.0 | 1.0 |

Rows 2, 3, 5 are the *many-launch / cheap-each* kernels the gap is attached to: 97, 97 and 48 launches per step whose
per-call cost is 2–6 us, i.e. within 6x of the pure 1.0 us launch gap. They are not compute; they are geometry.
Rows 1 and 8 are the largest *count* in the whole profile (145 and 241 launches/step) and are pure second-pass
reductions of a cuBLAS split-K GEMM.
Rows 15/16 are the biggest small-looking cost but are communication kernels whose eager time is inflated by waiting for
the slowest rank (`PROFILE_TABLES.md` note on `cross_device_reduce_1stage<__half,4>` = 212 ms/step eager at 8x = 91.3%
of kernel time); see section E for why I did not propose touching them.
The big kernels (16x16 cutlass 3.480 ms/196 calls = 16.4% at 8x; 32x32 cutlass 7.327/340 = 25.4% at 24x; `w13_kernel`
2.183/3.581; `_qwen38_fp16_rows_gemv_kernel` 1.648/72; `_qwen38_fp16_gdn_input_rows_kernel` 1.811/36;
`_qsa_sparse_paged_gqa_splitk_kernel` 1.144/2.686; `w2_kernel` 1.201/1.878) are compute- or geometry-bound at 11–224
us/call and are *not* fusion targets — merging them into each other would change their reduction order (section D).

## B. Candidate launch sites

The 8x/24x HC chain is the `F.linear` fallback, not the fused Triton route: the fused multi-row route is refused
because `SX_OPT_ROWS_TABLE` caps the `hc` role at M=2.

**C1/C2 — HC pair chain.** Launch site `vllm\models\qwen4_exp\nvidia\sm70_fp16_hc.py:1134-1155` (function around
`:1115`), the `if not _runtime_ok(x, down_weight, up_weight)` branch:
`:1148 down_and_injection = torch.nn.functional.linear(x, down_weight)`,
`:1149-1151 lora = torch.ops.vllm.qwen4_exp_hc_silu(down_and_injection[..., :320], 4)`,
`:1152 injection = down_and_injection[..., 320:324]`,
`:1153 gate = torch.nn.functional.linear(lora, up_weight)`,
`:1154 block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, 4)`.
Reads/writes: `x` (M,10240) fp16 → down mm (`(324,10240)` weight, split-K cuBLAS + `splitKreduce`) → `lora` (M,320)
fp16 → `hc_silu` (fp32 `/4`, fp32 silu, fp16 store, grid = 1 program per row — the "97 launches/step" is one launch
per GatedResidual, not a grid) → up mm (`(2560,320)`) → `hc_gate_mix` (fp32 accumulate over the 4 HC streams, `/4`,
fp16 store, grid `(N, 5)` with BLOCK_SIZE 512). Kernel bodies: `ops\hc.py:83` (`_hc_silu_kernel`), `ops\hc.py:114`
(launch), `ops\hc.py:127` + `ops\hc.py:174` (`_hc_gate_mix_kernel` + launch). Module-level order:
`hyperconnection.py:131-159` (`_project`), caller `hyperconnection.py:161-198`. `_HC_RANK=320`, `_HC_COUNT=4`,
`_HC_DIM=2560` (`sm70_fp16_hc.py`). Already fused with a neighbour? No — 4 separate kernels; the combine+norm before
them *is* fused (row 4), and the M==1 route (`sm70_fp16_hc.py:1156-1216`) is a different, TP4-sharded 3-kernel chain.
Counts cross-check: `_hc_silu`/`_hc_gate_mix` 97/step vs `_hc_combine_norm` 95/step = 96 decoder layers x 2 pairs
(48 layers x {attn_hc, mlp_hc}) plus the head mixer built with `use_combine=False`
(`model.py:411`, `:415`, `:615`, `:617`); QSA kernels 12/step and GDN 36/step give the 12+36 = 48-layer split.

**C3 — cuBLAS split-K second passes.** No launch site in the fork: `aten::mm` for the HC down/up projections chooses a
split-K tile and CUDA emits `cublasLt::splitKreduce_kernel` as a second kernel (`raw\E_8000_c8.txt:24`,
`:28`, `:40`, `:42`, `:44`). Ordering in the profile is right after the GEMM of the same shape.

**C4 — `plan_v2_kernel`.** `csrc\sm70_turbomind\ops\nvfp4_grouped_decode_sm70.cu:110-164`, launched at
`:582 plan_v2_kernel<<<1, kPlanV2Threads, 0, stream>>>(...)` immediately before
`:588 w13_kernel<S,I><<<dim3(5, routes), 64*S, 0, stream>>>`. Computes, for the 8x48 = 384 routes: expert id bucket
(`__match_any_sync`/`__popc`), a per-expert exclusive scan into `base[]`, `rows[]/experts[]/sizes[]/total[]`, with
`live_route_limit()` (`:103-108`) masking padded decode rows via the pointer-stable `query_start_loc` tail. Reads
`ids` (routes) + `valid_tokens`; writes 4 int32 workspaces; ~20.5 KB static smem (`:117`). Not fused; one block means
its 5.05/5.57 us is all fixed cost.

**C5 — QSA indexer + lexicographic top-k.** `vllm\models\qwen4_exp\nvidia\ops\qsa_pre_indexer.py:467` launches
`_qsa_pre_indexer_kernel[(num_k_work + num_q_work,)]`; the top-k is a separate op (`qsa_lexicographic_topk_decode_`,
kernel `csrc\qsa_lexicographic_topk.cuh:430`, launcher `:466-471`, bound at `csrc\libtorch_stable\topk.cu:336`). The
attention itself is already fused (`ops\qsa.py:3647` splitk + `:3695` `_qsa_merge_splitk_kernel` writes `out` and the
output gate in one pass).

**C6 — GDN per-layer small kernels.** `_causal_conv1d_update_kernel` and `layer_norm_fwd_kernel`/`rmsnorm_gated_exact_kernel`
run 36/step each, in `vllm\models\qwen4_exp\nvidia\` GDN layer code (they appear in the eager profile with no CPU op or
under `vllm::qwen_gdn_attention_core_stan`, `raw\E_8000_c8.txt:13/:35/:36`), i.e. they are C++-side launches inside the
GDN op rather than separate python modules.

**C7 — MoE shared expert tail.** `vllm\model_executor\models\qwen2_moe.py:333-403`: `:342 out, _ = self.down_proj(out)`,
then the exact gate at `:382 sm70_ops.qwen38_shared_gate_exact_out(out, x, gate_weight)` (admitted because
`SX_OPT_SHARED_GATE_ROWS` defaults on, `qwen2_moe.py:112`, and `_SX_SHARED_GATE_ROWS_MAX_TOKENS = 32`, `:115`, covers
M=8 and M=24). The fallback at `:384-393` (`expert_gate(x)` + `F.sigmoid` + `*`) is *not* what runs here.

## C. Proposals, with the arithmetic

Model: saving = (launches removed) × 1.0 us + (measured time of kernels that disappear). Measured per-step times are
from section A; the arithmetic is in `lever4_savings.py`.

**P1 — fold `hc_silu` into the HC down projection's epilogue** (97 launches). Down mm is split-K today
(96 + 97 launches of `splitKreduce`, 0.292 + 0.295 ms/step at 8x, `lever4_savings.py` [A]).
`97 launches x 1.0 us = 0.097 ms`, plus the kernel's own 0.205 ms (8x) / 0.201 ms (24x) = **0.30 ms/step ≈ 1.42% of the
8x step, 1.04% of the 24x step**. Needs: a down-projection kernel with a silu epilogue — i.e. the already-written
`_qwen38_hc_down_silu_inject_rows_kernel` (`sm70_fp16_hc.py:1028-1078`) or the `fp8_qpn8_hc_down_silu_reduce_sm70_kernel`
pattern (`csrc\sm70_turbomind\ops\fp8_qpn8_sm70.cu:1144`). Extra shared memory 0, extra registers 0, no extra output
(the fp16 `lora` is still materialized because the up mm reads it).

**P2 — fold `hc_gate_mix` into the HC up projection's epilogue** (97 launches). **0.097 ms + 0.310 ms = 0.41 ms/step at
8x (1.92%), 0.097 + 0.357 = 0.45 ms at 24x (1.57%)**. Needs: an up-projection kernel that also reads `x` and writes
the mixed `(M,2560)` block instead of the `(M,2560)` gate — the `_qwen38_hc_up_gate_mix_row4_rows_kernel`
(`sm70_fp16_hc.py:1028-1078`) and the M==1 route's `sm70_qwen38_hc_up_mix_allgather` are exactly this shape. Extra
shared memory: 2 x BLOCK_N fp16 tiles per program, no extra output.

**P1+P2 together: 194 launches, 0.194 ms of gap + 0.515 ms (8x) / 0.558 ms (24x) of kernel time = 0.71 ms/step at 8x
(3.34% of the step) or 0.75 ms/step at 24x (2.60%).** This is the whole HC chain becoming 2 launches per pair instead
of 4 (2 x 97 = 194 > the 95 `_hc_combine_norm` pairs because of the head mixer), and it also removes the two split-K
second passes for those two GEMMs if the fused kernel replaces the cuBLAS call: a further 193 launches, 0.193 ms of gap
= ~0.9% at 8x.

**P3 — kill the split-K second passes without touching the rows kernels.** `splitKreduce` is 145 launches/step at 8x
(0.441 ms, 2.08%) and 241 at 24x (0.781 ms, 2.70%) — the single largest launch count in the profile. Forcing a
K-unsplit tile for the HC down/up shapes (K=10240 and K=320) would remove the pair of launches *and* the 0.292/0.295 ms
second pass at 8x. **Estimate**, not a measurement: eliminating the HC-related subset is 193 launches ≈ 0.193 ms
(0.91% at 8x); the remainder (the 1536x2560 and 2560x24/512/320 projections) would need a cuBLAS heuristic change.
This one is a numeric-contract change, not just a scheduling change — see section D.

**P4 — move `plan_v2_kernel`'s work into the router top-k.** 48 launches: `48 x 1.0 us = 0.048 ms = 0.23% (8x) / 0.17%
(24x)`, plus up to 0.242/0.267 ms if the scan really disappears (a router kernel cannot do the cross-route scan, so
the realistic number is the gap only). It needs a grid-wide barrier (the plan is consumed by `w13_kernel`), i.e. a
persistent/cooperative restructure of the grouped NVFP4 MoE op — high risk for <=0.23%.

**P5 — pair up the 48-launch MoE small kernels.** `qwen38_shared_gate_exact_kernel` (0.196/0.202 ms) + `act_and_mul`
(0.192/0.191) + `down_proj` = the shared expert's tail (`qwen2_moe.py:342-403`) = 144 launches/step. Folding the gate
multiply into `down_proj`'s epilogue removes 48 launches (0.048 ms, 0.23%) plus 0.196 ms if the gate dot also moves —
but the gate it replaces is a *weight* dot (`expert_gate(x)`), so this is only a win if the down projection is a
custom kernel, i.e. it lands in the same bucket as P3. **Estimate.**

**P6 — QSA indexer into the top-k.** 24 launches total (12 + 12) = 0.024 ms of gap + 0.072 + 0.153 = 0.25 ms at 8x
(1.17%); at 24x 0.069 + 0.120 = 0.21 ms (0.72%). Both kernels are small-grid scalar/selection work, so the merge is a
straight append of the top-k's per-row pass to the indexer's last program stage; it needs the indexer's `(num_k_work +
num_q_work,)` grid to carry the top-k's `512`-wide per-row state.

**Cumulative, 8x: P1+P2+P3+P4+P6 = 0.71 + 0.19 + 0.05 + 0.25 = 1.20 ms/step ≈ 5.7% of the 21.22 ms step.** At 24x
P1+P2+P3+P6 = 0.75 + 0.19 + 0.21 ≈ 1.15 ms ≈ 4.0% of the 28.89 ms step. **Estimates for the totals; the per-kernel
terms are measurements.** LEVERS.md's "+4-6% from merging small kernels" is therefore the right order of magnitude
**only if the HC pair chain is included** — the small-kernel list alone (rows 5-14 of Table A) is ~0.5% of a step.

## D. Batch-invariance risk (per proposal)

The hard constraint (`sm70_fp16_gemv.py:9-21`): every output row of the multi-row decode path must be *bitwise* equal
to the unchanged M=1 kernel run on that row alone — "Triton therefore assigns the M=1 layout: the same per-slot FMA
chain over the K chunks, the same in-thread add, xor shuffle tree and cross-warp combine. Every output row is bitwise
equal to the unchanged M=1 kernel run on that row alone (batch invariant)." `SX_OPT_ROWS_TABLE`
(`sm70_fp16_gemv.py:30-36`, default at `:225-233` = `gdn_in=8,gdn_out=4,qsa_qkv=8,qsa_o=4,qsa_index=8,router=8,hc=2`)
is the admission gate, `_sx_rows_tile` refuses when `envs.VLLM_BATCH_INVARIANT` (`:377`), and
`sm70_fp16_hc.py:1005` refuses the rows plan under `VLLM_BATCH_INVARIANT`. `csrc\core\batch_invariant.hpp:9-16` is just
`VLLM_BATCH_INVARIANT` as a cached bool; `vllm\model_executor\layers\batch_invariant.py:982-991`
(`init_batch_invariance`) sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, NCCL single-channel `Simple`/`tree`, and IEEE fp32
matmul; `MTP` routes additionally require `allow_fp16_reduced_precision_reduction=True`
(`sm70_fp16_gemv.py:475-491`).

- **P1/P2 are the safe ones**, and the reason is structural: `hc_silu` and `hc_gate_mix` are *pure per-element*
  functions (no reduction across rows, no reduction over K). Moving `x/HC` + `x*sigmoid(x)` and
  `sum_{stream<4} sigmoid(g)*x` / 4 into an epilogue that runs in fp32 with the same order reproduces the same bits for
  every row independently of how many rows share the launch. The HC `F.linear` accumulations themselves are untouched
  — only the *producer* of `lora` and the *consumer* of `gate` change. Both HC elementwise kernels are already
  single-row (`_hc_silu_kernel` grid = num_tokens) or one block per row group (`_hc_gate_mix_kernel` grid = (N, 5)),
  so there is no cross-row reduction to lose in the first place.
- **P3 does NOT preserve the contract.** The MTP batch routes are documented as reproducing cuBLAS "with reduced
  precision split-K reduction allowed and FP32 accumulation" (`sm70_fp16_gemv.py:475-491`), and the M=1 oracle *uses*
  split-K: `raw\G_8000_c1.txt:35` shows `cublasLt::splitKreduce_kernel<32,16,int,float,__half,...>` at 7644 calls over
  156 launches = 49 launches/step at 1x. Removing split-K changes the fp32 accumulation order (partial sums added in a
  second pass vs one chain), so it is **only safe outside `VLLM_BATCH_INVARIANT`** and must not be taken as the default
  for the graph lane.
- **P4 does not change any floating-point value**: plan_v2 only computes integer expert buckets/prefix sums, and the
  `valid_tokens` masking (`nvfp4_grouped_decode_sm70.cu:98-108`) is unchanged, so the MoE expert arithmetic is
  bit-identical. Numerically safe; the risk is entirely mechanical (it needs a grid barrier before `w13_kernel`).
- **P5 changes the shared-expert reduction order** if the gate weight-dot moves into the down projection's epilogue:
  the gate is a separate K-reduction over `x`, and folding a second reduction into an epilogue changes the fp32 chain.
  Safe only outside batch-invariant mode unless the epilogue reproduces the gate's exact chain.
- **P6 changes a selection order, not arithmetic**: the lexicographic top-k is an integer/ordered comparison on `(value,
  index)` pairs (`csrc\qsa_lexicographic_topk.cuh:237`), and merging it with the indexer changes only which program does
  it, provided the comparison key and the tie-break order are untouched. Safe if the merged program keeps `topk`
  strictly ordered by `topk`; a `tl.sort`-style reordering would not be.
- **Never** merge two K-reductions into one accumulator (e.g. the HC down mm of two pairs, `w13` of two layers, QSA
  split-K partials of two heads): that is exactly the association change the M=1 contract forbids.

## E. Do not bother

- **Dense GEMM tiles** (`cutlass_70_wmma...16x16` 3.480 ms/196 at 8x; `...32x32` 7.327/340 at 24x; `volta_fp16_s884gemm...`
  1.848/97) — 11–22 us/call, compute/bandwidth-bound (180–546 GB/s on a 900 GB/s part, `PROFILE_TABLES.md` T4), and
  they are the oracle for the batch-invariant contract. Out of scope per the task rules (no BLOCK_K / num_warps /
  load-policy changes).
- **`_qwen38_fp16_rows_gemv_kernel` (72 calls, 1.648 ms, 7.77%) and `_qwen38_fp16_gdn_input_rows_kernel` (36 calls,
  1.811 ms, 8.53%)** — geometry-pinned by the batch-1 oracle (section D) and they *replace* cuBLAS, so merging another
  kernel into them would change the per-row FMA chain. Lever 2 territory, not lever 4.
- **`_hc_combine_norm_kernel` (95 calls, 0.442/0.515 ms)** — already a fusion (combine + RMSNorm, `ops\hc.py:268`).
  Its traffic is (8x10240 fp16 in + 8x10240 fp16 out) ≈ 327 KB, which at the measured 4.66 us is ~70 GB/s ≈ 8% of a
  V100's HBM2 peak: the kernel is launch/occupancy bound, and its per-call time does grow with M (4.66 us at M=8,
  5.42 us at M=24), so it is not a candidate for absorbing another kernel either.
- **All-reduce (rows 15/16) and the NCCL all-gather (0.153 ms at 24x)** — 50 and 48 launches, but the eager numbers
  (212 ms/step at 8x) are wall-clock waits on the slowest rank, so "0.566 ms" is not removable compute, and both
  variants are already fusions (`sm70_cross_device_reduce_sum2_1stage_push` does the elementwise add + reduce in one
  pass, `csrc\custom_all_reduce.cuh:951-979`, dispatch `:2344-2358`). LEVERS.md already tried fusing the adjacent norm
  into them and estimated 1-2%.
- **`_qsa_sparse_paged_gqa_splitk_kernel` (12 calls, 1.144/2.686 ms, 5.4/9.3%) and `_qsa_mqa_paged_kernel`** — 95/224
  us per call, one launch per layer, already split-K + a separate merge; force-merging would need `num_splits` changes
  which the batch-invariant path pins to 1 (`vllm\v1\attention\backends\flash_attn.py:1194` — same pattern in this
  fork). This is lever 5, not lever 4.
- **`w13_kernel` / `w2_kernel` / grouped NVFP4 MoE (48 launches each, 2.2–3.6 ms and 1.2–1.9 ms)** — 25–75 us/call and
  the NVFP4 contract has its own partition semantics; the only cheap-looking part is `plan_v2` (P4).
- **`_gumbel_sample_kernel` (1 call, 0.154/0.184 ms), `_gather_ple_fp8_from_pinned_kernel` (1 call, 0.101–0.107 ms),
  `_scatter_gather_elementwise_kernel` (36 calls, 0.293 ms), `aten::index_select` (36 calls, 0.213 ms),
  `aten::copy_ [[256],[256]]` (24 calls, 0.105 ms), `aten::arange` (78 calls, 0.101 ms)** — 1–78 launches whose cost is
  dominated by one-time work (a single 0.15 ms sampler kernel cannot be merged with anything: it is the last op of the
  step). Note `aten::arange` at 78 calls/step and `aten::copy_` at 24 look like graph-capture hygiene rather than
  kernels worth fusing; **estimate**: they are ~0.2% of a step together.
- **`layer_norm_fwd_kernel` / `rmsnorm_gated_exact_kernel` (36 calls, 0.108–0.136 ms)** — 3.0–3.8 us/call, i.e. at the
  1 us-gap floor; fusing two 36-launch elementwise passes would save ≤0.07 ms (0.33%) and both are already the fused
  form of norm + gate.

## F. What I could not measure (stated so nobody estimates it silently)

- The raw traces (~35 MB x 4 cards per window) are not in the repo; only `analyze_trace.py`'s text output is
  (`prof_1003.sh:52-56`), and the analysis scripts live on the server (`/mnt/2t/build/cmp1001`), so no finer
  intra-step timeline exists. All gap reasoning here is `span - GPU busy` divided by launch count.
- `record_shapes` was off for the graph windows (PROF_G) and on for the eager windows (PROF_E), which is why every
  tensor shape above comes from `E_8000_c8.txt` / `E_200_c24.txt` and every graph count from `G_*`.
- P1–P6 savings are launch-count arithmetic plus measured kernel times; none of them was benchmarked. The one number I
  would measure first is the HC pair chain (P1+P2), because it is 2 of the top 3 small kernels and 15.7% of an 8x step
  end to end.
