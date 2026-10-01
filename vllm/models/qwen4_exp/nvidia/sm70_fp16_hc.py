# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fused checkpoint-FP16 HyperConnection decode route for SM70.

SX batch-1 overlay (dense-multirow, design_4 MR3 + MR9a): decode widths
M >= 2 in the FULL decode graph run the replicated fused HC chain with
multi-row Triton kernels instead of the cuBLAS down/SiLU/up/gate-mix chain.
Per token row the down projection (BLOCK_K 256 / 4 warps, FP16 boundary,
/4, SiLU) and the row-4 up/gate/mix (BLOCK_K 512 / 8 warps, FP16 gate,
sigmoid, branch-ordered FMA) are bitwise equal to the M=1 replicated kernels,
which 1Cat validated bitwise against the production M=1 TP4-sharded route.
Switches: SX_OPT_ROWS / SX_OPT_ROWS_MAX_M / SX_OPT_ROWS_TABLE ("hc" role),
SX_OPT_ROWS_FUSED_REDUCE, SX_OPT_ROWS_HC_DOWN_TILE, SX_OPT_ROWS_HC_DOWN_NW,
SX_OPT_ROWS_HC_UP_TILE and SX_OPT_ROWS_HC_NORM (see sm70_fp16_gemv.py).

SX MTP batch HC (upstream 1Cat main@d30469863: 7b0b303a6, 3b7365925,
732e18417, 1ef9f45a5 on the batch HC of 1ae340320 / c39d4b7a4)
------------------------------------------------------------------------
In the native-MTP lane with k = 4, the target's M5 / M10 verify batches run
a TP4-sharded packed tensor-core HC instead of the replicated cuBLAS chain
(F.linear down -> FP16 -> /4 SiLU, F.linear up -> FP16 gate -> sigmoid mix):
every rank computes its 80 LoRA rows (rank 3 also the four injection rows)
as twenty K512 m8n8k4 partials, each rounded to FP16 like the MTP cuBLAS
split-K, reduces them left to right in FP32, applies the FP16 / SiLU
epilogue, gathers the [M, 320] LoRA as lossless half+tag packets, computes
its 640 hidden columns of the four-branch gate (one ordered K320 MMA, FP16
gate, sigmoid, branch-ordered FMA, /4) and gathers the [M, 2560] block.
Upstream measured it bit-exact against that replicated chain for all 96 real
pairs, four ranks, M5/M10 and changing graph inputs, at 33.6 -> 21.1 us (M5)
/ 34.6 -> 23.6 us (M10) per HC pair with the cooperative fully unrolled
launch. Packed copies: down [3, 640, 2, 32, 8] (this rank's 88 rows, zero
padded to 96) and up [80, 20, 2, 4, 8, 8] (this rank's 640 hidden columns of
all four branches), 3.4375 MiB per pair, i.e. 330 MiB/rank for the 96 pairs;
the checkpoint weights stay for M1/prefill. The custom all-reduce's push
buffer grows by two batch channels (~365 KiB/rank of IPC memory).
Switches: SX_OPT_MTP_HC_BATCH, SX_OPT_MTP_HC_COOPERATIVE and
SX_OPT_MTP_HC_FULL_UNROLL (see sm70_fp16_gemv.py).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .sm70_fp16_gemv import (
    Qwen38SM70FP16LinearMethod,
    Qwen38SM70PackOnlyLinearMethod,
    _exact_runtime_contract,
    _sx_decode_graph_active,
    _sx_mtp_batch_config,
    _sx_mtp_batch_contract,
    _sx_mtp_batch_rows_ok,
    _sx_mtp_batch_takes_width,
    _sx_rows_config,
    _sx_rows_max_m,
    _sx_rows_reduce,
    _sx_rows_x,
    _sx_split_rows,
    _sx_tp4_custom_ar,
)

logger = init_logger(__name__)

_HC_COUNT = 4
_HC_DIM = 2560
_HC_RANK = 320
_HC_HIDDEN = _HC_COUNT * _HC_DIM

# Set when enable_qwen38_sm70_fp16_fused_hc marks at least one module; the
# MR9a combine-norm tile switch in ops/hc.py is confined to such models.
_SX_HC_FUSED_MODULES = 0

_HC_BATCH_DOWN_SHAPE = (3, 640, 2, 32, 8)
_HC_BATCH_UP_SHAPE = (80, 20, 2, 4, 8, 8)


def _pack_hc_batch_weight(weight: torch.Tensor, role: str, rank: int) -> torch.Tensor:
    """Lossless TP4 packs; keep checkpoint layout for M1 and prefill."""
    if weight.dtype != torch.float16 or not 0 <= rank < 4:
        raise ValueError("HC batch packing requires FP16 and a TP4 rank")
    weight = weight.detach()
    if role == "down" and weight.shape == (336, 10240):
        padded = weight.new_zeros((96, 10240))
        padded[:88].copy_(weight[rank * 80 : rank * 80 + 88])
        return padded.reshape(3, 32, 640, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
    if role == "up" and weight.shape == (10240, 320):
        shard = weight.reshape(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640]
        return (
            shard.contiguous()
            .reshape(4, 80, 8, 20, 2, 8)
            .permute(1, 3, 4, 0, 2, 5)
            .contiguous()
        )
    raise ValueError("Unsupported HC batch weight role/geometry")


def _prepare_hc_batch_weight(layer: nn.Module) -> None:
    """Attach this rank's packed copy (called after the weights are final)."""
    weight = layer.weight
    if not weight.is_cuda or weight.dtype != torch.float16:
        return
    from vllm import _custom_ops as ops

    custom_ar = _sx_tp4_custom_ar()
    probe = weight.new_empty((5, _HC_HIDDEN))
    if not (
        ops.supports_sm70_qwen38_hc_batch()
        and custom_ar is not None
        and custom_ar.can_sm70_qwen38_hc_batch(probe)
    ):
        # An older extension, a disabled custom all-reduce or no registered
        # TP4 push buffers: the route could never run, so allocate nothing.
        logger.warning_once(
            "SX_OPT_MTP_HC_BATCH: the TP4 batch HC op or its registered "
            "communicator is unavailable; keeping the replicated M5/M10 HC."
        )
        return
    layer.register_buffer(
        "_sm70_qwen38_hc_batch_packed",
        _pack_hc_batch_weight(
            weight, layer._sm70_qwen38_hc_batch_role, int(custom_ar.rank)
        ),
        persistent=False,
    )


def _batch_runtime_ok(
    x: torch.Tensor,
    packed_down: torch.Tensor | None,
    packed_up: torch.Tensor | None,
    rows_tile: int = 0,
) -> bool:
    # Packed copies exist only where the loader admitted the MTP lane, so
    # every other deployment stops at the first two checks. rows_tile > 0:
    # an admitted SX_OPT_ROWS multi-row HC kernel already serves this width
    # (SX_OPT_ROWS_TABLE hc >= M) and keeps it unless OVER_ROWS is set.
    return bool(
        packed_down is not None
        and packed_up is not None
        and _sx_mtp_batch_config().hc
        and _sx_mtp_batch_rows_ok(x)
        and x.shape[1] == _HC_HIDDEN
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and tuple(packed_down.shape) == _HC_BATCH_DOWN_SHAPE
        and tuple(packed_up.shape) == _HC_BATCH_UP_SHAPE
        and all(
            w.device == x.device
            and w.dtype == x.dtype
            and w.is_contiguous()
            and w.data_ptr() % 16 == 0
            for w in (packed_down, packed_up)
        )
        and _sx_mtp_batch_takes_width(rows_tile)
    )


def _sx_hc_batch_forward(
    x: torch.Tensor,
    packed_down: torch.Tensor,
    packed_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """TP4 batch HC for an admitted M5/M10 verify batch: (block, injection)."""
    custom_ar = _sx_tp4_custom_ar()
    if custom_ar is None or not custom_ar.can_sm70_qwen38_hc_batch(x):
        return None
    config = _sx_mtp_batch_config()
    m = x.shape[0]
    partials = x.new_empty((20, m, 96), dtype=torch.float32)
    lora, local_block, block, injection = (
        x.new_empty((m, n)) for n in (_HC_RANK, _HC_DIM // 4, _HC_DIM, _HC_COUNT)
    )
    custom_ar.sm70_qwen38_hc_batch(
        x,
        packed_down,
        packed_up,
        partials,
        lora,
        local_block,
        block,
        injection,
        round_down_partials=True,
        cooperative=config.hc_cooperative,
        full_unroll=config.hc_full_unroll,
    )
    logger.info_once(
        "SM70 Qwen3.8 MTP TP4 batch HC enabled for M5/M10 (SX_OPT_MTP_HC_BATCH, "
        "cooperative=%s, full_unroll=%s).",
        config.hc_cooperative,
        config.hc_full_unroll,
    )
    return block, injection


@triton.jit
def _qwen38_hc_down_silu_inject_kernel(
    x_ptr,
    weight_ptr,
    lora_ptr,
    injection_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(
            x_ptr + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        weight = tl.load(
            weight_ptr + row * K + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Preserve the baseline GEMV -> FP16 -> SiLU boundary.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    is_lora = row < RANK_VALUE
    scaled = value / HC_COUNT
    tl.store(lora_ptr + row, scaled * tl.sigmoid(scaled), mask=is_lora)
    tl.store(injection_ptr + row - RANK_VALUE, value, mask=~is_lora)


@triton.jit
def _qwen38_hc_down_local_shard_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    TP_RANK: tl.constexpr,
):
    """Compute this TP rank's 80 low-rank rows and one injection row."""
    row = tl.program_id(0)
    active = row < 81
    checkpoint_row = tl.where(row < 80, TP_RANK * 80 + row, 320 + TP_RANK)
    offsets = tl.arange(0, 256)
    acc = tl.zeros((256,), dtype=tl.float32)
    for block_start in tl.static_range(0, 10240, 256):
        indices = block_start + offsets
        x = tl.load(
            x_ptr + indices,
            mask=active,
            other=0.0,
            eviction_policy="evict_last",
        )
        weight = tl.load(
            weight_ptr + checkpoint_row * 10240 + indices,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * weight.to(tl.float32)

    # Match the replicated projection's FP16 materialization before SiLU.
    value = tl.sum(acc, axis=0).to(tl.float16).to(tl.float32)
    scaled = value / 4
    value = tl.where(row < 80, scaled * tl.sigmoid(scaled), value)
    tl.store(output_ptr + row, value, mask=active)
    # Keep the 88-element communication packet aligned to 16 bytes. Padding
    # is canonical zero and is discarded after the rank-ordered gather.
    tl.store(output_ptr + row, 0.0, mask=~active)


@triton.jit
def _qwen38_hc_up_local_gate_kernel(
    lora_ptr,
    weight_ptr,
    gate_ptr,
    TP_RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute the 2560 gate rows owned by this TP rank."""
    hidden = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, 512)
    hidden_mask = hidden < 2560
    k_mask = offsets < 320
    lora = tl.load(
        lora_ptr + offsets,
        mask=k_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    checkpoint_row = TP_RANK * 2560 + hidden
    weight = tl.load(
        weight_ptr + checkpoint_row[:, None] * 320 + offsets[None, :],
        mask=hidden_mask[:, None] & k_mask[None, :],
        other=0.0,
        eviction_policy="evict_first",
    )
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    # The communication kernel applies the original FP16 gate boundary,
    # sigmoid, rank-ordered FP32 FMA, and final FP16 materialization.
    tl.store(gate_ptr + hidden, gate, mask=hidden_mask)


@triton.jit
def _qwen38_hc_up_hidden_shard_kernel(
    lora_ptr,
    weight_ptr,
    branches_ptr,
    out_ptr,
    TP_RANK: tl.constexpr,
):
    """Mix all four branches locally for two of this rank's 640 hidden rows."""
    rows = tl.arange(0, 8)
    hidden = tl.program_id(0) * 2 + rows // 4
    checkpoint_row = (rows % 4) * 2560 + TP_RANK * 640 + hidden
    offsets = tl.arange(0, 512)
    lora = tl.load(lora_ptr + offsets, offsets < 320, 0).to(tl.float32)
    weight = tl.load(
        weight_ptr + checkpoint_row[:, None] * 320 + offsets[None, :],
        offsets[None, :] < 320,
        0,
    )
    # Keep the existing two-K-warp reduction, FP16 gate boundary, and
    # branch-ordered FP32 FMA. Only row ownership changes; weights are neither
    # repacked nor duplicated, and prefill keeps its original layout.
    gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
    gate = gate.to(tl.float16).to(tl.float32).reshape((2, 4))
    branches = tl.load(branches_ptr + checkpoint_row).to(tl.float32).reshape((2, 4))
    result = tl.full((2,), 0, tl.float32)
    for branch in tl.static_range(4):
        index = tl.full((2, 1), branch, tl.int32)
        g = tl.gather(gate, index, 1).reshape((2,))
        x = tl.gather(branches, index, 1).reshape((2,))
        result = tl.fma(tl.sigmoid(g), x, result)
    tl.store(out_ptr + tl.program_id(0) * 2 + tl.arange(0, 2), result / 4)


@triton.jit
def _qwen38_hc_up_gate_mix_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    hidden = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    lora = tl.load(
        lora_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = 0.0
    for stream in tl.static_range(HC_COUNT):
        row = stream * HC_DIMENSION + hidden
        weight = tl.load(
            weight_ptr + row * K + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        # Preserve the baseline GEMV -> FP16 gate -> sigmoid boundary.
        gate = tl.sum(lora * weight.to(tl.float32), axis=0)
        gate = gate.to(tl.float16).to(tl.float32)
        branch = tl.load(x_ptr + stream * HC_DIMENSION + hidden).to(tl.float32)
        result += tl.sigmoid(gate) * branch
    tl.store(out_ptr + hidden, result / HC_COUNT)


@triton.jit
def _qwen38_hc_up_gate_mix_row4_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reuse the low-rank input across four bitwise-equivalent output rows."""
    hidden = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    hidden_mask = hidden < HC_DIMENSION
    k_mask = offsets < K
    lora = tl.load(
        lora_ptr + offsets,
        mask=k_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    result = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for stream in tl.static_range(HC_COUNT):
        row = stream * HC_DIMENSION + hidden
        weight = tl.load(
            weight_ptr + row[:, None] * K + offsets[None, :],
            mask=hidden_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        # Keep the established FP32 reduction and FP16 gate boundary. Row
        # tiling changes only work assignment and shares the lora read.
        gate = tl.sum(lora[None, :] * weight.to(tl.float32), axis=1)
        gate = gate.to(tl.float16).to(tl.float32)
        branch = tl.load(
            x_ptr + stream * HC_DIMENSION + hidden,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        result += tl.sigmoid(gate) * branch
    tl.store(out_ptr + hidden, result / HC_COUNT, mask=hidden_mask)


@triton.jit
def _sx_hc_down_store(
    acc_sum,
    row,
    m,
    M,
    lora_ptr,
    injection_ptr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    MASK_ROWS: tl.constexpr,
):
    # Same epilogue as _qwen38_hc_down_silu_inject_kernel for token row m.
    value = acc_sum.to(tl.float16).to(tl.float32)
    is_lora = row < RANK_VALUE
    is_injection = ~is_lora
    if MASK_ROWS:
        row_ok = m < M
        is_lora = is_lora & row_ok
        is_injection = is_injection & row_ok
    scaled = value / HC_COUNT
    tl.store(
        lora_ptr + m * RANK_VALUE + row,
        scaled * tl.sigmoid(scaled),
        mask=is_lora,
    )
    tl.store(
        injection_ptr + m * HC_COUNT + row - RANK_VALUE,
        value,
        mask=is_injection,
    )


@triton.jit
def _sx_hc_down_epilogue(
    a0,
    a1,
    a2,
    a3,
    row,
    m0,
    M,
    lora_ptr,
    injection_ptr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    FUSED_REDUCE: tl.constexpr,
):
    v0, v1, v2, v3, _u4, _u5, _u6, _u7 = _sx_rows_reduce(
        a0, a1, a2, a3, a0, a0, a0, a0, ROWS, FUSED_REDUCE, 0
    )
    _sx_hc_down_store(
        v0, row, m0, M, lora_ptr, injection_ptr, RANK_VALUE, HC_COUNT, MASK_ROWS
    )
    if ROWS > 1:
        _sx_hc_down_store(
            v1, row, m0 + 1, M, lora_ptr, injection_ptr, RANK_VALUE, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 2:
        _sx_hc_down_store(
            v2, row, m0 + 2, M, lora_ptr, injection_ptr, RANK_VALUE, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 3:
        _sx_hc_down_store(
            v3, row, m0 + 3, M, lora_ptr, injection_ptr, RANK_VALUE, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip


@triton.jit
def _qwen38_hc_down_silu_inject_rows_kernel(
    x_ptr,
    weight_ptr,
    lora_ptr,
    injection_ptr,
    M,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RANK_VALUE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    NW: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    FUSED_REDUCE: tl.constexpr,
):
    """NW checkpoint rows x ROWS token rows per program (MR3 down).

    Oracle: _qwen38_hc_down_silu_inject_kernel (BLOCK_K 256 / 4 warps).  Each
    (weight row, token row) pair owns one 1-D accumulator with the M=1
    layout; x chunks are shared by the NW weight rows, weight chunks by the
    ROWS token rows.  Grid (cdiv(M, ROWS), 324 // NW): the 324 computed rows
    are exactly the M=1 grid; the 12 pad rows are never computed.
    """
    tl.static_assert(NW >= 1)
    tl.static_assert(NW <= 4)
    tl.static_assert(NW != 3)
    tl.static_assert(ROWS >= 1)
    tl.static_assert(ROWS <= 4)
    m0 = tl.program_id(0) * ROWS
    n0 = tl.program_id(1) * NW
    offsets = tl.arange(0, BLOCK_K)
    a00 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a01 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a02 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a03 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a10 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a11 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a12 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a13 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a20 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a21 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a22 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a23 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a30 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a31 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a32 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    a33 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x0 = _sx_rows_x(x_ptr, m0, M, K, indices, mask, True, MASK_ROWS)
        x1 = x0
        x2 = x0
        x3 = x0
        if ROWS > 1:
            x1 = _sx_rows_x(x_ptr, m0 + 1, M, K, indices, mask, True, MASK_ROWS)
        if ROWS > 2:
            x2 = _sx_rows_x(x_ptr, m0 + 2, M, K, indices, mask, True, MASK_ROWS)
        if ROWS > 3:
            x3 = _sx_rows_x(x_ptr, m0 + 3, M, K, indices, mask, True, MASK_ROWS)

        w0 = tl.load(
            weight_ptr + n0 * K + indices,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a00 += x0 * w0
        if ROWS > 1:
            a01 += x1 * w0
        if ROWS > 2:
            a02 += x2 * w0
        if ROWS > 3:
            a03 += x3 * w0
        if NW > 1:
            w1 = tl.load(
                weight_ptr + (n0 + 1) * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            a10 += x0 * w1
            if ROWS > 1:
                a11 += x1 * w1
            if ROWS > 2:
                a12 += x2 * w1
            if ROWS > 3:
                a13 += x3 * w1
        if NW > 2:
            w2 = tl.load(
                weight_ptr + (n0 + 2) * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            w3 = tl.load(
                weight_ptr + (n0 + 3) * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            a20 += x0 * w2
            a30 += x0 * w3
            if ROWS > 1:
                a21 += x1 * w2
                a31 += x1 * w3
            if ROWS > 2:
                a22 += x2 * w2
                a32 += x2 * w3
            if ROWS > 3:
                a23 += x3 * w2
                a33 += x3 * w3

    _sx_hc_down_epilogue(
        a00, a01, a02, a03, n0, m0, M, lora_ptr, injection_ptr,
        RANK_VALUE, HC_COUNT, ROWS, MASK_ROWS, FUSED_REDUCE,
    )  # fmt: skip
    if NW > 1:
        _sx_hc_down_epilogue(
            a10, a11, a12, a13, n0 + 1, m0, M, lora_ptr, injection_ptr,
            RANK_VALUE, HC_COUNT, ROWS, MASK_ROWS, FUSED_REDUCE,
        )  # fmt: skip
    if NW > 2:
        _sx_hc_down_epilogue(
            a20, a21, a22, a23, n0 + 2, m0, M, lora_ptr, injection_ptr,
            RANK_VALUE, HC_COUNT, ROWS, MASK_ROWS, FUSED_REDUCE,
        )  # fmt: skip
        _sx_hc_down_epilogue(
            a30, a31, a32, a33, n0 + 3, m0, M, lora_ptr, injection_ptr,
            RANK_VALUE, HC_COUNT, ROWS, MASK_ROWS, FUSED_REDUCE,
        )  # fmt: skip


@triton.jit
def _sx_hc_lora_row(
    lora_ptr,
    m,
    M,
    K: tl.constexpr,
    offsets,
    k_mask,
    MASK_ROWS: tl.constexpr,
):
    if MASK_ROWS:
        load_mask = k_mask & (m < M)
    else:
        load_mask = k_mask
    return tl.load(
        lora_ptr + m * K + offsets,
        mask=load_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)


@triton.jit
def _sx_hc_branch_row(
    x_ptr,
    m,
    M,
    hidden,
    hidden_mask,
    STREAM: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    MASK_ROWS: tl.constexpr,
):
    if MASK_ROWS:
        load_mask = hidden_mask & (m < M)
    else:
        load_mask = hidden_mask
    return tl.load(
        x_ptr + m * (HC_COUNT * HC_DIMENSION) + STREAM * HC_DIMENSION + hidden,
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _sx_hc_up_stream(
    weight,
    l0,
    l1,
    l2,
    l3,
    l4,
    l5,
    l6,
    l7,
    r0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    x_ptr,
    m0,
    M,
    hidden,
    hidden_mask,
    STREAM: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    FUSED_REDUCE: tl.constexpr,
):
    """One HC stream of _qwen38_hc_up_gate_mix_row4_kernel for ROWS rows."""
    # Branch loads are issued before the reduction barriers so their latency
    # overlaps the gate reduction; the values are unchanged.
    b0 = _sx_hc_branch_row(
        x_ptr, m0, M, hidden, hidden_mask, STREAM, HC_DIMENSION, HC_COUNT,
        MASK_ROWS,
    )  # fmt: skip
    b1 = b0
    b2 = b0
    b3 = b0
    b4 = b0
    b5 = b0
    b6 = b0
    b7 = b0
    if ROWS > 1:
        b1 = _sx_hc_branch_row(
            x_ptr, m0 + 1, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 2:
        b2 = _sx_hc_branch_row(
            x_ptr, m0 + 2, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 3:
        b3 = _sx_hc_branch_row(
            x_ptr, m0 + 3, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 4:
        b4 = _sx_hc_branch_row(
            x_ptr, m0 + 4, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 5:
        b5 = _sx_hc_branch_row(
            x_ptr, m0 + 5, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 6:
        b6 = _sx_hc_branch_row(
            x_ptr, m0 + 6, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 7:
        b7 = _sx_hc_branch_row(
            x_ptr, m0 + 7, M, hidden, hidden_mask, STREAM, HC_DIMENSION,
            HC_COUNT, MASK_ROWS,
        )  # fmt: skip

    # Keep the M=1 product form lora[None, :] * weight_f32 inside the sum:
    # the first in-thread term is a plain multiply, the rest contract to FMA.
    wf = weight.to(tl.float32)
    p0 = l0[None, :] * wf
    p1 = p0
    p2 = p0
    p3 = p0
    p4 = p0
    p5 = p0
    p6 = p0
    p7 = p0
    if ROWS > 1:
        p1 = l1[None, :] * wf
    if ROWS > 2:
        p2 = l2[None, :] * wf
    if ROWS > 3:
        p3 = l3[None, :] * wf
    if ROWS > 4:
        p4 = l4[None, :] * wf
    if ROWS > 5:
        p5 = l5[None, :] * wf
    if ROWS > 6:
        p6 = l6[None, :] * wf
    if ROWS > 7:
        p7 = l7[None, :] * wf
    g0, g1, g2, g3, g4, g5, g6, g7 = _sx_rows_reduce(
        p0, p1, p2, p3, p4, p5, p6, p7, ROWS, FUSED_REDUCE, 1
    )
    # Baseline GEMV -> FP16 gate -> sigmoid boundary, branch-ordered FMA.
    r0 += tl.sigmoid(g0.to(tl.float16).to(tl.float32)) * b0
    if ROWS > 1:
        r1 += tl.sigmoid(g1.to(tl.float16).to(tl.float32)) * b1
    if ROWS > 2:
        r2 += tl.sigmoid(g2.to(tl.float16).to(tl.float32)) * b2
    if ROWS > 3:
        r3 += tl.sigmoid(g3.to(tl.float16).to(tl.float32)) * b3
    if ROWS > 4:
        r4 += tl.sigmoid(g4.to(tl.float16).to(tl.float32)) * b4
    if ROWS > 5:
        r5 += tl.sigmoid(g5.to(tl.float16).to(tl.float32)) * b5
    if ROWS > 6:
        r6 += tl.sigmoid(g6.to(tl.float16).to(tl.float32)) * b6
    if ROWS > 7:
        r7 += tl.sigmoid(g7.to(tl.float16).to(tl.float32)) * b7
    return r0, r1, r2, r3, r4, r5, r6, r7


@triton.jit
def _sx_hc_up_store(
    out_ptr,
    result,
    m,
    M,
    hidden,
    hidden_mask,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    MASK_ROWS: tl.constexpr,
):
    if MASK_ROWS:
        store_mask = hidden_mask & (m < M)
    else:
        store_mask = hidden_mask
    tl.store(out_ptr + m * HC_DIMENSION + hidden, result / HC_COUNT, mask=store_mask)


@triton.jit
def _qwen38_hc_up_gate_mix_row4_rows_kernel(
    lora_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    HC_DIMENSION: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    FUSED_REDUCE: tl.constexpr,
):
    """Row-4 HC up/gate/mix for ROWS token rows sharing the weight tiles.

    Oracle: _qwen38_hc_up_gate_mix_row4_kernel (BLOCK_N 4, BLOCK_K 512,
    8 warps).  The four stream weight tiles are loaded before any reduction
    (pure data movement: the M=1 kernel re-issues each load after the
    previous stream's reduction barrier).
    """
    tl.static_assert(HC_COUNT == 4)
    tl.static_assert(ROWS >= 1)
    tl.static_assert(ROWS <= 8)
    m0 = tl.program_id(0) * ROWS
    hidden = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    hidden_mask = hidden < HC_DIMENSION
    k_mask = offsets < K
    weight_mask = hidden_mask[:, None] & k_mask[None, :]

    l0 = _sx_hc_lora_row(lora_ptr, m0, M, K, offsets, k_mask, MASK_ROWS)
    l1 = l0
    l2 = l0
    l3 = l0
    l4 = l0
    l5 = l0
    l6 = l0
    l7 = l0
    if ROWS > 1:
        l1 = _sx_hc_lora_row(lora_ptr, m0 + 1, M, K, offsets, k_mask, MASK_ROWS)
    if ROWS > 2:
        l2 = _sx_hc_lora_row(lora_ptr, m0 + 2, M, K, offsets, k_mask, MASK_ROWS)
    if ROWS > 3:
        l3 = _sx_hc_lora_row(lora_ptr, m0 + 3, M, K, offsets, k_mask, MASK_ROWS)
    if ROWS > 4:
        l4 = _sx_hc_lora_row(lora_ptr, m0 + 4, M, K, offsets, k_mask, MASK_ROWS)
    if ROWS > 5:
        l5 = _sx_hc_lora_row(lora_ptr, m0 + 5, M, K, offsets, k_mask, MASK_ROWS)
    if ROWS > 6:
        l6 = _sx_hc_lora_row(lora_ptr, m0 + 6, M, K, offsets, k_mask, MASK_ROWS)
    if ROWS > 7:
        l7 = _sx_hc_lora_row(lora_ptr, m0 + 7, M, K, offsets, k_mask, MASK_ROWS)

    w_s0 = tl.load(
        weight_ptr + (0 * HC_DIMENSION + hidden)[:, None] * K + offsets[None, :],
        mask=weight_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    w_s1 = tl.load(
        weight_ptr + (1 * HC_DIMENSION + hidden)[:, None] * K + offsets[None, :],
        mask=weight_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    w_s2 = tl.load(
        weight_ptr + (2 * HC_DIMENSION + hidden)[:, None] * K + offsets[None, :],
        mask=weight_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    w_s3 = tl.load(
        weight_ptr + (3 * HC_DIMENSION + hidden)[:, None] * K + offsets[None, :],
        mask=weight_mask,
        other=0.0,
        eviction_policy="evict_first",
    )

    r0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r6 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r7 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r0, r1, r2, r3, r4, r5, r6, r7 = _sx_hc_up_stream(
        w_s0, l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7,
        x_ptr, m0, M, hidden, hidden_mask, 0, HC_DIMENSION, HC_COUNT, ROWS,
        MASK_ROWS, FUSED_REDUCE,
    )  # fmt: skip
    r0, r1, r2, r3, r4, r5, r6, r7 = _sx_hc_up_stream(
        w_s1, l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7,
        x_ptr, m0, M, hidden, hidden_mask, 1, HC_DIMENSION, HC_COUNT, ROWS,
        MASK_ROWS, FUSED_REDUCE,
    )  # fmt: skip
    r0, r1, r2, r3, r4, r5, r6, r7 = _sx_hc_up_stream(
        w_s2, l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7,
        x_ptr, m0, M, hidden, hidden_mask, 2, HC_DIMENSION, HC_COUNT, ROWS,
        MASK_ROWS, FUSED_REDUCE,
    )  # fmt: skip
    r0, r1, r2, r3, r4, r5, r6, r7 = _sx_hc_up_stream(
        w_s3, l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7,
        x_ptr, m0, M, hidden, hidden_mask, 3, HC_DIMENSION, HC_COUNT, ROWS,
        MASK_ROWS, FUSED_REDUCE,
    )  # fmt: skip

    _sx_hc_up_store(
        out_ptr, r0, m0, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT, MASK_ROWS
    )
    if ROWS > 1:
        _sx_hc_up_store(
            out_ptr, r1, m0 + 1, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 2:
        _sx_hc_up_store(
            out_ptr, r2, m0 + 2, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 3:
        _sx_hc_up_store(
            out_ptr, r3, m0 + 3, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 4:
        _sx_hc_up_store(
            out_ptr, r4, m0 + 4, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 5:
        _sx_hc_up_store(
            out_ptr, r5, m0 + 5, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 6:
        _sx_hc_up_store(
            out_ptr, r6, m0 + 6, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip
    if ROWS > 7:
        _sx_hc_up_store(
            out_ptr, r7, m0 + 7, M, hidden, hidden_mask, HC_DIMENSION, HC_COUNT,
            MASK_ROWS,
        )  # fmt: skip


class _SxHcRowsPlan(NamedTuple):
    down_rows: int
    down_nw: int
    up_rows: int
    fused_reduce: bool


def _sx_hc_rows_plan_for_m(m: int) -> _SxHcRowsPlan | None:
    if m < 2 or m > _sx_rows_max_m("hc"):
        return None
    config = _sx_rows_config()
    return _SxHcRowsPlan(
        down_rows=_sx_split_rows(m, config.hc_down_tile),
        down_nw=config.hc_down_nw,
        up_rows=_sx_split_rows(m, config.hc_up_tile),
        fused_reduce=config.fused_reduce,
    )


def _sx_hc_rows_plan(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> _SxHcRowsPlan | None:
    if x.ndim != 2 or envs.VLLM_BATCH_INVARIANT:
        return None
    plan = _sx_hc_rows_plan_for_m(x.shape[0])
    if plan is None:
        return None
    if not (
        x.shape[1] == _HC_HIDDEN
        and down_weight.shape == (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN)
        and up_weight.shape == (_HC_HIDDEN, _HC_RANK)
        and x.dtype == torch.float16
        and down_weight.dtype == torch.float16
        and up_weight.dtype == torch.float16
        and x.is_cuda
        and x.device == down_weight.device == up_weight.device
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and up_weight.is_contiguous()
        and _sx_decode_graph_active()
    ):
        return None
    return plan


def _sx_hc_rows_forward(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    plan: _SxHcRowsPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicated exact HC mix for x (M, 10240): returns (block, injection)."""
    m = x.shape[0]
    down_rows = (_HC_RANK + _HC_COUNT) // plan.down_nw
    if down_rows * plan.down_nw != _HC_RANK + _HC_COUNT:
        raise ValueError(f"HC down weight-row tile {plan.down_nw} must divide 324")
    lora = x.new_empty((m, _HC_RANK))
    injection = x.new_empty((m, _HC_COUNT))
    block = x.new_empty((m, _HC_DIM))
    _qwen38_hc_down_silu_inject_rows_kernel[
        (triton.cdiv(m, plan.down_rows), down_rows)
    ](
        x,
        down_weight,
        lora,
        injection,
        m,
        K=_HC_HIDDEN,
        BLOCK_K=256,
        RANK_VALUE=_HC_RANK,
        HC_COUNT=_HC_COUNT,
        NW=plan.down_nw,
        ROWS=plan.down_rows,
        MASK_ROWS=m % plan.down_rows != 0,
        FUSED_REDUCE=plan.fused_reduce,
        num_warps=4,
    )
    _qwen38_hc_up_gate_mix_row4_rows_kernel[
        (triton.cdiv(m, plan.up_rows), triton.cdiv(_HC_DIM, 4))
    ](
        lora,
        up_weight,
        x,
        block,
        m,
        K=_HC_RANK,
        HC_DIMENSION=_HC_DIM,
        HC_COUNT=_HC_COUNT,
        BLOCK_N=4,
        BLOCK_K=512,
        ROWS=plan.up_rows,
        MASK_ROWS=m % plan.up_rows != 0,
        FUSED_REDUCE=plan.fused_reduce,
        num_warps=8,
    )
    return block, injection


def sx_hc_rows_norm_admitted(
    n: int, hc_dim: int, hc_count: int, dtype: torch.dtype
) -> bool:
    """MR9a: use the M=1 combine-norm tile at width n (see ops/hc.py)."""
    return bool(
        _SX_HC_FUSED_MODULES > 0
        and hc_dim == _HC_DIM
        and hc_count == _HC_COUNT
        and dtype == torch.float16
        and not envs.VLLM_BATCH_INVARIANT
        and _sx_rows_config().hc_norm
        and _sx_hc_rows_plan_for_m(n) is not None
        and _sx_decode_graph_active()
    )


def _runtime_ok(
    x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor
) -> bool:
    return bool(
        x.ndim == 2
        and x.shape == (1, _HC_HIDDEN)
        and down_weight.shape == (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN)
        and up_weight.shape == (_HC_HIDDEN, _HC_RANK)
        and x.dtype == torch.float16
        and down_weight.dtype == torch.float16
        and up_weight.dtype == torch.float16
        and x.is_cuda
        and down_weight.is_cuda
        and up_weight.is_cuda
        and x.is_contiguous()
        and down_weight.is_contiguous()
        and up_weight.is_contiguous()
        and x.device == down_weight.device == up_weight.device
    )


def _qwen38_sm70_fp16_fused_hc(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    packed_down: torch.Tensor | None = None,
    packed_up: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SX_OPT_MTP_HC_BATCH: the MTP lane's M5/M10 verify batches (packed
    # copies exist only when the loader admitted the lane); every other
    # width keeps the routes below unchanged.
    if packed_down is not None and packed_up is not None:
        rows_plan = _sx_hc_rows_plan(x, down_weight, up_weight)
        if _batch_runtime_ok(x, packed_down, packed_up, int(rows_plan is not None)):
            batch = _sx_hc_batch_forward(x, packed_down, packed_up)
            if batch is not None:
                return batch
    if not _runtime_ok(x, down_weight, up_weight):
        # SX MR3: exact multi-row replicated chain for admitted decode widths
        # inside the FULL decode graph (rows == M=1 route, bitwise).
        rows_plan = _sx_hc_rows_plan(x, down_weight, up_weight)
        if rows_plan is not None:
            logger.info_once(
                "SM70 Qwen3.8 exact multi-row fused FP16 HC route enabled "
                "(SX_OPT_ROWS)."
            )
            return _sx_hc_rows_forward(x, down_weight, up_weight, rows_plan)
        # Preserve the ordinary projection and FP16 materialization boundaries
        # for prefill and any unsupported runtime shape. This fallback lives
        # inside the opaque op so a prefill-first dynamic compile cannot bake
        # the M > 1 decision into subsequent decode graphs.
        down_and_injection = torch.nn.functional.linear(x, down_weight)
        lora = torch.ops.vllm.qwen4_exp_hc_silu(
            down_and_injection[..., :_HC_RANK], _HC_COUNT
        )
        injection = down_and_injection[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        gate = torch.nn.functional.linear(lora, up_weight)
        block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, _HC_COUNT)
        return block, injection
    try:
        from vllm.distributed.parallel_state import get_tp_group

        device_communicator = get_tp_group().device_communicator
        custom_ar = getattr(device_communicator, "ca_comm", None)
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        custom_ar = None

    if custom_ar is not None and custom_ar.can_sm70_qwen38_hc_shard(x):
        tp_rank = int(custom_ar.rank)
        local_down = x.new_empty((1, 88))
        gathered_down = x.new_empty((1, 336))
        block = x.new_empty((1, _HC_DIM))
        _qwen38_hc_down_local_shard_kernel[(88,)](
            x,
            down_weight,
            local_down,
            TP_RANK=tp_rank,
            num_warps=4,
        )
        custom_ar.sm70_qwen38_hc_down_allgather(local_down, gathered_down)
        if custom_ar.supports_sm70_qwen38_hc_up_mix_allgather():
            custom_ar.sm70_qwen38_hc_up_mix_allgather(
                gathered_down, up_weight, x, block
            )
            logger.info_once(
                "SM70 Qwen3.8 exact TP4 fused FP16 HC up/mix/gather enabled."
            )
            return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]
        if custom_ar.supports_sm70_qwen38_hc_output_allgather():
            local_block = x.new_empty((1, _HC_DIM // _HC_COUNT))
            _qwen38_hc_up_hidden_shard_kernel[(320,)](
                gathered_down,
                up_weight,
                x,
                local_block,
                TP_RANK=tp_rank,
                num_warps=8,
            )
            custom_ar.sm70_qwen38_hc_output_allgather(local_block, block)
            logger.info_once(
                "SM70 Qwen3.8 exact TP4 hidden-sharded FP16 HC route enabled."
            )
            return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]

        # An older wheel/sidecar can still use the established gate-sharded
        # route. Never pass its opaque communicator to a different DSO.
        local_gate = x.new_empty((1, _HC_DIM))
        _qwen38_hc_up_local_gate_kernel[(triton.cdiv(_HC_DIM, 8),)](
            gathered_down,
            up_weight,
            local_gate,
            TP_RANK=tp_rank,
            BLOCK_N=8,
            num_warps=8,
        )
        custom_ar.sm70_qwen38_hc_gate_mix(local_gate, x, block)
        logger.info_once(
            "SM70 Qwen3.8 exact TP4-sharded checkpoint-FP16 HC route enabled."
        )
        return block, gathered_down[..., _HC_RANK : _HC_RANK + _HC_COUNT]

    lora = x.new_empty((1, _HC_RANK))
    injection = x.new_empty((1, _HC_COUNT))
    block = x.new_empty((1, _HC_DIM))
    _qwen38_hc_down_silu_inject_kernel[(_HC_RANK + _HC_COUNT,)](
        x,
        down_weight,
        lora,
        injection,
        K=_HC_HIDDEN,
        BLOCK_K=256,
        RANK_VALUE=_HC_RANK,
        HC_COUNT=_HC_COUNT,
        num_warps=4,
    )
    _qwen38_hc_up_gate_mix_row4_kernel[(triton.cdiv(_HC_DIM, 4),)](
        lora,
        up_weight,
        x,
        block,
        K=_HC_RANK,
        HC_DIMENSION=_HC_DIM,
        HC_COUNT=_HC_COUNT,
        BLOCK_N=4,
        BLOCK_K=512,
        num_warps=8,
    )
    logger.info_once("SM70 Qwen3.8 fused checkpoint-FP16 HC M=1 route enabled.")
    return block, injection


def _qwen38_sm70_fp16_fused_hc_fake(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    packed_down: torch.Tensor | None = None,
    packed_up: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del down_weight, up_weight, packed_down, packed_up
    return (
        x.new_empty((*x.shape[:-1], _HC_DIM)),
        x.new_empty((*x.shape[:-1], _HC_COUNT)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_fused_hc",
    op_func=_qwen38_sm70_fp16_fused_hc,
    fake_impl=_qwen38_sm70_fp16_fused_hc_fake,
)


def maybe_apply_qwen38_sm70_fp16_fused_hc(
    down_layer: nn.Module,
    up_layer: nn.Module,
    x: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not enabled or not use_sm70_decode_graph_semantics():
        return None
    down_weight = getattr(down_layer, "weight", None)
    up_weight = getattr(up_layer, "weight", None)
    if down_weight is None or up_weight is None:
        return None
    if down_weight.shape != (
        _HC_RANK + _HC_COUNT + 12,
        _HC_HIDDEN,
    ) or up_weight.shape != (_HC_HIDDEN, _HC_RANK):
        return None
    return torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
        x,
        down_weight,
        up_weight,
        getattr(down_layer, "_sm70_qwen38_hc_batch_packed", None),
        getattr(up_layer, "_sm70_qwen38_hc_batch_packed", None),
    )


def enable_qwen38_sm70_fp16_fused_hc(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Mark exact base-model HC modules for the fused M=1 route."""
    if (
        not envs.VLLM_SM70_QWEN38_FUSED_HC_FP16
        or envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        or dtype != torch.float16
        or not current_platform.is_device_capability((7, 0))
        or not _exact_runtime_contract(vllm_config)
    ):
        return

    # SX_OPT_MTP_HC_BATCH: tag both projections for packing once the
    # checkpoint is loaded (Qwen38SM70FP16LinearMethod.process_weights_after_
    # loading). The up projection has no M1 plan; its apply() still runs the
    # unquantized path outside decode semantics and is bypassed inside them.
    batch_hc = bool(
        _sx_mtp_batch_contract(vllm_config) and _sx_mtp_batch_config().hc
    )
    enabled_count = 0
    for child in module.modules():
        if not (
            getattr(child, "use_combine", False)
            and getattr(child, "lora_rank", None) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
            and hasattr(child, "input_mix_weight_down_block_inject")
            and hasattr(child, "input_mix_weight_up")
        ):
            continue
        child._sm70_qwen38_fp16_fused_hc = True
        if batch_hc:
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod

            for role, layer in (
                ("down", child.input_mix_weight_down_block_inject),
                ("up", child.input_mix_weight_up),
            ):
                if type(layer.quant_method) is UnquantizedLinearMethod:
                    layer.quant_method = Qwen38SM70PackOnlyLinearMethod()
                if not isinstance(layer.quant_method, Qwen38SM70FP16LinearMethod):
                    raise RuntimeError("Batched HC requires checkpoint-FP16 linears")
                layer._sm70_qwen38_hc_batch_role = role
        enabled_count += 1

    global _SX_HC_FUSED_MODULES
    _SX_HC_FUSED_MODULES += enabled_count
    if enabled_count:
        logger.info_once(
            "Prepared %d Qwen3.8 SM70 fused checkpoint-FP16 HC modules.",
            enabled_count,
        )
    if enabled_count and batch_hc:
        logger.info_once(
            "SX_OPT_MTP_HC_BATCH: %d HC pairs will carry TP4 packed copies "
            "(3.4375 MiB/pair/rank) for the M5/M10 MTP verify.",
            enabled_count,
        )


__all__ = [
    "_qwen38_hc_down_local_shard_kernel",
    "_qwen38_hc_down_silu_inject_rows_kernel",
    "_qwen38_hc_up_gate_mix_row4_rows_kernel",
    "_qwen38_hc_up_local_gate_kernel",
    "enable_qwen38_sm70_fp16_fused_hc",
    "maybe_apply_qwen38_sm70_fp16_fused_hc",
    "sx_hc_rows_norm_admitted",
]
