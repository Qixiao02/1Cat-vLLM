# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in checkpoint-FP16 Qwen3.8 decode GEMV kernels for SM70.

The route is deliberately narrow: exact Qwen3.8 Flash Next topology, TP4,
no speculative decoding, FP16 checkpoint weights, and one decode token. All
prefill and unsupported shapes retain the ordinary unquantized linear path.

SX batch-1 overlay (dense-multirow, design_4 MR1/MR2; MR3 lives in
sm70_fp16_hc.py and reuses the helpers below)
-------------------------------------------------------------------------
Decode widths M >= 2 inside the FULL decode CUDA graph used to fall back to
cuBLAS (``F.linear``).  They now use weight-stationary Triton kernels that
keep one 1-D ``(BLOCK_K,)`` FP32 accumulator per token row with the M=1
plan's BLOCK_K and num_warps.  Triton therefore assigns the M=1 layout:
the same per-slot FMA chain over the K chunks, the same in-thread add, xor
shuffle tree and cross-warp combine.  Every output row is bitwise equal to
the unchanged M=1 kernel run on that row alone (batch invariant).  The M=1
branches are untouched.  Admission requires the dual-compile lane *and* an
active FULL decode-graph capture, so eager/mixed/prefill steps and every
other deployment keep their old behaviour.

Switches (``os.environ``, read once per process; ``_sx_rows_config.
cache_clear()`` re-reads them):

``SX_OPT_ROWS=1``
    ``0`` disables every multi-row route (old cuBLAS fallback for M > 1).
``SX_OPT_ROWS_MAX_M=24``
    Global cap on the decode width M for every role.
``SX_OPT_ROWS_TABLE``
    Per-role maximum M merged over ``_SX_ROWS_DEFAULT_MAX_M``, e.g.
    ``"gdn_in=8,gdn_out=4,qsa_qkv=8,qsa_o=4,qsa_index=8,router=8,hc=2"``
    (the V100-validated default; the implementation's first guess was
    ``router=24,hc=8,qsa_o=8,gdn_out=8``).
    ``0`` disables a role.  Roles: gdn_in (fused GDN qkvz+ba), gdn_out,
    qsa_qkv, qsa_o, qsa_index, router, hc (fused HC mix + combine-norm tile).
``SX_OPT_ROWS_FUSED_REDUCE=1``
    Reduce all token rows of a program in one multi-operand ``tl.reduce``
    (one shared-memory round trip).  ``tl.sum`` is ``tl.reduce`` with
    ``a + b``; the multi-operand lowering applies the identical in-thread,
    shuffle and cross-warp sequence to each operand.  ``0`` = one ``tl.sum``
    per row (the literal M=1 form).
``SX_OPT_ROWS_GEMV_TILE=8``
    Maximum token rows per program for the row GEMV / GDN input kernels.
``SX_OPT_ROWS_HC_DOWN_TILE=4``, ``SX_OPT_ROWS_HC_DOWN_NW=4``,
``SX_OPT_ROWS_HC_UP_TILE=8``
    HC down token rows x weight rows (1, 2 or 4) per program; HC up token
    rows per program.
``SX_OPT_ROWS_HC_NORM=1``
    MR9a: when the HC rows route is admitted at width N, ``hc_combine_norm``
    uses the M=1 tile (BLOCK_SIZE 1024 + weight prefetch) so the normalized
    HC input is also batch invariant.  ``0`` keeps the generic 512 tile.

SX batch 3a (lane-core, design_1 MTP-1)
---------------------------------------
The SM70 Qwen3.8 native-MTP lane (``SX_OPT_MTP_LANE``, contract in
vllm/config/vllm.py) now admits these routes too when it runs the target in
the dual-compile lane: FULL uniform verify graphs (W = B*(k+1) rows) are
traced by the decode compiler, so W <= role maximum takes the rows kernels
above (each row bitwise equal to the M=1 kernel on that row) and wider verify
steps keep cuBLAS; prefill/mixed/eager steps never reach them.
``SX_OPT_MTP_ROWS=0`` disables only the multi-row kernels in that lane.

SX MTP batch routes (upstream 1Cat main@d30469863 MTP4 kernels)
---------------------------------------------------------------
Packed tensor-core kernels that upstream qualified for its native-MTP
single/two-request verify: the native-MTP lane above with k = 4 only, and
only at the target's verify widths M5 / M10 inside the FULL verify-graph
capture (``_sx_mtp_batch_rows_ok``). Each kernel reproduces the arithmetic
of the cuBLAS path it replaces at those widths (same K partitions, FP16
partial / output boundaries and FP32 reduction order, under the MTP lane's
``allow_fp16_reduced_precision_reduction=True`` and
``allow_fp16_accumulation=False``; any other precision policy falls back).
Packed weight copies are made in ``process_weights_after_loading`` only when
the switch is on and the lane contract holds (``_sx_mtp_batch_contract``);
the checkpoint weights keep serving M1, prefill and every other width. The
no-MTP lane, other speculative methods and k != 4 never allocate or run any
of this. Switches (``os.environ``, read once per process;
``_sx_mtp_batch_config.cache_clear()`` re-reads them; an ``SX_OPT_*`` value
wins over the upstream alias, whose name keeps working):

``SX_OPT_MTP_HC_BATCH=1`` (alias ``VLLM_SM70_MTP_HC_BATCH``)
    TP4-sharded packed-MMA HyperConnection (sm70_fp16_hc.py) through the
    custom all-reduce's batch channels: +330 MiB/rank for the 96 HC pairs.
``SX_OPT_MTP_HC_COOPERATIVE=1`` (alias ``VLLM_SM70_MTP_HC_COOPERATIVE``)
    One cooperative launch for down / gather+SiLU / up+mix / gather instead
    of four launches (identical arithmetic and transport).
``SX_OPT_MTP_HC_FULL_UNROLL=1`` (alias ``VLLM_SM70_MTP_HC_FULL_UNROLL``)
    Fully unrolled K loops of the cooperative kernel (same K order).
"""

from __future__ import annotations

import functools
import os
from typing import NamedTuple

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import (
    is_sm70_decode_graph_compiling,
    sm70_mtp_lane_installed,
    use_sm70_decode_graph_semantics,
)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class _GemvPlan(NamedTuple):
    block_k: int
    num_warps: int
    load_policy: int


_HC_DOWN_SUFFIX = ".input_mix_weight_down_block_inject"
_GDN_QKVZ_SUFFIX = ".linear_attn.in_proj_qkvz"
_GDN_BA_SUFFIX = ".linear_attn.in_proj_ba"
_GDN_OUT_SUFFIX = ".linear_attn.out_proj"
_QSA_QKV_SUFFIX = ".self_attn.qkv_proj"
_QSA_OUT_SUFFIX = ".self_attn.o_proj"
_QSA_INDEX_SUFFIX = ".self_attn.indexer.index_qk_proj"
_ROUTER_SUFFIX = ".mlp.gate"

# Plans are cold-cache CUDA Graph winners on real checkpoint weights. Keep the
# role in the key: GDN and QSA can share a physical shape while remaining
# independently auditable.
_ROLE_PLANS: tuple[tuple[str, tuple[int, int], _GemvPlan], ...] = (
    (_HC_DOWN_SUFFIX, (336, 10240), _GemvPlan(512, 4, 1)),
    (_GDN_QKVZ_SUFFIX, (4096, 2560), _GemvPlan(512, 2, 0)),
    (_GDN_BA_SUFFIX, (24, 2560), _GemvPlan(512, 4, 0)),
    (_GDN_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 1)),
    (_QSA_QKV_SUFFIX, (3584, 2560), _GemvPlan(512, 2, 0)),
    (_QSA_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 0)),
    (_QSA_INDEX_SUFFIX, (640, 2560), _GemvPlan(512, 2, 0)),
    (_ROUTER_SUFFIX, (512, 2560), _GemvPlan(1024, 8, 0)),
)

# Retain the legacy two-argument custom-op behavior for external callers.
# Model layers pass their role explicitly: same-shape GDN/QSA plans differ.
_SHAPE_PLANS = {shape: plan for _, shape, plan in _ROLE_PLANS}

# ---------------------------------------------------------------------------
# SX multi-row admission (see the module docstring for the switches).
# ---------------------------------------------------------------------------

# Default per-role maximum decode width.  Production FULL-graph widths are
# 2/4/8/16/24.  design_4: M<=8 sits below the FP32-pipe/DRAM ridge for every
# role; the router stays profitable at 16/24 because cuBLAS reaches only
# ~154 GB/s there.  The validation screen decides; adjust through
# SX_OPT_ROWS_TABLE / SX_OPT_ROWS_MAX_M without a code change.
#
# Validation (opt180dev1, V100 CUDA-graph microbenchmarks with rotating weights):
# rows kernels must be <= 0.97x cuBLAS at every M up to the role maximum. The
# implementation defaults (router=24, hc=8, qsa_o=8, gdn_out=8) lost to cuBLAS
# at the larger widths (router M16/24 0.74x/0.42x; HC chain M4 0.89x, M8 0.63x,
# i.e. -1.7 ms/step; qsa_o/gdn_out M8 0.94x/0.98x), so the measured table is
# the default: gdn_in=8,gdn_out=4,qsa_qkv=8,qsa_o=4,qsa_index=8,router=8,hc=2.
_SX_ROWS_DEFAULT_MAX_M: dict[str, int] = {
    "gdn_in": 8,
    "gdn_out": 4,
    "qsa_qkv": 8,
    "qsa_o": 4,
    "qsa_index": 8,
    "router": 8,
    "hc": 2,
}

_SX_ROWS_ROLE_KEYS: dict[str, str] = {
    _HC_DOWN_SUFFIX: "hc",
    _GDN_QKVZ_SUFFIX: "gdn_in",
    _GDN_BA_SUFFIX: "gdn_in",
    _GDN_OUT_SUFFIX: "gdn_out",
    _QSA_QKV_SUFFIX: "qsa_qkv",
    _QSA_OUT_SUFFIX: "qsa_o",
    _QSA_INDEX_SUFFIX: "qsa_index",
    _ROUTER_SUFFIX: "router",
}

# Kernels carry up to eight named accumulators per program.
_SX_ROWS_MAX_TILE = 8


class _SxRowsConfig(NamedTuple):
    enabled: bool
    max_m: int
    table: dict[str, int]
    fused_reduce: bool
    gemv_tile: int
    hc_down_tile: int
    hc_down_nw: int
    hc_up_tile: int
    hc_norm: bool


def _sx_env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d.", name, raw, default)
        return default
    return max(lo, min(hi, value))


@functools.lru_cache(maxsize=1)
def _sx_rows_config() -> _SxRowsConfig:
    table = dict(_SX_ROWS_DEFAULT_MAX_M)
    for item in os.environ.get("SX_OPT_ROWS_TABLE", "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip()
        try:
            if not sep or key not in table:
                raise ValueError(item)
            table[key] = max(0, int(value))
        except ValueError:
            logger.warning("Ignoring invalid SX_OPT_ROWS_TABLE entry %r.", item)
    hc_down_nw = _sx_env_int("SX_OPT_ROWS_HC_DOWN_NW", 4, 1, 4)
    if hc_down_nw == 3:
        hc_down_nw = 2
    config = _SxRowsConfig(
        enabled=os.environ.get("SX_OPT_ROWS", "1").strip() != "0",
        max_m=_sx_env_int("SX_OPT_ROWS_MAX_M", 24, 0, 1 << 20),
        table=table,
        fused_reduce=os.environ.get("SX_OPT_ROWS_FUSED_REDUCE", "1").strip()
        != "0",
        gemv_tile=_sx_env_int("SX_OPT_ROWS_GEMV_TILE", 8, 1, _SX_ROWS_MAX_TILE),
        hc_down_tile=_sx_env_int("SX_OPT_ROWS_HC_DOWN_TILE", 4, 1, 4),
        hc_down_nw=hc_down_nw,
        hc_up_tile=_sx_env_int("SX_OPT_ROWS_HC_UP_TILE", 8, 1, _SX_ROWS_MAX_TILE),
        hc_norm=os.environ.get("SX_OPT_ROWS_HC_NORM", "1").strip() != "0",
    )
    # Plain info: the dict field is unhashable for info_once's lru_cache, and
    # this function is itself cached (logs once per process / cache_clear).
    logger.info("SX exact multi-row decode configuration: %r", config)
    return config


def _sx_rows_max_m(key: str) -> int:
    config = _sx_rows_config()
    if not config.enabled:
        return 0
    return min(config.max_m, config.table.get(key, 0))


def _sx_split_rows(m: int, cap: int) -> int:
    """Token rows per program: fewest programs, then the most even split."""
    programs = triton.cdiv(m, cap)
    return triton.cdiv(m, programs)


def _sx_mtp_rows_allowed() -> bool:
    # SX batch 3a: SX_OPT_MTP_ROWS=0 keeps the MTP verify graphs on cuBLAS for
    # M > 1. Outside the native-MTP lane this is always True.
    return (
        not sm70_mtp_lane_installed()
        or os.environ.get("SX_OPT_MTP_ROWS", "1").strip() != "0"
    )


def _sx_decode_graph_active() -> bool:
    # The model-side call sites already require decode-graph semantics. In
    # the legacy single-compile lane those semantics are always on, so also
    # require the dual-compile lane and an active FULL decode-graph capture:
    # eager, mixed and prefill steps never reach the multi-row kernels. In the
    # native-MTP lane the FULL decode-graph captures are the uniform verify
    # graphs of the target (the drafter never captures under this context).
    return bool(
        envs.VLLM_SM70_QWEN38_DUAL_COMPILE
        and is_sm70_decode_graph_compiling()
        and current_platform.is_device_capability(70)
        and _sx_mtp_rows_allowed()
    )


def _sx_role_key(role: str, shape: tuple[int, int]) -> str | None:
    if not role:
        # Legacy shape-only callers keep the old M > 1 behaviour.
        return None
    for suffix, expected_shape, _ in _ROLE_PLANS:
        if role.endswith(suffix) and shape == expected_shape:
            return _SX_ROWS_ROLE_KEYS[suffix]
    return None


def _sx_rows_inputs_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return bool(
        x.ndim == 2
        and weight.ndim == 2
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
    )


def _sx_rows_tile(x: torch.Tensor, weight: torch.Tensor, key: str | None) -> int:
    """Token rows per program for an admitted (role, M); 0 = not admitted."""
    if key is None or x.ndim != 2:
        return 0
    m = x.shape[0]
    if m < 2 or m > _sx_rows_max_m(key) or envs.VLLM_BATCH_INVARIANT:
        return 0
    if not _sx_rows_inputs_ok(x, weight) or not _sx_decode_graph_active():
        return 0
    return _sx_split_rows(m, _sx_rows_config().gemv_tile)


# ---------------------------------------------------------------------------
# SX MTP batch admission (see "SX MTP batch routes" in the module docstring).
# ---------------------------------------------------------------------------

# Upstream qualified these kernels for MTP4 (k = 4) only: one / two requests
# give the M5 / M10 verify batches. Never widened to other widths.
_SX_MTP_BATCH_NUM_SPEC = 4
_SX_MTP_BATCH_ROWS = (5, 10)


class _SxMtpBatchConfig(NamedTuple):
    hc: bool
    hc_cooperative: bool
    hc_full_unroll: bool


def _sx_mtp_switch(name: str, alias: str, default: str) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raw = os.environ.get(alias, "").strip()
    return (raw or default) != "0"


@functools.lru_cache(maxsize=1)
def _sx_mtp_batch_config() -> _SxMtpBatchConfig:
    config = _SxMtpBatchConfig(
        hc=_sx_mtp_switch("SX_OPT_MTP_HC_BATCH", "VLLM_SM70_MTP_HC_BATCH", "1"),
        hc_cooperative=_sx_mtp_switch(
            "SX_OPT_MTP_HC_COOPERATIVE", "VLLM_SM70_MTP_HC_COOPERATIVE", "1"
        ),
        hc_full_unroll=_sx_mtp_switch(
            "SX_OPT_MTP_HC_FULL_UNROLL", "VLLM_SM70_MTP_HC_FULL_UNROLL", "1"
        ),
    )
    logger.info("SX MTP batch route configuration: %r", config)
    return config


def _sx_mtp_batch_contract(vllm_config=None) -> bool:
    """Load-time admission: the native-MTP lane target with k = 4.

    _exact_runtime_contract admits a speculative config only for the SM70
    Qwen3.8 TP4 native-MTP lane with the dual-compile target, so this is
    False for the no-MTP lane, every other speculative method and k != 4.
    """
    if envs.VLLM_BATCH_INVARIANT:
        return False
    try:
        config = vllm_config or get_current_vllm_config()
        speculative = config.speculative_config
        ubatching = bool(getattr(config.parallel_config, "use_ubatching", False))
    except (AssertionError, AttributeError, RuntimeError):
        return False
    return bool(
        speculative is not None
        and getattr(speculative, "method", None) == "mtp"
        and getattr(speculative, "num_speculative_tokens", None)
        == _SX_MTP_BATCH_NUM_SPEC
        and not ubatching
        and _exact_runtime_contract(config)
    )


def _sx_mtp_batch_rows_ok(x: torch.Tensor) -> bool:
    """Runtime admission inside the opaque ops: an M5/M10 FULL verify capture.

    The MTP lane's cuBLAS contract that the kernels reproduce bit for bit is
    reduced-precision split-K reduction allowed and FP32 accumulation; any
    other policy keeps the original projections.
    """
    matmul = torch.backends.cuda.matmul
    return bool(
        x.ndim == 2
        and x.shape[0] in _SX_MTP_BATCH_ROWS
        and not envs.VLLM_BATCH_INVARIANT
        and sm70_mtp_lane_installed()
        and is_sm70_decode_graph_compiling()
        and matmul.allow_fp16_reduced_precision_reduction
        and not matmul.allow_fp16_accumulation
    )


def _sx_tp4_custom_ar():
    """The TP group's custom all-reduce communicator, or None."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        return getattr(get_tp_group().device_communicator, "ca_comm", None)
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        return None


@triton.jit
def _sx_add2(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@triton.jit
def _sx_add3(a0, a1, a2, b0, b1, b2):
    return a0 + b0, a1 + b1, a2 + b2


@triton.jit
def _sx_add4(a0, a1, a2, a3, b0, b1, b2, b3):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3


@triton.jit
def _sx_add5(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4


@triton.jit
def _sx_add6(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4, a5 + b5


@triton.jit
def _sx_add7(a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4, a5 + b5, a6 + b6


@triton.jit
def _sx_add8(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7):
    return (
        a0 + b0,
        a1 + b1,
        a2 + b2,
        a3 + b3,
        a4 + b4,
        a5 + b5,
        a6 + b6,
        a7 + b7,
    )


@triton.jit
def _sx_rows_reduce(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    ROWS: tl.constexpr,
    FUSED: tl.constexpr,
    AXIS: tl.constexpr,
):
    """Sum the first ROWS operands along AXIS with the tl.sum tree.

    tl.sum(x) is tl.reduce(x, axis, a + b).  A multi-operand tl.reduce runs
    the same in-thread accumulation, xor-shuffle levels and cross-warp
    shared-memory step for each operand, so both forms give every operand
    the M=1 tree; the fused form shares one barrier pair.  Outputs past ROWS
    are placeholders and must not be stored.
    """
    if FUSED:
        if ROWS == 1:
            v0 = tl.sum(a0, axis=AXIS)
            v1 = v0
            v2 = v0
            v3 = v0
            v4 = v0
            v5 = v0
            v6 = v0
            v7 = v0
        elif ROWS == 2:
            v0, v1 = tl.reduce((a0, a1), AXIS, _sx_add2)
            v2 = v0
            v3 = v0
            v4 = v0
            v5 = v0
            v6 = v0
            v7 = v0
        elif ROWS == 3:
            v0, v1, v2 = tl.reduce((a0, a1, a2), AXIS, _sx_add3)
            v3 = v0
            v4 = v0
            v5 = v0
            v6 = v0
            v7 = v0
        elif ROWS == 4:
            v0, v1, v2, v3 = tl.reduce((a0, a1, a2, a3), AXIS, _sx_add4)
            v4 = v0
            v5 = v0
            v6 = v0
            v7 = v0
        elif ROWS == 5:
            v0, v1, v2, v3, v4 = tl.reduce((a0, a1, a2, a3, a4), AXIS, _sx_add5)
            v5 = v0
            v6 = v0
            v7 = v0
        elif ROWS == 6:
            v0, v1, v2, v3, v4, v5 = tl.reduce(
                (a0, a1, a2, a3, a4, a5), AXIS, _sx_add6
            )
            v6 = v0
            v7 = v0
        elif ROWS == 7:
            v0, v1, v2, v3, v4, v5, v6 = tl.reduce(
                (a0, a1, a2, a3, a4, a5, a6), AXIS, _sx_add7
            )
            v7 = v0
        else:
            v0, v1, v2, v3, v4, v5, v6, v7 = tl.reduce(
                (a0, a1, a2, a3, a4, a5, a6, a7), AXIS, _sx_add8
            )
    else:
        v0 = tl.sum(a0, axis=AXIS)
        v1 = v0
        v2 = v0
        v3 = v0
        v4 = v0
        v5 = v0
        v6 = v0
        v7 = v0
        if ROWS > 1:
            v1 = tl.sum(a1, axis=AXIS)
        if ROWS > 2:
            v2 = tl.sum(a2, axis=AXIS)
        if ROWS > 3:
            v3 = tl.sum(a3, axis=AXIS)
        if ROWS > 4:
            v4 = tl.sum(a4, axis=AXIS)
        if ROWS > 5:
            v5 = tl.sum(a5, axis=AXIS)
        if ROWS > 6:
            v6 = tl.sum(a6, axis=AXIS)
        if ROWS > 7:
            v7 = tl.sum(a7, axis=AXIS)
    return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _sx_rows_x(
    x_ptr,
    m,
    M,
    K: tl.constexpr,
    indices,
    mask,
    EVICT_LAST: tl.constexpr,
    MASK_ROWS: tl.constexpr,
):
    """Load token row m exactly like the M=1 kernels load row 0."""
    if MASK_ROWS:
        load_mask = mask & (m < M)
    else:
        load_mask = mask
    if EVICT_LAST:
        x = tl.load(
            x_ptr + m * K + indices,
            mask=load_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
    else:
        x = tl.load(x_ptr + m * K + indices, mask=load_mask, other=0.0)
    return x.to(tl.float32)


@triton.jit
def _sx_rows_store(out_row_ptr, value, m, M, N: tl.constexpr, MASK_ROWS: tl.constexpr):
    if MASK_ROWS:
        tl.store(out_row_ptr + m * N, value, mask=m < M)
    else:
        tl.store(out_row_ptr + m * N, value)


@triton.jit
def _qwen38_gdn_projection_split_kernel(
    qkvz,
    ba,
    qkv,
    z,
    b,
    a,
    QKV: tl.constexpr,
    Z: tl.constexpr,
    B: tl.constexpr,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
    tl.store(qkv + row * QKV + col, value, col < QKV)
    tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
    if tl.program_id(1) == 0:
        tl.static_assert(B + A <= BLOCK)
        gate_col = tl.arange(0, BLOCK)
        gate = tl.load(ba + row * (B + A) + gate_col, gate_col < B + A, other=0)
        tl.store(b + row * B + gate_col, gate, gate_col < B)
        tl.store(a + row * A + gate_col - B, gate, (gate_col >= B) & (gate_col < B + A))


def _split_gdn_projection_outputs(qkvz, ba):
    m = qkvz.shape[0]
    out = tuple(qkvz.new_empty((m, n)) for n in (2560, 1536, 12, 12))
    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
        qkvz,
        ba,
        *out,
        QKV=2560,
        Z=1536,
        B=12,
        A=12,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return out


def _can_fuse_gdn_projection_split(qkvz: torch.Tensor, ba: torch.Tensor) -> bool:
    # This copy-only operation does not change either GEMM. Keep the existing
    # M1 path and all unsupported layouts; no maximum batch/sequence binding.
    return bool(
        envs.VLLM_SM70_GDN_BATCH_SPLIT_COPY
        and _is_packed_row_major(qkvz)
        and _is_packed_row_major(ba)
        and qkvz.shape[0] > 1
        and qkvz.shape[1] == 4096
        and ba.shape == (qkvz.shape[0], 24)
        and qkvz.dtype == ba.dtype == torch.float16
        and qkvz.is_cuda
        and ba.is_cuda
        and qkvz.device == ba.device
        and current_platform.is_device_capability(70)
    )


@triton.jit
def _qwen38_fp16_row_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
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
        else:
            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        acc += x.to(tl.float32) * weight.to(tl.float32)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _qwen38_fp16_gdn_input_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    is_qkvz = row < 4096
    ba_row = row - 4096
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        acc += x.to(tl.float32) * weight.to(tl.float32)

    value = tl.sum(acc, axis=0)
    is_qkv = is_qkvz & (row < 2560)
    is_z = is_qkvz & (row >= 2560)
    is_b = (~is_qkvz) & (ba_row < 12)
    is_a = (~is_qkvz) & (ba_row >= 12)
    tl.store(qkv_out_ptr + row, value, mask=is_qkv)
    tl.store(z_out_ptr + row - 2560, value, mask=is_z)
    tl.store(b_out_ptr + ba_row, value, mask=is_b)
    tl.store(a_out_ptr + ba_row - 12, value, mask=is_a)


@triton.jit
def _qwen38_fp16_rows_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    FUSED_REDUCE: tl.constexpr,
):
    """ROWS token rows x one weight row (MR2).

    Oracle: _qwen38_fp16_row_gemv_kernel.  Same BLOCK_K/num_warps/loads per
    token row, one 1-D accumulator per row, FMA written as x_f32 * w_f32 +
    acc, so each row's reduction tree and rounding equal the M=1 kernel.
    The token tile index varies fastest so a weight row is re-read from L2.
    """
    tl.static_assert(ROWS >= 1)
    tl.static_assert(ROWS <= 8)
    m0 = tl.program_id(0) * ROWS
    row = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        w = weight.to(tl.float32)
        acc0 += _sx_rows_x(x_ptr, m0, M, K, indices, mask, LOAD_POLICY, MASK_ROWS) * w
        if ROWS > 1:
            acc1 += (
                _sx_rows_x(x_ptr, m0 + 1, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )
        if ROWS > 2:
            acc2 += (
                _sx_rows_x(x_ptr, m0 + 2, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )
        if ROWS > 3:
            acc3 += (
                _sx_rows_x(x_ptr, m0 + 3, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )
        if ROWS > 4:
            acc4 += (
                _sx_rows_x(x_ptr, m0 + 4, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )
        if ROWS > 5:
            acc5 += (
                _sx_rows_x(x_ptr, m0 + 5, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )
        if ROWS > 6:
            acc6 += (
                _sx_rows_x(x_ptr, m0 + 6, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )
        if ROWS > 7:
            acc7 += (
                _sx_rows_x(x_ptr, m0 + 7, M, K, indices, mask, LOAD_POLICY, MASK_ROWS)
                * w
            )

    v0, v1, v2, v3, v4, v5, v6, v7 = _sx_rows_reduce(
        acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, ROWS, FUSED_REDUCE, 0
    )
    out_row = out_ptr + row
    _sx_rows_store(out_row, v0, m0, M, N, MASK_ROWS)
    if ROWS > 1:
        _sx_rows_store(out_row, v1, m0 + 1, M, N, MASK_ROWS)
    if ROWS > 2:
        _sx_rows_store(out_row, v2, m0 + 2, M, N, MASK_ROWS)
    if ROWS > 3:
        _sx_rows_store(out_row, v3, m0 + 3, M, N, MASK_ROWS)
    if ROWS > 4:
        _sx_rows_store(out_row, v4, m0 + 4, M, N, MASK_ROWS)
    if ROWS > 5:
        _sx_rows_store(out_row, v5, m0 + 5, M, N, MASK_ROWS)
    if ROWS > 6:
        _sx_rows_store(out_row, v6, m0 + 6, M, N, MASK_ROWS)
    if ROWS > 7:
        _sx_rows_store(out_row, v7, m0 + 7, M, N, MASK_ROWS)


@triton.jit
def _sx_gdn_rows_store(
    value,
    m,
    M,
    row,
    ba_row,
    is_qkv,
    is_z,
    is_b,
    is_a,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    MASK_ROWS: tl.constexpr,
):
    if MASK_ROWS:
        row_ok = m < M
        is_qkv = is_qkv & row_ok
        is_z = is_z & row_ok
        is_b = is_b & row_ok
        is_a = is_a & row_ok
    tl.store(qkv_out_ptr + m * 2560 + row, value, mask=is_qkv)
    tl.store(z_out_ptr + m * 1536 + row - 2560, value, mask=is_z)
    tl.store(b_out_ptr + m * 12 + ba_row, value, mask=is_b)
    tl.store(a_out_ptr + m * 12 + ba_row - 12, value, mask=is_a)


@triton.jit
def _qwen38_fp16_gdn_input_rows_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    M,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    FUSED_REDUCE: tl.constexpr,
):
    """ROWS token rows x one qkvz/ba weight row in one launch (MR1).

    Oracle: _qwen38_fp16_gdn_input_kernel (row meaning, masked where-select
    of the two weights, fma chain, tl.sum tree and four masked stores).
    """
    tl.static_assert(ROWS >= 1)
    tl.static_assert(ROWS <= 8)
    m0 = tl.program_id(0) * ROWS
    row = tl.program_id(1)
    is_qkvz = row < 4096
    ba_row = row - 4096
    offsets = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        w = weight.to(tl.float32)
        acc0 += _sx_rows_x(x_ptr, m0, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 1:
            acc1 += _sx_rows_x(x_ptr, m0 + 1, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 2:
            acc2 += _sx_rows_x(x_ptr, m0 + 2, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 3:
            acc3 += _sx_rows_x(x_ptr, m0 + 3, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 4:
            acc4 += _sx_rows_x(x_ptr, m0 + 4, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 5:
            acc5 += _sx_rows_x(x_ptr, m0 + 5, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 6:
            acc6 += _sx_rows_x(x_ptr, m0 + 6, M, K, indices, mask, False, MASK_ROWS) * w
        if ROWS > 7:
            acc7 += _sx_rows_x(x_ptr, m0 + 7, M, K, indices, mask, False, MASK_ROWS) * w

    v0, v1, v2, v3, v4, v5, v6, v7 = _sx_rows_reduce(
        acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, ROWS, FUSED_REDUCE, 0
    )
    is_qkv = is_qkvz & (row < 2560)
    is_z = is_qkvz & (row >= 2560)
    is_b = (~is_qkvz) & (ba_row < 12)
    is_a = (~is_qkvz) & (ba_row >= 12)
    _sx_gdn_rows_store(
        v0, m0, M, row, ba_row, is_qkv, is_z, is_b, is_a,
        qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
    )  # fmt: skip
    if ROWS > 1:
        _sx_gdn_rows_store(
            v1, m0 + 1, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 2:
        _sx_gdn_rows_store(
            v2, m0 + 2, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 3:
        _sx_gdn_rows_store(
            v3, m0 + 3, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 4:
        _sx_gdn_rows_store(
            v4, m0 + 4, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 5:
        _sx_gdn_rows_store(
            v5, m0 + 5, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 6:
        _sx_gdn_rows_store(
            v6, m0 + 6, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip
    if ROWS > 7:
        _sx_gdn_rows_store(
            v7, m0 + 7, M, row, ba_row, is_qkv, is_z, is_b, is_a,
            qkv_out_ptr, z_out_ptr, b_out_ptr, a_out_ptr, MASK_ROWS,
        )  # fmt: skip


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return bool(
        not envs.VLLM_BATCH_INVARIANT
        and x.shape[0] == 1
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _sx_rows_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    plan: _GemvPlan,
    tile: int,
    fused_reduce: bool | None = None,
) -> torch.Tensor:
    """Launch the exact multi-row GEMV for x (M, K) with the role's M=1 plan."""
    if fused_reduce is None:
        fused_reduce = _sx_rows_config().fused_reduce
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _qwen38_fp16_rows_gemv_kernel[(triton.cdiv(m, tile), n)](
        x,
        weight,
        out,
        m,
        N=n,
        K=k,
        BLOCK_K=plan.block_k,
        LOAD_POLICY=plan.load_policy,
        ROWS=tile,
        MASK_ROWS=m % tile != 0,
        FUSED_REDUCE=bool(fused_reduce),
        num_warps=plan.num_warps,
    )
    return out


def _qwen38_sm70_fp16_gemv(
    x: torch.Tensor, weight: torch.Tensor, role: str = ""
) -> torch.Tensor:
    shape = (weight.shape[0], weight.shape[1])
    plan = _plan_for(role, shape) if role else _SHAPE_PLANS.get(shape)
    if plan is None or not _runtime_ok(x, weight):
        tile = (
            _sx_rows_tile(x, weight, _sx_role_key(role, shape))
            if plan is not None
            else 0
        )
        if tile:
            logger.info_once(
                "SM70 Qwen3.8 exact multi-row FP16 GEMV route enabled (SX_OPT_ROWS)."
            )
            return _sx_rows_gemv(x, weight, plan, tile)
        return torch.nn.functional.linear(x, weight)

    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _qwen38_fp16_row_gemv_kernel[(weight.shape[0],)](
        x,
        weight,
        out,
        K=weight.shape[1],
        BLOCK_K=plan.block_k,
        LOAD_POLICY=plan.load_policy,
        num_warps=plan.num_warps,
    )
    logger.info_once("SM70 Qwen3.8 checkpoint-FP16 M=1 GEMV route enabled.")
    return out


def _qwen38_sm70_fp16_gemv_fake(
    x: torch.Tensor, weight: torch.Tensor, role: str = ""
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gemv",
    op_func=_qwen38_sm70_fp16_gemv,
    fake_impl=_qwen38_sm70_fp16_gemv_fake,
)


def _sx_gdn_rows_tile(
    x: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor
) -> int:
    if not (
        qkvz_weight.shape == (4096, 2560)
        and ba_weight.shape == (24, 2560)
        and _sx_rows_inputs_ok(x, ba_weight)
    ):
        return 0
    return _sx_rows_tile(x, qkvz_weight, "gdn_in")


def _sx_rows_gdn_input(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    tile: int,
    fused_reduce: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact multi-row fused GDN input for x (M, 2560); rows == M=1 kernel."""
    if fused_reduce is None:
        fused_reduce = _sx_rows_config().fused_reduce
    m = x.shape[0]
    qkv = x.new_empty((m, 2560))
    z = x.new_empty((m, 1536))
    b = x.new_empty((m, 12))
    a = x.new_empty((m, 12))
    _qwen38_fp16_gdn_input_rows_kernel[(triton.cdiv(m, tile), 4096 + 24)](
        x,
        qkvz_weight,
        ba_weight,
        qkv,
        z,
        b,
        a,
        m,
        K=2560,
        BLOCK_K=512,
        ROWS=tile,
        MASK_ROWS=m % tile != 0,
        FUSED_REDUCE=bool(fused_reduce),
        num_warps=2,
    )
    return qkv, z, b, a


def _qwen38_sm70_fp16_gdn_input(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (
        qkvz_weight.shape == (4096, 2560)
        and ba_weight.shape == (24, 2560)
        and _runtime_ok(x, qkvz_weight)
        and _runtime_ok(x, ba_weight)
    ):
        tile = _sx_gdn_rows_tile(x, qkvz_weight, ba_weight)
        if tile:
            logger.info_once(
                "SM70 Qwen3.8 exact multi-row fused GDN input route enabled "
                "(SX_OPT_ROWS)."
            )
            return _sx_rows_gdn_input(x, qkvz_weight, ba_weight, tile)
        qkvz = torch.nn.functional.linear(x, qkvz_weight)
        ba = torch.nn.functional.linear(x, ba_weight)
        if _can_fuse_gdn_projection_split(qkvz, ba):
            logger.info_once("SM70 GDN batched projection split-copy fusion enabled.")
            return _split_gdn_projection_outputs(qkvz, ba)
        return (
            qkvz[..., :2560].contiguous(),
            qkvz[..., 2560:].contiguous(),
            ba[..., :12].contiguous(),
            ba[..., 12:].contiguous(),
        )

    qkv = x.new_empty((1, 2560))
    z = x.new_empty((1, 1536))
    b = x.new_empty((1, 12))
    a = x.new_empty((1, 12))
    _qwen38_fp16_gdn_input_kernel[(4096 + 24,)](
        x,
        qkvz_weight,
        ba_weight,
        qkv,
        z,
        b,
        a,
        K=2560,
        BLOCK_K=512,
        num_warps=2,
    )
    logger.info_once("SM70 Qwen3.8 checkpoint-FP16 fused GDN input route enabled.")
    return qkv, z, b, a


def _qwen38_sm70_fp16_gdn_input_fake(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qkvz_weight, ba_weight
    batch_shape = x.shape[:-1]
    return (
        x.new_empty((*batch_shape, 2560)),
        x.new_empty((*batch_shape, 1536)),
        x.new_empty((*batch_shape, 12)),
        x.new_empty((*batch_shape, 12)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gdn_input",
    op_func=_qwen38_sm70_fp16_gdn_input,
    fake_impl=_qwen38_sm70_fp16_gdn_input_fake,
)


class Qwen38SM70FP16LinearMethod(UnquantizedLinearMethod):
    """Use the row-GEMV custom op for admitted single-token projections.

    Layers tagged by the SX MTP batch loaders also get their packed copies
    here, after the checkpoint weights are final.
    """

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if getattr(layer, "_sm70_qwen38_hc_batch_role", None) is not None:
            from .sm70_fp16_hc import _prepare_hc_batch_weight

            _prepare_hc_batch_weight(layer)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The first dynamic compile sees prefill (M > 1). Keep the M=1
        # decision inside the opaque op so decode does not inherit a baked-in
        # prefill branch.
        if bias is None and use_sm70_decode_graph_semantics():
            return torch.ops.vllm.qwen38_sm70_fp16_gemv(
                x, layer.weight, getattr(layer, "prefix", "")
            )
        return super().apply(layer, x, bias)


def _plan_for(prefix: str, shape: tuple[int, int]) -> _GemvPlan | None:
    for suffix, expected_shape, plan in _ROLE_PLANS:
        if prefix.endswith(suffix) and shape == expected_shape:
            return plan
    return None


def _sx_mtp_lane_speculation_ok(config) -> bool:
    """SX batch 3a: the native-MTP lane with the dual-compile target.

    The routes only pay off (and are only validated) when FULL verify graphs
    are traced by the decode compiler; with VLLM_SM70_QWEN38_DUAL_COMPILE
    disabled the MTP lane keeps the previous unquantized linears.
    """
    if not envs.VLLM_SM70_QWEN38_DUAL_COMPILE:
        return False
    try:
        from vllm.config.vllm import _is_sm70_qwen38_mtp_lane_contract

        return _is_sm70_qwen38_mtp_lane_contract(
            config.model_config, config.speculative_config, config.parallel_config
        )
    except Exception:  # noqa: BLE001 - fail closed on partial configs
        return False


def _exact_runtime_contract(vllm_config=None) -> bool:
    try:
        config = vllm_config or get_current_vllm_config()
        text_config = config.model_config.hf_text_config
        tp_size = int(config.parallel_config.tensor_parallel_size)
    except (AssertionError, AttributeError, RuntimeError):
        return False

    return bool(
        tp_size == 4
        and (
            config.speculative_config is None
            or _sx_mtp_lane_speculation_ok(config)
        )
        and int(getattr(text_config, "hidden_size", 0)) == 2560
        and int(getattr(text_config, "num_hidden_layers", 0)) == 48
        and int(getattr(text_config, "num_experts", 0)) == 512
        and int(getattr(text_config, "num_experts_per_tok", 0)) == 10
        and int(getattr(text_config, "moe_intermediate_size", 0)) == 640
        and int(getattr(text_config, "hc_count", 0)) == 4
        and int(getattr(text_config, "hc_lowrank", 0)) == 320
        and int(getattr(text_config, "num_attention_heads", 0)) == 24
        and int(getattr(text_config, "num_key_value_heads", 0)) == 2
        and int(getattr(text_config, "indexer_head_dim", 0)) == 128
        and int(getattr(text_config, "indexer_budget", 0)) == 2048
        and int(getattr(text_config, "indexer_compress_ratio", 0)) == 4
    )


def enable_qwen38_sm70_fp16_gemv(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Replace admitted unquantized methods before checkpoint loading."""
    if not envs.VLLM_SM70_QWEN38_FP16_GEMV:
        return
    capability_ok = current_platform.is_device_capability((7, 0))
    contract_ok = _exact_runtime_contract(vllm_config)
    if (
        envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        or dtype != torch.float16
        or not capability_ok
        or not contract_ok
    ):
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in rejected: "
            "online_qpn8=%s dtype=%s sm70=%s exact_contract=%s.",
            envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8,
            dtype,
            capability_ok,
            contract_ok,
        )
        return

    replaced = 0
    for child in module.modules():
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        if _plan_for(str(getattr(child, "prefix", "")), shape) is None:
            continue
        child.quant_method = Qwen38SM70FP16LinearMethod()
        replaced += 1

    fused_gdn_inputs = 0
    if envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        for child in module.modules():
            qkvz = getattr(child, "in_proj_qkvz", None)
            ba = getattr(child, "in_proj_ba", None)
            qkvz_weight = getattr(qkvz, "weight", None)
            ba_weight = getattr(ba, "weight", None)
            if not (
                isinstance(qkvz_weight, torch.Tensor)
                and qkvz_weight.shape == (4096, 2560)
                and isinstance(ba_weight, torch.Tensor)
                and ba_weight.shape == (24, 2560)
                and isinstance(
                    getattr(qkvz, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and isinstance(
                    getattr(ba, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and not bool(getattr(child, "gqa_interleaved_layout", True))
                and not bool(getattr(child, "disable_tp_for_ba_proj", True))
            ):
                continue
            child.sm70_qwen38_fp16_fused_input = True
            fused_gdn_inputs += 1

    if replaced:
        logger.info_once(
            "Prepared %d Qwen3.8 checkpoint-FP16 SM70 M=1 GEMV projections.",
            replaced,
        )
    else:
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in matched the runtime but no "
            "target projections were found."
        )
    if fused_gdn_inputs:
        logger.info_once(
            "Prepared %d Qwen3.8 fused checkpoint-FP16 GDN inputs.",
            fused_gdn_inputs,
        )
    elif envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        logger.warning_once(
            "Qwen3.8 fused checkpoint-FP16 GDN input opt-in found no targets."
        )


__all__ = [
    "Qwen38SM70FP16LinearMethod",
    "_qwen38_fp16_gdn_input_kernel",
    "_qwen38_fp16_gdn_input_rows_kernel",
    "_qwen38_fp16_rows_gemv_kernel",
    "enable_qwen38_sm70_fp16_gemv",
]
