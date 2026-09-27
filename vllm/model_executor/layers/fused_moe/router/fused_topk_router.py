# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SX_OPT_ROUTER32 (default "1"; "0" restores the previous behaviour):
#   Widen the exact SM70 Qwen3.8 E512/K10 Triton router top-k from 1<=M<=16 to
#   1<=M<=32 (design_1 [C2]). The widened route launches
#   _sm70_qwen38_router_topk_runtime_m_kernel, a copy of the M<=16 kernel whose
#   row count M is a runtime (non-specialized) argument, so every width 1..32
#   shares one compiled variant (plus the M1 packed-half-key variant) instead
#   of JIT-compiling a new variant per width mid-serving. M is only used for
#   the integer source-row index; the floating-point body is textually
#   identical, so each row is bitwise identical to the M<=16 kernel.
#   It is only armed for routers built while no speculative decoding (MTP /
#   EAGLE / ...) is configured, tensor_parallel_size == 4 and the text hidden
#   size is 2560 (the deployed Qwen3.8-Flash-Next contract; Qwen3-Next shares
#   the E512/K10 router shape but not the hidden size); other deployments
#   keep the M<=16 constexpr-M kernel and the
#   generic topk_softmax above M16 exactly as before. Numerics: ids and
#   source rows equal topk_softmax; weights within the admitted 1e-7 contract
#   (M17..32 previously used topk_softmax, so those widths are not bitwise
#   vs. the old route; M1..16 are bitwise vs. the old Triton route).
#
# SX_OPT_MTP_MOE_ROUTES (default "1"; "0" restores the 1.8.0-dev2 MTP lane
#   for every item of the batch-3a "moe-verify" group), design_1 [MTP-3] /
#   design_4 [MTP-K4]: the SM70 Qwen3.8-Flash-Next TP4 native-MTP lane
#   (speculative method "mtp" with the Qwen4Exp MTP drafter, 1 <= k <= 7,
#   standard rejection sampling, no parallel drafting, the exact no-MTP
#   Qwen3.8 TP4 model contract of vllm/config/vllm.py on the *target* model,
#   no DBO, SM70) is admitted to the SX_OPT fast paths that were gated on
#   ``speculative_config is None``. ``sx_sm70_qwen38_mtp_verify_q`` below is
#   the single lane contract shared by this router (SX_OPT_ROUTER32 for the
#   target verify rows and the MTP draft), qwen2_moe.py
#   (SX_OPT_SHARED_GATE_ROWS), nvfp4_sm70_moe.py (verify-width MoE routes,
#   SX_OPT_MOE_EAGER_IOTA) and custom_all_reduce.py (25-KiB MTP5 push). Every
#   other speculative configuration keeps the previous behaviour, and without
#   a speculative config nothing changes (the no-MTP lane is untouched).
#   Router numerics in the lane: M1..16 bitwise vs the previous constexpr-M
#   kernel (only the compiled variant is shared); M17..32 (verify widths
#   20/24/25/28/30/32 and wide draft step-0 widths) move from topk_softmax to
#   the runtime-M kernel: ids and source rows equal, weights within 3e-7, each
#   row bitwise equal to the no-MTP lane's router on that row.
#   SX_OPT_MTP_LANE=0 (the batch-3a MTP-lane master of vllm/config/vllm.py,
#   also honoured by the QSA / GEMV / model lane items) disables this group's
#   items as well, so that one switch restores the 1.8.0-dev2 MTP lane.
import os
from collections.abc import Callable
from typing import Any

import torch

import vllm._custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_SX_OPT_ROUTER32 = os.environ.get("SX_OPT_ROUTER32", "1") != "0"
# SX_OPT_MTP_MOE_ROUTES: master switch of the MTP-lane admissions (see top).
_SX_OPT_MTP_MOE_ROUTES = os.environ.get("SX_OPT_MTP_MOE_ROUTES", "1") != "0"
# Width limits of the exact SM70 Qwen3.8 E512/K10 Triton router.
_SM70_QWEN38_ROUTER_TOPK_LEGACY_MAX_M = 16
_SM70_QWEN38_ROUTER_TOPK_RUNTIME_M_MAX_M = 32
# Verify rows per request (1 + k) admitted by the MTP-lane contract.
_SX_MTP_LANE_MAX_VERIFY_Q = 8


def sx_sm70_qwen38_mtp_verify_q(config: Any) -> int:
    """Rows per request (1 + num_speculative_tokens) of the SM70 Qwen3.8 MTP lane.

    Returns 0 unless ``config`` is the SM70 Qwen3.8-Flash-Next TP4 native-MTP
    deployment (see SX_OPT_MTP_MOE_ROUTES at the top of this file) and the
    master switch is on. Pure host logic on the vLLM config; evaluated at
    model construction / weight processing / custom-AR init, never per step.
    The model identity is checked on the *target* model config (the MTP
    drafter is built under the draft model config), with the same exact
    no-MTP Qwen3.8 TP4 contract as vllm/config/vllm.py's dual-compile lane;
    any import or attribute problem fails closed (0). SX_OPT_MTP_LANE=0 (the
    lane-wide master, read here at call time like the other lane groups do)
    also returns 0.
    """
    if not _SX_OPT_MTP_MOE_ROUTES or config is None:
        return 0
    if os.environ.get("SX_OPT_MTP_LANE", "1").strip() == "0":
        return 0
    spec = getattr(config, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "mtp":
        return 0
    use_qwen4_exp_mtp = getattr(spec, "use_qwen4_exp_mtp", None)
    try:
        if not callable(use_qwen4_exp_mtp) or not use_qwen4_exp_mtp():
            return 0
        num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0)
        # Chain drafting only: every verify request carries exactly 1 + k
        # rows (tree verification would carry more state tokens than k).
        state_tokens_fn = getattr(spec, "num_speculative_state_tokens", None)
        state_tokens = (
            int(state_tokens_fn()) if callable(state_tokens_fn) else num_spec
        )
    except Exception:  # noqa: BLE001 - malformed/partial configs fail closed
        return 0
    if (
        not 1 <= num_spec <= _SX_MTP_LANE_MAX_VERIFY_Q - 1
        or state_tokens != num_spec
    ):
        return 0
    if getattr(spec, "parallel_drafting", False):
        return 0
    if getattr(spec, "rejection_sample_method", "standard") != "standard":
        return 0
    model_config = getattr(spec, "target_model_config", None) or getattr(
        config, "model_config", None
    )
    parallel_config = getattr(spec, "target_parallel_config", None) or getattr(
        config, "parallel_config", None
    )
    if parallel_config is None:
        return 0
    if (
        getattr(parallel_config, "enable_dbo", False)
        or getattr(parallel_config, "use_ubatching", False)
        or int(getattr(parallel_config, "ubatch_size", 0) or 0) > 1
    ):
        return 0
    try:
        from vllm.config.vllm import _is_sm70_qwen38_nomtp_dual_compile_contract

        if not _is_sm70_qwen38_nomtp_dual_compile_contract(
            model_config, None, parallel_config
        ):
            return 0
        if not current_platform.is_device_capability(70):
            return 0
    except Exception:  # noqa: BLE001 - defensive; never fail model build
        return 0
    return num_spec + 1


def _sm70_qwen38_router_runtime_m_for_current_config() -> bool:
    """Arm the M<=32 runtime-M router for the no-spec TP4 contract and the
    SM70 Qwen3.8 MTP lane (SX_OPT_MTP_MOE_ROUTES).

    Evaluated once per router at model construction, where the vLLM config is
    current. Without a config (standalone kernels/benchmarks) the deployment
    contract is assumed. Speculative decoding outside the MTP lane, a TP size
    other than 4, or a text hidden size other than Qwen3.8-Flash-Next's 2560
    (e.g. Qwen3-Next, which shares the E512/K10 router shape) keeps the
    previous route (M<=16 constexpr-M kernel, topk_softmax above).
    """
    if not _SX_OPT_ROUTER32:
        return False
    try:
        from vllm.config.vllm import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    except Exception:  # pragma: no cover - defensive import guard
        config = None
    if config is None:
        return True
    if getattr(config, "speculative_config", None) is not None:
        # Target verify rows and the MTP draft (built under the draft model
        # config; the lane contract resolves the target model config).
        return sx_sm70_qwen38_mtp_verify_q(config) > 0
    hidden_size = getattr(
        getattr(getattr(config, "model_config", None), "hf_text_config", None),
        "hidden_size",
        None,
    )
    if isinstance(hidden_size, int) and hidden_size != 2560:
        return False
    tp_size = getattr(
        getattr(config, "parallel_config", None), "tensor_parallel_size", None
    )
    return tp_size is None or int(tp_size) == 4


@triton.jit
def _sm70_qwen38_router_topk_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    token_expert_indices_ptr,
    E: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    PACKED_HALF_KEY: tl.constexpr = False,
) -> None:
    """Sort one exact Qwen3.8 decode or MTP verifier row per program."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_E)
    valid = offsets < E
    logits = tl.load(
        gating_ptr + row * E + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    max_logit = tl.max(logits, axis=0)

    # Match topk_softmax's degenerate-row behavior: NaN, +Inf, or all -Inf
    # produces the first K expert IDs with zero weights.
    has_nan = tl.max((logits != logits).to(tl.int32), axis=0) != 0
    invalid_row = has_nan | (max_logit == float("inf")) | (max_logit == -float("inf"))
    sort_logits = tl.where(invalid_row, -offsets.to(tl.float32), logits)
    # Numeric ties use the lower expert ID in the generic CUDA op. Canonicalize
    # signed zero before bit packing so -0.0 and +0.0 remain one tie class.
    sort_logits = tl.where(sort_logits == 0.0, 0.0, sort_logits)

    # Transform float32 into an ascending-sortable key. Packing the expert ID
    # into the low bits preserves the generic kernel's lower-ID tie break.
    if PACKED_HALF_KEY:
        # FP16 -> FP32 above is exact. Sort the original 16-bit values plus
        # nine expert-ID bits in one int32, without quantizing any logits.
        # Degenerate rows use -offsets (0..511), also exactly representable.
        tl.static_assert(E == 512 and BLOCK_E == 512)
        bits = sort_logits.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        key = tl.where(bits < 0, bits ^ 0x8000, bits ^ 0xFFFF) & 0xFFFF
        # Original int64 sort is signed: flip the key sign bit when moving
        # to a positive 25-bit key so positive logits still precede negatives.
        packed = ((key ^ 0x8000) << 9) | offsets
        sorted_packed = tl.sort(packed, descending=False)
        sorted_keys = (sorted_packed >> 9) ^ 0x8000
        sorted_ids = sorted_packed & 0x1FF
        sorted_bits = tl.where(
            (sorted_keys & 0x8000) != 0,
            sorted_keys ^ 0xFFFF,
            sorted_keys ^ 0x8000,
        ).to(tl.uint16)
        sorted_logits = sorted_bits.to(tl.float16, bitcast=True).to(tl.float32)
    else:
        min_i32: tl.constexpr = -2147483648
        logit_bits = sort_logits.to(tl.int32, bitcast=True)
        sign = logit_bits >> 31
        key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ min_i32)
        key = tl.where(valid, key, 0x7FFFFFFF)
        packed = ((key.to(tl.int64) & 0xFFFFFFFF) << 32) | offsets.to(tl.int64)
        sorted_packed = tl.sort(packed, descending=False)

        sorted_keys = ((sorted_packed >> 32) & 0xFFFFFFFF).to(tl.int32)
        sorted_ids = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
        sorted_sign = sorted_keys >> 31
        sorted_bits = tl.where(sorted_sign < 0, sorted_keys ^ -1, sorted_keys ^ min_i32)
        sorted_logits = sorted_bits.to(tl.float32, bitcast=True)

    raw_weights = tl.math.exp2((sorted_logits - max_logit) * 1.4426950408889634)
    raw_weights = tl.where(invalid_row, 0.0, raw_weights)
    top_mask = offsets < K
    denominator = tl.sum(tl.where(top_mask, raw_weights, 0.0), axis=0)
    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    weights = raw_weights / denominator

    output_offsets = row * K + offsets
    tl.store(topk_ids_ptr + output_offsets, sorted_ids, mask=top_mask)
    tl.store(topk_weights_ptr + output_offsets, weights, mask=top_mask)
    # Match topkGating's rank-major source-row convention.
    tl.store(
        token_expert_indices_ptr + output_offsets, offsets * M + row, mask=top_mask
    )


def _sm70_qwen38_router_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    num_tokens = gating_output.shape[0]
    _sm70_qwen38_router_topk_kernel[(num_tokens,)](
        gating_output,
        topk_weights,
        topk_ids,
        token_expert_indices,
        E=512,
        K=10,
        M=num_tokens,
        BLOCK_E=512,
        PACKED_HALF_KEY=(gating_output.dtype == torch.float16 and num_tokens == 1),
        num_warps=8,
    )


@triton.jit(do_not_specialize=["M"])
def _sm70_qwen38_router_topk_runtime_m_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    token_expert_indices_ptr,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    PACKED_HALF_KEY: tl.constexpr = False,
) -> None:
    """_sm70_qwen38_router_topk_kernel with a runtime, non-specialized M.

    Everything except the source-row index store is a verbatim copy of the
    constexpr-M kernel above (same BLOCK_E/K/num_warps layout, sort, softmax
    and normalization reduction), so every row is bitwise identical to it.
    """

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_E)
    valid = offsets < E
    logits = tl.load(
        gating_ptr + row * E + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    max_logit = tl.max(logits, axis=0)

    # Match topk_softmax's degenerate-row behavior: NaN, +Inf, or all -Inf
    # produces the first K expert IDs with zero weights.
    has_nan = tl.max((logits != logits).to(tl.int32), axis=0) != 0
    invalid_row = has_nan | (max_logit == float("inf")) | (max_logit == -float("inf"))
    sort_logits = tl.where(invalid_row, -offsets.to(tl.float32), logits)
    # Numeric ties use the lower expert ID in the generic CUDA op. Canonicalize
    # signed zero before bit packing so -0.0 and +0.0 remain one tie class.
    sort_logits = tl.where(sort_logits == 0.0, 0.0, sort_logits)

    # Transform float32 into an ascending-sortable key. Packing the expert ID
    # into the low bits preserves the generic kernel's lower-ID tie break.
    if PACKED_HALF_KEY:
        # FP16 -> FP32 above is exact. Sort the original 16-bit values plus
        # nine expert-ID bits in one int32, without quantizing any logits.
        # Degenerate rows use -offsets (0..511), also exactly representable.
        tl.static_assert(E == 512 and BLOCK_E == 512)
        bits = sort_logits.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        key = tl.where(bits < 0, bits ^ 0x8000, bits ^ 0xFFFF) & 0xFFFF
        # Original int64 sort is signed: flip the key sign bit when moving
        # to a positive 25-bit key so positive logits still precede negatives.
        packed = ((key ^ 0x8000) << 9) | offsets
        sorted_packed = tl.sort(packed, descending=False)
        sorted_keys = (sorted_packed >> 9) ^ 0x8000
        sorted_ids = sorted_packed & 0x1FF
        sorted_bits = tl.where(
            (sorted_keys & 0x8000) != 0,
            sorted_keys ^ 0xFFFF,
            sorted_keys ^ 0x8000,
        ).to(tl.uint16)
        sorted_logits = sorted_bits.to(tl.float16, bitcast=True).to(tl.float32)
    else:
        min_i32: tl.constexpr = -2147483648
        logit_bits = sort_logits.to(tl.int32, bitcast=True)
        sign = logit_bits >> 31
        key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ min_i32)
        key = tl.where(valid, key, 0x7FFFFFFF)
        packed = ((key.to(tl.int64) & 0xFFFFFFFF) << 32) | offsets.to(tl.int64)
        sorted_packed = tl.sort(packed, descending=False)

        sorted_keys = ((sorted_packed >> 32) & 0xFFFFFFFF).to(tl.int32)
        sorted_ids = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
        sorted_sign = sorted_keys >> 31
        sorted_bits = tl.where(sorted_sign < 0, sorted_keys ^ -1, sorted_keys ^ min_i32)
        sorted_logits = sorted_bits.to(tl.float32, bitcast=True)

    raw_weights = tl.math.exp2((sorted_logits - max_logit) * 1.4426950408889634)
    raw_weights = tl.where(invalid_row, 0.0, raw_weights)
    top_mask = offsets < K
    denominator = tl.sum(tl.where(top_mask, raw_weights, 0.0), axis=0)
    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    weights = raw_weights / denominator

    output_offsets = row * K + offsets
    tl.store(topk_ids_ptr + output_offsets, sorted_ids, mask=top_mask)
    tl.store(topk_weights_ptr + output_offsets, weights, mask=top_mask)
    # Match topkGating's rank-major source-row convention (runtime M).
    tl.store(
        token_expert_indices_ptr + output_offsets, offsets * M + row, mask=top_mask
    )


def _sm70_qwen38_router_topk_runtime_m(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    """Launch the runtime-M router; one compiled variant serves M1..M32."""
    num_tokens = gating_output.shape[0]
    _sm70_qwen38_router_topk_runtime_m_kernel[(num_tokens,)](
        gating_output,
        topk_weights,
        topk_ids,
        token_expert_indices,
        num_tokens,
        E=512,
        K=10,
        BLOCK_E=512,
        PACKED_HALF_KEY=(gating_output.dtype == torch.float16 and num_tokens == 1),
        num_warps=8,
    )


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
    sm70_qwen38_router_runtime_m: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused top-k routing.

    ``sm70_qwen38_router_runtime_m`` (set only by FusedTopKRouter when
    SX_OPT_ROUTER32 is on for the no-spec Qwen3.8-Flash-Next TP4 contract, see
    _sm70_qwen38_router_runtime_m_for_current_config) widens the
    exact SM70 Qwen3.8 E512/K10 Triton router from M<=16 to M<=32 using the
    runtime-M kernel. Other callers keep the previous M<=16 constexpr route.
    """
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        router_max_m = (
            _SM70_QWEN38_ROUTER_TOPK_RUNTIME_M_MAX_M
            if sm70_qwen38_router_runtime_m
            else _SM70_QWEN38_ROUTER_TOPK_LEGACY_MAX_M
        )
        if (
            envs.VLLM_SM70_QWEN38_ROUTER_TOPK
            and 1 <= M <= router_max_m
            and gating_output.shape == (M, 512)
            and gating_output.dtype == torch.float16
            and gating_output.is_contiguous()
            and topk == 10
            and renormalize
            and topk_ids.dtype == torch.int32
            and current_platform.is_device_capability(70)
        ):
            if sm70_qwen38_router_runtime_m:
                logger.info_once(
                    "SM70 Qwen3.8 E512/K10 router top-k path enabled for M=%d "
                    "(runtime-M kernel, M<=%d).",
                    M,
                    router_max_m,
                )
                _sm70_qwen38_router_topk_runtime_m(
                    topk_weights,
                    topk_ids,
                    token_expert_indices,
                    gating_output,
                )
            else:
                logger.info_once(
                    "SM70 Qwen3.8 E512/K10 router top-k path enabled for M=%d.", M
                )
                _sm70_qwen38_router_topk(
                    topk_weights,
                    topk_ids,
                    token_expert_indices,
                    gating_output,
                )
            return topk_weights, topk_ids, token_expert_indices

        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            indices_type_getter=indices_type_getter,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        # SX_OPT_ROUTER32: decided once at construction (config is current).
        self._sm70_qwen38_router_runtime_m = (
            _sm70_qwen38_router_runtime_m_for_current_config()
        )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
            sm70_qwen38_router_runtime_m=getattr(
                self, "_sm70_qwen38_router_runtime_m", False
            ),
        )

        return topk_weights, topk_ids
