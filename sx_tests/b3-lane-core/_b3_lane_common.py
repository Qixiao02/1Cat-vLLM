# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the b3 lane-core tests (not a test module).

The tests import it from their own directory (pytest puts the test directory
on sys.path; script mode inserts it explicitly). GPU helpers reuse the
dense-multirow harness (sx_tests/dense-multirow/_sx_rows_common.py).
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSE_MULTIROW = os.path.join(os.path.dirname(HERE), "dense-multirow")

K_VALUES = (1, 2, 3, 4)
# MTP-K5 default request counts of the verify graphs (max_num_seqs 24).
LANE_REQS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)
# Requests of the verify widths named in the task (B in {1,2,4,8,12,16,20,24}).
TASK_REQS = (1, 2, 4, 8, 12, 16, 20, 24)
GRID_1024 = [
    32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
    576, 640, 704, 768, 832, 896, 960, 1024,
]  # fmt: skip
MAX_SEQS = 24


def verify_widths(reqs=LANE_REQS, ks=K_VALUES) -> list[int]:
    """Every verify width W = B*(k+1)."""
    return sorted({(k + 1) * b for k in ks for b in reqs})


@contextlib.contextmanager
def sx_env(**values: str | None) -> Iterator[None]:
    """Set/unset env vars (None = unset), restoring them afterwards."""
    from vllm import envs

    saved = {key: os.environ.get(key) for key in values}
    getattr(envs, "disable_envs_cache", lambda: None)()
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def target_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        multimodal_config=SimpleNamespace(language_model_only=True),
        dtype=torch.float16,
        quantization="modelopt_fp4",
        hf_text_config=SimpleNamespace(
            hidden_size=2560,
            num_hidden_layers=48,
            num_experts=512,
            num_experts_per_tok=10,
            moe_intermediate_size=640,
            hc_count=4,
            hc_lowrank=320,
            num_attention_heads=24,
            num_key_value_heads=2,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        ),
    )


def mtp_spec(k: int, method: str = "mtp") -> SimpleNamespace:
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        parallel_drafting=False,
        rejection_sample_method="standard",
        target_model_config=target_model_config(),
        draft_model_config=None,
    )
    spec.use_qwen4_exp_mtp = lambda: method == "mtp"
    spec.num_speculative_state_tokens = lambda: k
    return spec


def lane_config(
    k: int = 4,
    sizes: list[int] | None = None,
    spec: str | None = "mtp",
    max_num_seqs: int = MAX_SEQS,
) -> SimpleNamespace:
    """A VllmConfig stand-in of the SM70 Qwen3.8 TP4 lane (MTP or no-MTP)."""
    import vllm.config.vllm as cfg_mod
    from vllm.config.compilation import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
    )

    q = k + 1
    if sizes is None:
        sizes = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(max_num_seqs, q)
    compilation_config = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=list(sizes),
        max_cudagraph_capture_size=max(sizes),
        splitting_ops=list(CompilationConfig._attention_ops)
        + ["vllm::unified_kv_cache_update"],
        use_inductor_graph_partition=False,
        pass_config=SimpleNamespace(enable_sp=False),
    )
    compilation_config.adjust_cudagraph_sizes_for_spec_decode = types.MethodType(
        CompilationConfig.adjust_cudagraph_sizes_for_spec_decode, compilation_config
    )
    return SimpleNamespace(
        model_config=target_model_config(),
        speculative_config=None if spec is None else mtp_spec(k, spec),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=8192
        ),
        compilation_config=compilation_config,
        lora_config=None,
    )


LANE_ENV = dict(
    SX_OPT_MTP_LANE=None,
    SX_OPT_MTP_ROWS=None,
    SX_OPT_MTP_PW=None,
    SX_OPT_MTP_PW_DRAFT=None,
    SX_OPT_MTP_GRAPH_MAX_REQS=None,
    SX_OPT_MTP_GRAPH_REQS=None,
    SX_OPT_PIECEWISE_MIXED=None,
    SX_OPT_PIECEWISE_MAX_TOKENS=None,
    SX_OPT_PIECEWISE_SIZES=None,
    SX_OPT_PIECEWISE_EAGER_PADDED=None,
    VLLM_SM70_QWEN38_DUAL_COMPILE="1",
    VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH="1",
    VLLM_SM70_QWEN38_HYBRID_PLE="0",
    VLLM_PLE_CPU_OFFLOAD="0",
    VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS="1",
    VLLM_SM70_E4M3_LONG_ATTENTION="0",
)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def rows_common():
    """The dense-multirow harness module (_sx_rows_common)."""
    if DENSE_MULTIROW not in sys.path:
        sys.path.insert(0, DENSE_MULTIROW)
    import _sx_rows_common  # noqa: PLC0415

    return _sx_rows_common


@contextlib.contextmanager
def mtp_verify_ctx(**overrides: str | None) -> Iterator[None]:
    """The target FULL verify-graph capture of the installed native-MTP lane.

    Dual compile on, lane installed, decode-graph compilation context, the
    production rows table unless SX_OPT_ROWS_TABLE is given.
    """
    from vllm.compilation import sm70_decode_graph as dg

    C = rows_common()
    overrides.setdefault("SX_OPT_ROWS_TABLE", None)
    overrides.setdefault("SX_OPT_ROWS", None)
    overrides.setdefault("SX_OPT_ROWS_MAX_M", None)
    overrides.setdefault("SX_OPT_MTP_ROWS", None)
    saved = dg.sm70_mtp_lane_installed()
    with (
        C.sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="1", **overrides),
        dg.sm70_decode_graph_compilation(True),
    ):
        dg.set_sm70_mtp_lane_installed(True)
        try:
            yield
        finally:
            dg.set_sm70_mtp_lane_installed(saved)


@contextlib.contextmanager
def mtp_main_ctx(**overrides: str | None) -> Iterator[None]:
    """The target main backbone (prefill/mixed/PW-1 steps) of the lane."""
    from vllm.compilation import sm70_decode_graph as dg

    C = rows_common()
    saved = dg.sm70_mtp_lane_installed()
    with (
        C.sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="1", **overrides),
        dg.sm70_target_main_backbone(),
    ):
        dg.set_sm70_mtp_lane_installed(True)
        try:
            yield
        finally:
            dg.set_sm70_mtp_lane_installed(saved)
