# SPDX-License-Identifier: Apache-2.0
"""PW-1 (design_2): PIECEWISE graphs for mixed/prefill steps of 25..1024 tokens.

GPU: none. Pure config / descriptor / dispatch checks; runs anywhere the
overlaid vllm imports (inside the image: no GPU work, no CUDA context).

  /opt/venv/bin/python -m pytest -q sx_tests/b2-piecewise-graphs/test_pw_dispatch_cpu.py
  /opt/venv/bin/python sx_tests/b2-piecewise-graphs/test_pw_dispatch_cpu.py  # + tables

Asserts:
  * size grid: default 23 sizes 32..1024 (gap <= 64 tokens, <= 1/3 relative),
    SX_OPT_PIECEWISE_MAX_TOKENS / SX_OPT_PIECEWISE_SIZES / max_num_batched_tokens
    caps, SX_OPT_PIECEWISE_MIXED=0 -> no sizes.
  * contract gate (real _is_sm70_qwen38_nomtp_dual_compile_contract): sizes only
    for the exact Qwen3.8 TP4 no-MTP dual-compile FULL_AND_PIECEWISE lane with
    hybrid PLE off, no LoRA/DP/DBO/EP/SP, full splitting-op list, SM70;
    every other configuration returns [] (previous behaviour).
  * decode compiler range: _make_qwen38_decode_compile_config keeps the exact
    same (identical object) capture list and range [1, 24] for the deployed
    sizes; in the PW-1 lane an operator list with sizes > max_num_seqs no
    longer widens it (unless SX_OPT_PIECEWISE_MIXED=0); outside the lane
    (FULL / FULL_DECODE_ONLY mode, spec, TP2) the range is max(capture sizes)
    as before.
  * CudaGraphManager: FULL descriptors identical with the switch on and off;
    PIECEWISE descriptors = previous + the new sizes; capture order largest
    first; only ModelCudaGraphManager adds sizes.
  * dispatch for every token count 1..1100, pure decode / mixed / single
    prefill: FULL decode and <=24 PIECEWISE identical to the switch-off manager,
    25..1024 -> PIECEWISE at the next grid size, >1024 -> NONE; the
    design_2 examples dispatch(24, 474, None) -> PIECEWISE 512 and
    dispatch(24, 24, 1) -> FULL 24; SX_OPT_PIECEWISE_EAGER_PADDED=1 turns only
    the new sizes into NONE at the padded token count.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm.config.vllm as cfg_mod
from vllm import envs
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as cgu

FULL = CUDAGraphMode.FULL
PIECEWISE = CUDAGraphMode.PIECEWISE
NONE = CUDAGraphMode.NONE
DEPLOYED_SIZES = [1, 2, 4, 8, 16, 24]
DEFAULT_GRID = [
    32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
    576, 640, 704, 768, 832, 896, 960, 1024,
]
_ENV_KEYS = (
    "SX_OPT_PIECEWISE_MIXED",
    "SX_OPT_PIECEWISE_MAX_TOKENS",
    "SX_OPT_PIECEWISE_SIZES",
    "SX_OPT_PIECEWISE_EAGER_PADDED",
    "VLLM_SM70_QWEN38_DUAL_COMPILE",
    "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH",
    "VLLM_SM70_QWEN38_HYBRID_PLE",
    "VLLM_PLE_CPU_OFFLOAD",
    "VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS",
)


@contextmanager
def sx_env(**values):
    """Set/unset env vars (None = unset), restoring them afterwards."""
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


LANE_ENV = dict(
    SX_OPT_PIECEWISE_MIXED=None,
    SX_OPT_PIECEWISE_MAX_TOKENS=None,
    SX_OPT_PIECEWISE_SIZES=None,
    SX_OPT_PIECEWISE_EAGER_PADDED=None,
    VLLM_SM70_QWEN38_DUAL_COMPILE="1",
    VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH="1",
    VLLM_SM70_QWEN38_HYBRID_PLE="0",
    VLLM_PLE_CPU_OFFLOAD="0",
    VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=None,
)


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    """Deployed lane env + SM70 platform answer for every test."""
    monkeypatch.setattr(cfg_mod, "_sx_piecewise_platform_ok", lambda: True)
    with sx_env(**LANE_ENV):
        yield


def _hf_text_config(**overrides):
    values = dict(
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
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_vllm_config(
    *,
    capture_sizes=None,
    max_num_seqs=24,
    max_num_batched_tokens=8192,
    tp=4,
    spec=None,
    cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    mode=CompilationMode.VLLM_COMPILE,
    splitting_ops=None,
    lora=None,
    dp=1,
    hidden=2560,
    partition=False,
):
    if splitting_ops is None:
        splitting_ops = list(CompilationConfig._attention_ops) + [
            "vllm::unified_kv_cache_update",
            "vllm::unified_mla_kv_cache_update",
        ]
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen4ExpForConditionalGeneration"],
            multimodal_config=SimpleNamespace(language_model_only=True),
            dtype=torch.float16,
            hf_text_config=_hf_text_config(hidden_size=hidden),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=dp,
            enable_dbo=False,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        compilation_config=SimpleNamespace(
            mode=mode,
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=list(capture_sizes or DEPLOYED_SIZES),
            max_cudagraph_capture_size=max(capture_sizes or DEPLOYED_SIZES),
            splitting_ops=splitting_ops,
            use_inductor_graph_partition=partition,
            pass_config=SimpleNamespace(enable_sp=False),
            compile_ranges_endpoints=[max_num_batched_tokens + 1],
            compile_sizes=[],
            cache_dir="/tmp/x",
            local_cache_dir="/tmp/x/rank",
            traced_files={"a.py"},
        ),
        speculative_config=spec,
        lora_config=lora,
    )


# ----------------------------------------------------------------------------
# size grid and contract gate
# ----------------------------------------------------------------------------
def test_default_grid():
    grid = cfg_mod._sm70_qwen38_mixed_piecewise_grid(1024)
    assert grid == DEFAULT_GRID
    for prev, size in zip([24] + grid[:-1], grid):
        worst_pad = size - (prev + 1)  # step prev+1 pads to size
        assert worst_pad <= 63, (prev, size)
        assert worst_pad / size <= 1 / 3 + 1e-9, (prev, size)
    assert cfg_mod._sm70_qwen38_mixed_piecewise_grid(1000)[-3:] == [896, 960, 1000]
    big = cfg_mod._sm70_qwen38_mixed_piecewise_grid(2048)
    assert big[-9:] == [1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    assert cfg_mod._sm70_qwen38_mixed_piecewise_grid(24) == [24]
    assert cfg_mod._sm70_qwen38_mixed_piecewise_grid(0) == []


def test_contract_default_sizes():
    sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
        fake_vllm_config(), DEPLOYED_SIZES
    )
    assert sizes == DEFAULT_GRID


def test_env_knobs():
    cfg = fake_vllm_config()
    with sx_env(SX_OPT_PIECEWISE_MIXED="0"):
        assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(cfg, DEPLOYED_SIZES) == []
        assert not cfg_mod._sx_piecewise_mixed_enabled()
    with sx_env(SX_OPT_PIECEWISE_MAX_TOKENS="0"):
        assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(cfg, DEPLOYED_SIZES) == []
    with sx_env(SX_OPT_PIECEWISE_MAX_TOKENS="512"):
        sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(cfg, DEPLOYED_SIZES)
        assert sizes == [s for s in DEFAULT_GRID if s <= 512]
    with sx_env(SX_OPT_PIECEWISE_MAX_TOKENS="junk"):
        sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(cfg, DEPLOYED_SIZES)
        assert sizes == DEFAULT_GRID
    with sx_env(SX_OPT_PIECEWISE_SIZES=" 64,128 ,abc,16,24,4096,,512"):
        sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(cfg, DEPLOYED_SIZES)
        assert sizes == [64, 128, 512]  # <=24 and > max tokens dropped
    small = fake_vllm_config(max_num_batched_tokens=600)
    sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(small, DEPLOYED_SIZES)
    assert sizes[-2:] == [576, 600] and max(sizes) == 600


@pytest.mark.parametrize(
    "label, cfg_kwargs, env",
    [
        ("spec", dict(spec=SimpleNamespace(method="mtp", num_speculative_tokens=1)), {}),
        ("tp2", dict(tp=2), {}),
        ("hidden4096", dict(hidden=4096), {}),
        ("full_decode_only", dict(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY), {}),
        ("piecewise_only", dict(cudagraph_mode=CUDAGraphMode.PIECEWISE), {}),
        ("full_only", dict(cudagraph_mode=CUDAGraphMode.FULL), {}),
        ("no_compile", dict(mode=CompilationMode.NONE), {}),
        ("partition", dict(partition=True), {}),
        ("custom_split_ops", dict(splitting_ops=["vllm::unified_attention_with_output"]), {}),
        ("lora", dict(lora=SimpleNamespace()), {}),
        ("dp2", dict(dp=2), {}),
        ("hybrid_ple", {}, dict(VLLM_SM70_QWEN38_HYBRID_PLE="1")),
        ("cpu_offload_ple", {}, dict(VLLM_PLE_CPU_OFFLOAD="1")),
        ("no_dual_compile", {}, dict(VLLM_SM70_QWEN38_DUAL_COMPILE="0")),
        ("no_compile_graph", {}, dict(VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH="0")),
    ],
)
def test_contract_rejections(label, cfg_kwargs, env):
    cfg = fake_vllm_config(**cfg_kwargs)
    with sx_env(**env):
        assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(cfg, DEPLOYED_SIZES) == [], label


def test_platform_rejection(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_sx_piecewise_platform_ok", lambda: False)
    assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
        fake_vllm_config(), DEPLOYED_SIZES
    ) == []


def test_operator_sizes_above_max_seqs():
    # An explicit list with 32/64 keeps them (existing PIECEWISE-only sizes)
    # and only adds the grid sizes above its maximum.
    sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
        fake_vllm_config(), DEPLOYED_SIZES + [32, 64]
    )
    assert sizes == [s for s in DEFAULT_GRID if s > 64]


# ----------------------------------------------------------------------------
# decode compiler range decoupling (model.py)
# ----------------------------------------------------------------------------
def test_decode_graph_sizes():
    f = cfg_mod._sm70_qwen38_decode_graph_sizes
    assert f(DEPLOYED_SIZES, 24) == DEPLOYED_SIZES
    assert f(DEPLOYED_SIZES + [32, 64, 512], 24) == DEPLOYED_SIZES
    assert f([1, 2, 4, 8, 16, 32], 32) == [1, 2, 4, 8, 16, 32]
    assert f([2, 4, 8, 16, 32, 48, 64], 24, 2) == [2, 4, 8, 16, 32, 48]
    assert f([], 24) == [1]
    assert f([64, 128], 24) == [64]


def _decode_config(vllm_config):
    from vllm.models.qwen4_exp.nvidia.model import _make_qwen38_decode_compile_config

    return _make_qwen38_decode_compile_config(vllm_config)


def test_decode_compile_config_deployed_is_unchanged():
    cfg = fake_vllm_config()
    original_sizes = cfg.compilation_config.cudagraph_capture_sizes
    for env in ({}, dict(SX_OPT_PIECEWISE_MIXED="0")):
        with sx_env(**env):
            dec = _decode_config(cfg)
        assert dec.scheduler_config.max_num_batched_tokens == 24
        assert dec.compilation_config.compile_ranges_endpoints == [24]
        # Byte-for-byte: the very same list object, max size untouched.
        assert dec.compilation_config.cudagraph_capture_sizes is original_sizes
        assert dec.compilation_config.max_cudagraph_capture_size == 24
        assert dec.compilation_config.compile_sizes == []
    # The main config is never modified.
    assert cfg.scheduler_config.max_num_batched_tokens == 8192
    assert cfg.compilation_config.compile_ranges_endpoints == [8193]
    assert cfg.compilation_config.cudagraph_capture_sizes == DEPLOYED_SIZES


def test_decode_compile_config_operator_list():
    cfg = fake_vllm_config(capture_sizes=DEPLOYED_SIZES + [32, 64])
    dec = _decode_config(cfg)
    assert dec.scheduler_config.max_num_batched_tokens == 24
    assert dec.compilation_config.compile_ranges_endpoints == [24]
    assert dec.compilation_config.cudagraph_capture_sizes == DEPLOYED_SIZES
    assert dec.compilation_config.max_cudagraph_capture_size == 24
    assert cfg.compilation_config.cudagraph_capture_sizes == DEPLOYED_SIZES + [32, 64]
    with sx_env(SX_OPT_PIECEWISE_MIXED="0"):  # previous behaviour
        old = _decode_config(cfg)
    assert old.scheduler_config.max_num_batched_tokens == 64
    assert old.compilation_config.compile_ranges_endpoints == [64]


@pytest.mark.parametrize(
    "label, cfg_kwargs",
    [
        # FULL mode captures mixed FULL graphs at 32/64 through the decode
        # compiler: narrowing its range would make that capture assert.
        ("full_only", dict(cudagraph_mode=CUDAGraphMode.FULL)),
        ("full_decode_only", dict(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)),
        ("spec", dict(spec=SimpleNamespace(method="mtp", num_speculative_tokens=1))),
        ("tp2", dict(tp=2)),
    ],
)
def test_decode_compile_config_outside_lane_unchanged(label, cfg_kwargs):
    # Outside the PW-1 lane the decode range stays max(capture sizes), with
    # the switch on, exactly as before.
    cfg = fake_vllm_config(capture_sizes=DEPLOYED_SIZES + [32, 64], **cfg_kwargs)
    original_sizes = cfg.compilation_config.cudagraph_capture_sizes
    dec = _decode_config(cfg)
    assert dec.scheduler_config.max_num_batched_tokens == 64, label
    assert dec.compilation_config.compile_ranges_endpoints == [64], label
    assert dec.compilation_config.cudagraph_capture_sizes is original_sizes, label
    assert dec.compilation_config.max_cudagraph_capture_size == 64, label


# ----------------------------------------------------------------------------
# CudaGraphManager descriptors and dispatch
# ----------------------------------------------------------------------------
def make_manager(cfg=None, cls=cgu.ModelCudaGraphManager, captured=True):
    """Build the manager without CUDA/distributed state; runs the real
    _sx_resolve_piecewise_only_sizes, _init_candidates and dispatch."""
    cfg = cfg or fake_vllm_config()
    mgr = object.__new__(cls)
    mgr.vllm_config = cfg
    mgr.compilation_config = cfg.compilation_config
    mgr.max_num_reqs = cfg.scheduler_config.max_num_seqs
    mgr.cudagraph_mode = cfg.compilation_config.cudagraph_mode
    mgr.decode_query_len = 1
    mgr.decode_query_lens = (1,)
    mgr._sm70_dflash2_tail_graphs = False
    mgr.dp_size = cfg.parallel_config.data_parallel_size
    mgr.tp_size = cfg.parallel_config.tensor_parallel_size
    mgr.graphs = {}
    mgr._graphs_captured = False
    mgr._candidates = []
    mgr._capture_descs = {}
    mgr._capture_sizes = list(cfg.compilation_config.cudagraph_capture_sizes)
    # Same two calls, in the same order, as the end of CudaGraphManager.__init__.
    mgr._sx_init_piecewise_only()
    mgr._init_candidates()
    mgr._graphs_captured = captured
    return mgr


def _pad(n, sizes):
    return next((s for s in sorted(sizes) if s >= n), None)


def test_descriptors_on_vs_off():
    on = make_manager()
    with sx_env(SX_OPT_PIECEWISE_MIXED="0"):
        off = make_manager()
    assert on._sx_piecewise_only_sizes == tuple(DEFAULT_GRID)
    assert off._sx_piecewise_only_sizes == ()
    # FULL decode descriptors: identical list (content and order).
    assert on._capture_descs[FULL] == off._capture_descs[FULL]
    assert [d.num_tokens for d in off._capture_descs[FULL]] == [24, 16, 8, 4, 2, 1]
    assert all(d.num_reqs == d.num_tokens and d.uniform_token_count == 1
               for d in on._capture_descs[FULL])
    # PIECEWISE: previous list + the new sizes, captured largest first.
    old_pw = off._capture_descs[PIECEWISE]
    new_pw = on._capture_descs[PIECEWISE]
    assert [d.num_tokens for d in old_pw] == [24, 16, 8, 4, 2, 1]
    assert [d.num_tokens for d in new_pw] == sorted(
        DEFAULT_GRID + DEPLOYED_SIZES, reverse=True
    )
    assert set(old_pw) <= set(new_pw)
    assert all(d.num_reqs is None and d.uniform_token_count is None for d in new_pw)
    # Candidate lists for <= 24 tokens are the same objects' values.
    for n in range(1, 25):
        assert on._candidates[n] == off._candidates[n], n
    assert len(on._candidates) == 1025 and len(off._candidates) == 25


def test_only_model_manager_adds_sizes():
    base = make_manager(cls=cgu.CudaGraphManager)
    assert base._sx_piecewise_only_sizes == ()
    assert [d.num_tokens for d in base._capture_descs[PIECEWISE]] == [24, 16, 8, 4, 2, 1]


def _expected(kind, n, on):
    """Expected (mode, padded tokens) for a batch of n tokens."""
    full_sizes = DEPLOYED_SIZES
    pw_sizes = DEPLOYED_SIZES + (DEFAULT_GRID if on else [])
    if kind == "decode":
        return FULL, _pad(n, full_sizes)
    if kind == "single_prefill" and n == 1:
        return FULL, 1  # a 1-token chunk is indistinguishable from decode
    padded = _pad(n, pw_sizes)
    if padded is None:
        return NONE, n
    return PIECEWISE, padded


def _batch(kind, n):
    """(num_reqs, num_tokens, uniform_token_count) as execute_model computes."""
    if kind == "decode":
        return n, n, 1
    if kind == "single_prefill":
        return 1, n, n
    # mixed: up to 24 requests, all decodes (1 token) except one prefill
    # chunk with the remaining n - num_reqs + 1 tokens (>= 2).
    num_reqs = min(n - 1, 24)
    max_query_len = n - num_reqs + 1
    return num_reqs, n, cgu.get_uniform_token_count(num_reqs, n, max_query_len)


def _skip(kind, n):
    return (kind == "decode" and n > 24) or (kind == "mixed" and n < 2)


def dispatch_table(mgr, kinds=("decode", "mixed", "single_prefill"), max_n=1100):
    table = {}
    for kind in kinds:
        for n in range(1, max_n + 1):
            if _skip(kind, n):
                continue
            desc = mgr.dispatch(*_batch(kind, n))
            table[(kind, n)] = desc
    return table


def test_dispatch_1_to_1100():
    on = make_manager()
    with sx_env(SX_OPT_PIECEWISE_MIXED="0"):
        off = make_manager()
    t_on, t_off = dispatch_table(on), dispatch_table(off)
    for (kind, n), desc in t_on.items():
        mode, tokens = _expected(kind, n, True)
        assert (desc.cg_mode, desc.num_tokens) == (mode, tokens), (kind, n, desc)
        if mode == FULL:
            assert desc.num_reqs == tokens and desc.uniform_token_count == 1
        if mode == PIECEWISE:
            assert desc.num_reqs is None
        old = t_off[(kind, n)]
        omode, otokens = _expected(kind, n, False)
        assert (old.cg_mode, old.num_tokens) == (omode, otokens), (kind, n, old)
        if n <= 24:
            assert desc == old, (kind, n)  # previous behaviour kept exactly


def test_design_examples():
    mgr = make_manager()
    assert mgr.dispatch(24, 474, None) == cgu.BatchExecutionDescriptor(
        cg_mode=PIECEWISE, num_tokens=512, num_reqs=None
    )
    d24 = mgr.dispatch(24, 24, 1)
    assert (d24.cg_mode, d24.num_tokens, d24.num_reqs) == (FULL, 24, 24)
    d17 = mgr.dispatch(17, 17, 1)
    assert (d17.cg_mode, d17.num_tokens, d17.num_reqs) == (FULL, 24, 24)
    assert mgr.dispatch(17, 16 + 784, None).num_tokens == 832  # 16 dec + 784 chunk
    assert mgr.dispatch(17, 16 + 450, None).num_tokens == 512
    assert mgr.dispatch(1, 1024, 1024).cg_mode == PIECEWISE
    assert mgr.dispatch(24, 1025, None) == cgu.BatchExecutionDescriptor(
        cg_mode=NONE, num_tokens=1025, num_reqs=24
    )
    assert mgr.dispatch(2, 8192, 4096).cg_mode == NONE
    # Before capture every step is eager (profile / warmup runs).
    assert make_manager(captured=False).dispatch(24, 474, None).cg_mode == NONE


def test_eager_padded_diagnostic():
    with sx_env(SX_OPT_PIECEWISE_EAGER_PADDED="1"):
        mgr = make_manager()
    assert mgr._sx_eager_padded
    d = mgr.dispatch(24, 474, None)
    assert (d.cg_mode, d.num_tokens, d.num_reqs) == (NONE, 512, 24)
    assert mgr.dispatch(24, 20, None).cg_mode == PIECEWISE  # old sizes untouched
    assert mgr.dispatch(24, 24, 1).cg_mode == FULL
    assert mgr.dispatch(24, 2000, None) == cgu.BatchExecutionDescriptor(
        cg_mode=NONE, num_tokens=2000, num_reqs=24
    )
    with sx_env(SX_OPT_PIECEWISE_EAGER_PADDED="1", SX_OPT_PIECEWISE_MIXED="0"):
        off = make_manager()
    assert not off._sx_eager_padded


def test_padding_waste_report():
    """Prints the padding waste of the default grid (no assertion beyond
    the grid test): mean/max padded tokens per mixed step size band."""
    mgr = make_manager()
    rows = []
    for lo, hi in ((25, 128), (129, 256), (257, 512), (513, 1024)):
        waste = [mgr.dispatch(24, n, None).num_tokens - n for n in range(lo, hi + 1)]
        rows.append((lo, hi, sum(waste) / len(waste), max(waste)))
        assert max(waste) <= 63
    print("\n[PW-1] padding per mixed step (tokens): band, mean, max")
    for lo, hi, mean, worst in rows:
        print(f"  {lo:4d}..{hi:4d}: mean {mean:5.1f}  max {worst:3d}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
