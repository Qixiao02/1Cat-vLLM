# SPDX-License-Identifier: Apache-2.0
"""MTP-lane contract, options, host metadata builder and planners (no GPU).

Needs the image's vllm import; launches no GPU kernel (the SM70 capability
query is monkeypatched where a gate reads it).
  /opt/venv/bin/python -m pytest -q sx_tests/b3-qsa-mtp/test_mtp_lane_cpu.py

Covers design_1 [MTP-4] (host metadata in the MTP lane: query starts exact,
sequence lengths exact for prompt chunks and -1 for every request that may
carry an optimistic upper bound), the MTP-lane gate of every change
(speculative_config.method == "mtp", Qwen4Exp MTP drafter, 1 <= k <= 7,
standard rejection, the Qwen3.8 FP16 TP4 contract, SM70) and the
SX_OPT_QSA_MTP_* switch parsing.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402
import torch  # noqa: E402


def qsl(*values):
    return torch.tensor(values, dtype=torch.int32)


def qwen38_model_config(**overrides):
    hf_text_config = SimpleNamespace(
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
    config = SimpleNamespace(
        hf_text_config=hf_text_config,
        architectures=["Qwen4ExpForCausalLM"],
        multimodal_config=None,
        dtype=torch.float16,
        enforce_eager=False,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def mtp_vllm_config(
    method="mtp",
    k=4,
    tp=4,
    parallel_drafting=False,
    rejection="standard",
    qwen4_exp_mtp=True,
    model_config=None,
    spec=True,
    max_num_seqs=24,
    capture_sizes=(5, 10, 15, 20, 30, 40, 60, 80),
    max_capture=80,
    enforce_eager=False,
):
    model_config = model_config or qwen38_model_config(enforce_eager=enforce_eager)
    speculative_config = None
    if spec:
        speculative_config = SimpleNamespace(
            method=method,
            num_speculative_tokens=k,
            parallel_drafting=parallel_drafting,
            rejection_sample_method=rejection,
            use_qwen4_exp_mtp=lambda: bool(qwen4_exp_mtp and method == "mtp"),
            target_model_config=model_config,
        )
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1),
        compilation_config=SimpleNamespace(
            cudagraph_mode=None,
            max_cudagraph_capture_size=max_capture,
            cudagraph_capture_sizes=list(capture_sizes),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=8192
        ),
    )


@pytest.fixture
def sm70(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model
    from vllm.platforms import current_platform

    def is_capability(capability, *args, **kwargs):
        return capability in (70, (7, 0))

    monkeypatch.setattr(current_platform, "is_device_capability", is_capability)
    monkeypatch.setattr(qsa_model.current_platform, "is_device_capability", is_capability)
    ops = C.new_ops()
    monkeypatch.setattr(ops.current_platform, "is_device_capability", is_capability)
    return qsa_model


def qsa_model_module():
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    return qsa_model


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
BAD_CONFIGS = {
    "no-spec": dict(spec=False),
    "dflash": dict(method="dflash"),
    "eagle": dict(method="eagle"),
    "k0": dict(k=0),
    "k8": dict(k=8),
    "tp2": dict(tp=2),
    "parallel-drafting": dict(parallel_drafting=True),
    "synthetic-rejection": dict(rejection="synthetic"),
    "not-qwen4exp-drafter": dict(qwen4_exp_mtp=False),
    "bf16": dict(model_config=qwen38_model_config(dtype=torch.bfloat16)),
    "other-arch": dict(model_config=qwen38_model_config(architectures=["LlamaForCausalLM"])),
}


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 7])
def test_contract_admits_mtp_lane(sm70, k):
    config = mtp_vllm_config(k=k)
    assert sm70._sx_qsa_mtp_lane_contract(config)
    assert sm70._sx_qsa_local_mtp_lane_contract(config)
    assert sm70._sx_qsa_mtp_verify_rows(config) == k + 1


@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_contract_rejects(sm70, name):
    config = mtp_vllm_config(**BAD_CONFIGS[name])
    assert not sm70._sx_qsa_mtp_lane_contract(config)
    assert not sm70._sx_qsa_local_mtp_lane_contract(config)


def test_contract_requires_sm70(monkeypatch):
    qsa_model = qsa_model_module()
    monkeypatch.setattr(
        qsa_model.current_platform, "is_device_capability", lambda *a, **k: False
    )
    assert not qsa_model._sx_qsa_mtp_lane_contract(mtp_vllm_config())


def test_contract_honours_mtp_lane_master_switch(sm70, monkeypatch):
    try:
        from vllm.config.vllm import _is_sm70_qwen38_mtp_lane_contract  # noqa: F401
    except ImportError:
        pytest.skip("vllm/config/vllm.py has no shared MTP-lane contract")
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    assert not sm70._sx_qsa_mtp_lane_contract(mtp_vllm_config())


@pytest.fixture
def no_shared_contract(monkeypatch):
    """Simulate an image whose vllm/config/vllm.py predates the shared
    MTP-lane contract (only the QSA files overlaid): the import in
    _sx_qsa_mtp_lane_contract fails and the local fallback decides."""
    import vllm.config.vllm as config_module

    monkeypatch.delattr(
        config_module, "_is_sm70_qwen38_mtp_lane_contract", raising=False
    )
    return monkeypatch


def test_fallback_contract_parity(sm70, no_shared_contract):
    config = mtp_vllm_config()
    assert sm70._sx_qsa_mtp_lane_contract(config)  # fallback admits the lane
    # SX_OPT_MTP_LANE=0 must disable the QSA MTP-lane items with the fallback
    # too (review fix: the fallback ignored the master switch).
    no_shared_contract.setenv("SX_OPT_MTP_LANE", "0")
    assert not sm70._sx_qsa_mtp_lane_contract(config)
    assert not sm70._sx_qsa_local_mtp_lane_contract(config)
    no_shared_contract.setenv("SX_OPT_MTP_LANE", " 0 ")
    assert not sm70._sx_qsa_local_mtp_lane_contract(config)
    no_shared_contract.setenv("SX_OPT_MTP_LANE", "1")
    assert sm70._sx_qsa_local_mtp_lane_contract(config)
    # Tree verify (more state tokens than drafts) is not the sequential lane.
    config.speculative_config.num_speculative_state_tokens = lambda: 8
    assert not sm70._sx_qsa_mtp_lane_contract(config)
    assert not sm70._sx_qsa_local_mtp_lane_contract(config)
    config.speculative_config.num_speculative_state_tokens = lambda: 4
    assert sm70._sx_qsa_local_mtp_lane_contract(config)


@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_fallback_contract_rejects(sm70, no_shared_contract, name):
    assert not sm70._sx_qsa_mtp_lane_contract(mtp_vllm_config(**BAD_CONFIGS[name]))


def test_draft_model_config_resolves_through_target(sm70):
    """The MTP layer may see the draft model config (qwen4_exp_mtp) as
    vllm_config.model_config; the contract is judged on the target."""
    config = mtp_vllm_config()
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(), architectures=["Qwen4ExpMTP"], dtype=torch.float16
    )
    assert sm70._sx_qsa_mtp_lane_contract(config)


# ---------------------------------------------------------------------------
# Options and graph rows
# ---------------------------------------------------------------------------
MTP_ENVS = (
    "SX_OPT_QSA_MTP_DECODE_ROWS",
    "SX_OPT_QSA_MTP_DECODE_MAX_ROWS",
    "SX_OPT_QSA_MTP_TWO_WARP_MAX_ROWS",
    "SX_OPT_QSA_MTP_RESOLVED_MAX_ROWS",
    "SX_OPT_QSA_MTP_TOPK_MAX_ROWS",
    "SX_OPT_QSA_MTP_PAGE4_GRAPH_ROWS",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in MTP_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_lane_options_defaults(clean_env):
    ops = C.new_ops()
    assert ops.sx_qsa_mtp_lane_options(120) == ops.SxQsaMtpLane(63, 63, 63, 120)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"SX_OPT_QSA_MTP_DECODE_MAX_ROWS": "48"}, (48, 48, 48)),
        ({"SX_OPT_QSA_MTP_DECODE_MAX_ROWS": "100"}, (63, 63, 63)),
        ({"SX_OPT_QSA_MTP_DECODE_MAX_ROWS": "4"}, (16, 16, 16)),
        ({"SX_OPT_QSA_MTP_DECODE_MAX_ROWS": "x"}, (63, 63, 63)),
        ({"SX_OPT_QSA_MTP_TWO_WARP_MAX_ROWS": "40"}, (40, 63, 63)),
        (
            {"SX_OPT_QSA_MTP_DECODE_MAX_ROWS": "50", "SX_OPT_QSA_MTP_TOPK_MAX_ROWS": "63"},
            (50, 50, 63),
        ),
        ({"SX_OPT_QSA_MTP_DECODE_ROWS": "0"}, (32, 32, 32)),
        (
            {"SX_OPT_QSA_MTP_DECODE_ROWS": "0", "SX_OPT_QSA_MTP_TOPK_MAX_ROWS": "63"},
            (32, 32, 32),
        ),
    ],
)
def test_lane_options_env(clean_env, env, expected):
    ops = C.new_ops()
    for name, value in env.items():
        clean_env.setenv(name, value)
    options = ops.sx_qsa_mtp_lane_options(80)
    assert (
        options.two_warp_max_rows,
        options.resolved_max_rows,
        options.topk_max_rows,
    ) == expected
    assert options.page4_graph_rows == 80


def test_lane_options_page4_switch(clean_env):
    ops = C.new_ops()
    with C.patched(ops, _SX_OPT_QSA_MTP_PAGE4_CAPTURE=False):
        assert ops.sx_qsa_mtp_lane_options(120).page4_graph_rows == 0


def test_page4_graph_rows(clean_env):
    qsa_model = qsa_model_module()
    rows = qsa_model._sx_qsa_mtp_page4_graph_rows
    assert rows(mtp_vllm_config()) == 120  # 24 x (4 + 1) > listed max 80
    assert rows(mtp_vllm_config(max_num_seqs=8)) == 80  # listed max
    assert rows(mtp_vllm_config(k=7, max_num_seqs=24)) == 192
    assert rows(mtp_vllm_config(enforce_eager=True)) == 0
    config = mtp_vllm_config()
    config.compilation_config.cudagraph_mode = SimpleNamespace(
        has_full_cudagraphs=lambda: False
    )
    assert rows(config) == 0
    config = mtp_vllm_config(max_num_seqs=4000)
    assert rows(config) == 8192  # capped by max_num_batched_tokens
    clean_env.setenv("SX_OPT_QSA_MTP_PAGE4_GRAPH_ROWS", "64")
    assert rows(mtp_vllm_config()) == 64
    clean_env.setenv("SX_OPT_QSA_MTP_PAGE4_GRAPH_ROWS", "0")
    assert rows(mtp_vllm_config()) == 0


def test_call_signatures_carry_the_lane():
    qsa_model = qsa_model_module()
    ops = C.new_ops()
    from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer

    for fn in (
        ops.qsa_sparse_paged_attention,
        ops.qsa_select_paged_tokens,
        QSAIndexer.forward,
        qsa_model.Qwen4ExpQSAFlashAttentionImpl.forward_qsa,
    ):
        parameter = inspect.signature(fn).parameters.get("sx_mtp_lane")
        assert parameter is not None and parameter.default is None, fn


# ---------------------------------------------------------------------------
# Decode-row gates (MTP-9)
# ---------------------------------------------------------------------------
def test_decode_row_gates(sm70):
    ops = C.new_ops()
    lane = C.mtp_lane()
    two_warp = ops._use_sm70_qsa_two_warp_partial
    assert two_warp(32, 6, 256) and not two_warp(33, 6, 256)
    for rows in (33, 40, 48, 63):
        assert two_warp(rows, 6, 256, max_rows=lane.two_warp_max_rows)
    assert not two_warp(64, 6, 256, max_rows=63)
    assert not two_warp(40, 8, 256, max_rows=63)  # other geometry
    with C.patched(ops, _SX_OPT_QSA_TWO_WARP32=False):
        assert not two_warp(17, 6, 256, max_rows=63)  # switch keeps M <= 16
        assert two_warp(16, 6, 256, max_rows=63)

    resolved = ops._use_sm70_qsa_resolved_indices
    for rows in (1, 32, 33, 48, 63, 64):
        q = torch.empty(rows, 6, 256, dtype=torch.float16)
        k = torch.empty(1000, 784, 1, 256, dtype=torch.float16)
        indices = torch.empty(rows, 2051, dtype=torch.int32)
        assert resolved(q, k, indices, "auto") == (rows <= 32)
        assert resolved(q, k, indices, "auto", max_rows=63) == (rows <= 63)
        assert not resolved(q, k, indices, "fp8_e4m3", max_rows=63)
    with C.patched(ops, _SX_OPT_QSA_RESOLVED_ROWS=False):
        q = torch.empty(40, 6, 256, dtype=torch.float16)
        k = torch.empty(1000, 784, 1, 256, dtype=torch.float16)
        indices = torch.empty(40, 2051, dtype=torch.int32)
        assert not resolved(q, k, indices, "auto", max_rows=63)

    sentinel = object()
    with C.patched(ops, _sm70_qsa_lexicographic_topk_rows_op=lambda: sentinel):
        assert ops._sx_qsa_topk_rows_op(32) is sentinel
        assert ops._sx_qsa_topk_rows_op(33) is None
        assert ops._sx_qsa_topk_rows_op(1, 63) is None
        assert ops._sx_qsa_topk_rows_op(63, 63) is sentinel
        assert ops._sx_qsa_topk_rows_op(64, 63) is None
        with C.patched(ops, _SX_OPT_QSA_TOPK_ROWS=False):
            assert ops._sx_qsa_topk_rows_op(40, 63) is None


# ---------------------------------------------------------------------------
# Host metadata builder (MTP-4)
# ---------------------------------------------------------------------------
def _builder(qsa_model, vllm_config, monkeypatch):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

    monkeypatch.setattr(
        FlashAttentionMetadataBuilder,
        "build",
        lambda self, common_prefix_len, common, fast_build=False: SimpleNamespace(),
    )
    builder = object.__new__(qsa_model.Qwen4ExpQSAMetadataBuilder)
    builder.vllm_config = vllm_config
    return builder


def _mixed_common():
    # 3 verify requests (k=4: 5 rows), a decode without drafts (1 row), a
    # 900-row prompt chunk, a 6-row prompt tail, then a padded request slot.
    starts = qsl(0, 5, 10, 15, 16, 916, 922, 922)
    upper = qsl(2003, 3004, 4005, 5000, 8192, 70, 0)
    return SimpleNamespace(
        query_start_loc_cpu=starts,
        seq_lens_cpu_upper_bound=upper,
        num_reqs=7,
    )


def test_builder_mtp_lane_masks_optimistic_lengths(sm70, monkeypatch):
    common = _mixed_common()
    original = common.seq_lens_cpu_upper_bound.clone()
    builder = _builder(sm70, mtp_vllm_config(k=4), monkeypatch)
    starts, seq_lens = sm70._sx_qsa_host_metadata(builder.build(0, common))
    assert starts.tolist() == [0, 5, 10, 15, 16, 916, 922, 922]
    assert seq_lens.tolist() == [-1, -1, -1, -1, 8192, 70, -1]
    # The runner's buffer is never modified.
    assert torch.equal(common.seq_lens_cpu_upper_bound, original)
    assert seq_lens.device.type == "cpu" and starts.device.type == "cpu"


def test_builder_k1_threshold(sm70, monkeypatch):
    common = SimpleNamespace(
        query_start_loc_cpu=qsl(0, 2, 5, 6),
        seq_lens_cpu_upper_bound=qsl(100, 200, 300),
        num_reqs=3,
    )
    builder = _builder(sm70, mtp_vllm_config(k=1), monkeypatch)
    _, seq_lens = sm70._sx_qsa_host_metadata(builder.build(0, common))
    assert seq_lens.tolist() == [-1, 200, -1]  # 3 rows > 1 + k is a chunk


def test_builder_no_spec_unchanged(sm70, monkeypatch):
    common = _mixed_common()
    builder = _builder(sm70, mtp_vllm_config(spec=False), monkeypatch)
    starts, seq_lens = sm70._sx_qsa_host_metadata(builder.build(0, common))
    assert starts.tolist() == [0, 5, 10, 15, 16, 916, 922, 922]
    assert seq_lens.tolist() == [2003, 3004, 4005, 5000, 8192, 70, 0]
    # Views of the runner buffers, exactly as in 1.8.0-dev2.
    assert seq_lens.data_ptr() == common.seq_lens_cpu_upper_bound.data_ptr()


@pytest.mark.parametrize("name", ["dflash", "tp2", "parallel-drafting", "k8"])
def test_builder_other_spec_gets_nothing(sm70, monkeypatch, name):
    builder = _builder(sm70, mtp_vllm_config(**BAD_CONFIGS[name]), monkeypatch)
    assert sm70._sx_qsa_host_metadata(builder.build(0, _mixed_common())) == (None, None)


def test_builder_switches(sm70, monkeypatch):
    monkeypatch.setattr(sm70, "_SX_OPT_QSA_MTP_HOST_METADATA", False)
    builder = _builder(sm70, mtp_vllm_config(), monkeypatch)
    assert sm70._sx_qsa_host_metadata(builder.build(0, _mixed_common())) == (None, None)
    # The no-MTP lane is unaffected by the MTP switch.
    builder = _builder(sm70, mtp_vllm_config(spec=False), monkeypatch)
    assert sm70._sx_qsa_host_metadata(builder.build(0, _mixed_common()))[1] is not None
    monkeypatch.setattr(sm70, "_SX_OPT_QSA_MTP_HOST_METADATA", True)
    monkeypatch.setattr(sm70, "_SX_OPT_QSA_HOST_METADATA", False)
    builder = _builder(sm70, mtp_vllm_config(), monkeypatch)
    assert sm70._sx_qsa_host_metadata(builder.build(0, _mixed_common())) == (None, None)


def test_builder_draft_decode_metadata_gets_nothing(sm70, monkeypatch):
    """Draft decode steps build metadata without seq_lens_cpu_upper_bound."""
    builder = _builder(sm70, mtp_vllm_config(), monkeypatch)
    common = SimpleNamespace(
        query_start_loc_cpu=qsl(0, 1, 2, 3), seq_lens_cpu_upper_bound=None, num_reqs=3
    )
    assert sm70._sx_qsa_host_metadata(builder.build(0, common)) == (None, None)


# ---------------------------------------------------------------------------
# Planners with masked lengths
# ---------------------------------------------------------------------------
def test_host_bound_rejects_masked_length():
    ops = C.new_ops()
    table = torch.zeros(1, 168, dtype=torch.int32)
    fn = ops._qsa_host_max_visible
    assert fn(table, 784, 4, qsl(0, 784), qsl(8192)) == 2048
    assert fn(table, 784, 4, qsl(0, 784), qsl(-1)) is None
    assert fn(table, 5, 4, qsl(0, 5), qsl(-1)) is None


def test_cublas_planner_with_masked_verify_lengths(sm70, monkeypatch):
    ops = C.new_ops()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    rows = 8 * 5 + 1568
    q = torch.empty(rows, 4, 128, dtype=torch.float16)
    cache = torch.empty(8, 196, 1, 128, dtype=torch.float16)
    table = torch.zeros(9, 168, dtype=torch.int32)
    starts = qsl(*range(0, 41, 5), rows)
    exact = torch.tensor([2000] * 8 + [8192], dtype=torch.int32)
    masked = torch.tensor([-1] * 8 + [8192], dtype=torch.int32)
    plan = ops._plan_qsa_indexer_cublas_segments
    expected = [(0, 40, None, 0), (40, rows, 8, 2048)]
    assert plan(q, cache, table, starts, exact, 168 * 196, 512, 4) == expected
    assert plan(q, cache, table, starts, masked, 168 * 196, 512, 4) == expected
    # Were verify requests ever admitted (tiny MIN_ROWS), the mask keeps the
    # whole batch on the device path instead of trusting an optimistic width.
    optimistic = torch.tensor([6000] * 8 + [8192], dtype=torch.int32)
    with C.patched(
        ops,
        _SM70_INDEXER_CUBLAS_MIN_ROWS=4,
        _SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS=1,
    ):
        assert plan(q, cache, table, starts, masked, 168 * 196, 512, 4) is None
        unsafe = plan(q, cache, table, starts, optimistic, 168 * 196, 512, 4)
        assert unsafe is not None and unsafe[0][2] == 0  # would trust 6000


def test_page4_segments_for_verify_rows():
    """Mixed groups with 5-row verify requests: verify rows row-wise, the
    prompt chunk in whole groups of its own; old groups mix >= 3 requests."""
    ops = C.new_ops()
    rows = 8 * 5 + 784
    starts = qsl(*range(0, 41, 5), rows)
    segments = ops._plan_qsa_page4_request_segments(starts, rows)
    assert segments == [(0, 40, False), (40, rows, True)]
    assert not ops._qsa_page4_segments_match_baseline(segments, rows)
    assert ops._qsa_page4_old_group_max_requests(starts, rows) >= 3


def test_exact_prefill_seq_lens_property():
    """Random batches: prompt/recompute chunks (> 1 + k rows) keep their
    exact host length; every other request (verify rows, draft-less decodes,
    short chunks, padded slots) is -1; the inputs are never modified."""
    qsa_model = qsa_model_module()
    generator = torch.Generator().manual_seed(1234)

    def draw(low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=generator))

    for trial in range(200):
        k = draw(1, 8)
        rows = []
        for _ in range(draw(1, 25)):
            kind = draw(0, 4)
            if kind == 0:  # verify request with 0..k drafts
                rows.append(draw(1, k + 2))
            elif kind == 1:  # prompt / recompute chunk
                rows.append(draw(1, 8000))
            elif kind == 2:  # chunk right at the threshold
                rows.append(k + 1 + draw(0, 2))
            else:  # padded request slot
                rows.append(0)
        starts = [0]
        for count in rows:
            starts.append(starts[-1] + count)
        upper = [count + draw(0, 9000) for count in rows]
        starts_t = torch.tensor(starts, dtype=torch.int32)
        upper_t = torch.tensor(upper, dtype=torch.int32)
        starts_copy, upper_copy = starts_t.clone(), upper_t.clone()
        masked = qsa_model._sx_qsa_exact_prefill_seq_lens(starts_t, upper_t, k + 1)
        expected = [
            seq_len if count > k + 1 else -1 for count, seq_len in zip(rows, upper)
        ]
        assert masked.tolist() == expected, (trial, k, rows)
        assert masked.dtype == upper_t.dtype and masked.device.type == "cpu"
        assert torch.equal(starts_t, starts_copy) and torch.equal(upper_t, upper_copy)
        assert masked.data_ptr() != upper_t.data_ptr()


# ---------------------------------------------------------------------------
# Page4 CUDA-graph workspaces (SX_OPT_QSA_MTP_PAGE4_CAPTURE), host logic only:
# CPU tensors, torch.cuda.is_current_stream_capturing monkeypatched and a
# fake Flash-V100 module. The launchers are replaced by stubs that request
# exactly the workspaces the real launchers request during capture.
# ---------------------------------------------------------------------------
@pytest.fixture
def page4_host(monkeypatch):
    ops = C.new_ops()
    state = SimpleNamespace(capturing=False, requests=[], ops=ops)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    # Fresh module state (monkeypatch restores the originals).
    monkeypatch.setattr(ops, "_SM70_QSA_XQA_PAGE4_PARTITION_COUNTS", {})
    monkeypatch.setattr(ops, "_SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES", {})
    monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES", {})
    monkeypatch.setattr(ops, "_SM70_QSA_PAGE4_GRAPH_RETIRED", [])
    monkeypatch.setattr(ops, "_SM70_QSA_PAGE4_GRAPH_RESERVED", set())
    monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAGE4_ABI_CACHE", None)
    monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAGE4", True)
    monkeypatch.setattr(ops, "_SX_OPT_QSA_MTP_PAGE4_CAPTURE", True)

    def install_flash(grouped_abi: bool) -> None:
        attributes = {"decode_paged_xqa_fwd": object()}
        if grouped_abi:
            attributes.update(
                grouped_sparse_page4_abi_version=lambda: 2,
                grouped_sparse_page4_plan_fwd=lambda *args: None,
                grouped_sparse_page4_fwd=lambda *args: None,
            )
        interface = ModuleType("flash_attn_v100.flash_attn_interface")
        interface.flash_attn_v100_cuda = SimpleNamespace(**attributes)
        package = ModuleType("flash_attn_v100")
        package.flash_attn_interface = interface
        monkeypatch.setitem(sys.modules, "flash_attn_v100", package)
        monkeypatch.setitem(
            sys.modules, "flash_attn_v100.flash_attn_interface", interface
        )
        monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAGE4_ABI_CACHE", None)

    def fake_grouped(q, *args):
        if state.capturing:  # eager workspaces need a CUDA stream
            ops._qsa_grouped_page4_workspace(q)
        state.requests.append(("grouped", q.shape[0]))

    def fake_xqa_batch(q, k_cache, v_cache, logical_indices, *args):
        kv_cache_dtype = args[5]
        partition = 256 if kv_cache_dtype == "fp8_e4m3" else 1024
        partitions = math.ceil(logical_indices.shape[1] / partition)
        if state.capturing:
            ops._qsa_xqa_page4_workspace(q, partitions, kv_cache_dtype)
        state.requests.append(("xqa", q.shape[0]))

    monkeypatch.setattr(
        ops, "_qsa_sparse_paged_attention_sm70_grouped_page4", fake_grouped
    )
    monkeypatch.setattr(
        ops, "_qsa_sparse_paged_attention_sm70_xqa_page4_batch", fake_xqa_batch
    )
    state.install_flash = install_flash
    return state


def _page4_call(ops, rows: int, kv_cache_dtype: str, graph_rows: int) -> None:
    q = torch.zeros(rows, 6, 256, dtype=torch.float16)
    meta = torch.zeros(rows, dtype=torch.int32)
    ops._qsa_sparse_paged_attention_sm70_xqa_page4(
        q,
        torch.empty(0),
        torch.empty(0),
        torch.zeros(rows, 2051, dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.int32),
        meta,
        meta.long(),
        torch.zeros(1, dtype=torch.int32),
        torch.empty_like(q),
        kv_cache_dtype,
        1.0,
        1.0,
        sx_page4_graph_rows=graph_rows,
    )


def _page4_graph_state(ops):
    def ptrs(entries):
        return {
            key: tuple(
                item.data_ptr() if isinstance(item, torch.Tensor) else item
                for item in value
            )
            for key, value in entries.items()
        }

    return (
        ptrs(ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES),
        ptrs(ops._SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES),
        {
            key: value.data_ptr()
            for key, value in ops._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS.items()
        },
        len(ops._SM70_QSA_PAGE4_GRAPH_RETIRED),
    )


@pytest.mark.parametrize("graph_rows", [72, 120, 192, 320])
@pytest.mark.parametrize(
    "grouped_abi,kv_cache_dtype",
    [(True, "auto"), (False, "auto"), (True, "fp8_e4m3"), (False, "fp8_e4m3")],
    ids=["grouped-fp16", "xqa-fp16", "grouped-e4m3", "sliced-e4m3"],
)
def test_page4_reserve_covers_capture_routing(
    page4_host, grouped_abi, kv_cache_dtype, graph_rows
):
    """The reserve mirrors capture-time routing: after ONE eager MTP-lane
    call, capturing any page4 width up to graph_rows (>= 64 rows FP16, > 16
    rows E4M3; grouped + XQA remainder, XQA only, or 16-row E4M3 slices)
    allocates no workspace, retires nothing and creates no constant."""
    ops = page4_host.ops
    page4_host.install_flash(grouped_abi)
    page4_host.capturing = False
    _page4_call(ops, 64, kv_cache_dtype, graph_rows)  # eager: reserves
    reserved = _page4_graph_state(ops)
    assert reserved[2], "partition constant not created outside capture"
    page4_host.capturing = True
    first = 17 if kv_cache_dtype == "fp8_e4m3" else 64
    for rows in range(first, graph_rows + 1):
        page4_host.requests.clear()
        _page4_call(ops, rows, kv_cache_dtype, graph_rows)
        assert sum(count for _, count in page4_host.requests) == rows
        assert _page4_graph_state(ops) == reserved, (rows, page4_host.requests)
    # Beyond the reserve, capture still works (graph-pool allocation). The
    # sliced E4M3 route never needs more than its 16-row XQA workspace.
    _page4_call(ops, graph_rows * 2 + 9, kv_cache_dtype, graph_rows)
    sliced = not grouped_abi and kv_cache_dtype == "fp8_e4m3"
    assert (_page4_graph_state(ops) == reserved) == sliced


def test_page4_reserve_skipped_outside_lane_and_under_capture(page4_host):
    ops = page4_host.ops
    page4_host.install_flash(True)
    page4_host.capturing = False
    _page4_call(ops, 64, "auto", 0)  # no MTP lane: no reserve
    assert _page4_graph_state(ops) == ({}, {}, {}, 0)
    page4_host.capturing = True
    _page4_call(ops, 64, "auto", 120)  # never reserves inside a capture
    assert not ops._SM70_QSA_PAGE4_GRAPH_RESERVED
    page4_host.capturing = False
    with C.patched(ops, _SX_OPT_QSA_MTP_PAGE4_CAPTURE=False):
        _page4_call(ops, 64, "auto", 120)  # switch off: dev2 behaviour
    assert not ops._SM70_QSA_PAGE4_GRAPH_RESERVED


def test_page4_graph_workspace_growth_retires(page4_host):
    """Without the reserve a growing capture keeps the old buffers alive."""
    ops = page4_host.ops
    page4_host.capturing = True
    q = torch.zeros(65, 6, 256, dtype=torch.float16)
    first = ops._qsa_xqa_page4_graph_workspace(q, 3, 65, "auto")
    ((key, entry),) = ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES.items()
    assert entry[0] == 128 and first[0].shape[0] == 65
    old_ptr = entry[1].data_ptr()
    ops._qsa_xqa_page4_graph_workspace(
        torch.zeros(160, 6, 256, dtype=torch.float16), 3, 160, "auto"
    )
    assert ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES[key][0] == 256
    assert any(
        isinstance(item, tuple) and item[1].data_ptr() == old_ptr
        for item in ops._SM70_QSA_PAGE4_GRAPH_RETIRED
    )
    # A smaller later capture reuses the grown workspace (no allocation).
    before = _page4_graph_state(ops)
    ops._qsa_xqa_page4_graph_workspace(q, 3, 65, "auto")
    assert _page4_graph_state(ops) == before
    grouped = ops._qsa_grouped_page4_graph_workspace(q[:64], 8)
    assert [tensor.shape[0] for tensor in grouped] == [8, 8, 8, 64]


def test_partition_count_capture_fallback_is_private(page4_host):
    """A capture that meets no cached constant records a per-call fill that is
    kept alive but never cached: an eager call must not read a tensor whose
    value only exists after that graph has replayed."""
    ops = page4_host.ops
    cpu = torch.device("cpu")
    page4_host.capturing = True
    first = ops._qsa_xqa_page4_partition_count(cpu, 3)
    second = ops._qsa_xqa_page4_partition_count(cpu, 3)
    assert first.tolist() == [3] and second.tolist() == [3]
    assert first.data_ptr() != second.data_ptr()
    assert not ops._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS
    retired = ops._SM70_QSA_PAGE4_GRAPH_RETIRED
    assert sum(item is first or item is second for item in retired) == 2
    page4_host.capturing = False
    cached = ops._qsa_xqa_page4_partition_count(cpu, 3)
    assert ops._qsa_xqa_page4_partition_count(cpu, 3) is cached
    page4_host.capturing = True
    assert ops._qsa_xqa_page4_partition_count(cpu, 3) is cached
