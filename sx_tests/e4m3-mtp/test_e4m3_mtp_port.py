# SPDX-License-Identifier: Apache-2.0
"""E4M3 main KV cache with the native-MTP lane: host logic, no GPU.

Run (CPU is enough, no vLLM install needed; see ``e4m3_boot.py``):

    python -m pytest -q sx_tests/e4m3-mtp/test_e4m3_mtp_port.py

The same stubs run upstream's CPU tests of the port and this fork's page4
workspace tests unchanged (``--noconftest`` keeps tests/conftest.py, which
imports vLLM, out):

    PYTHONPATH=sx_tests/e4m3-mtp python -m pytest -q --noconftest -p e4m3_boot \\
        tests/models/qwen4_exp/test_e4m3_mtp.py \\
        tests/models/qwen4_exp/test_qsa_kv_calibration_mtp.py
    PYTHONPATH=sx_tests/e4m3-mtp python -m pytest -q --noconftest -p e4m3_boot \\
        sx_tests/b3-qsa-mtp/test_mtp_lane_cpu.py \\
        -k "page4_reserve or page4_graph_workspace or partition_count"

Asserted here:
* the XQA page4 scratch is FP32 for E4M3 K/V and FP16 otherwise on all three
  allocation paths (per-stream eager, CUDA-graph workspace, the
  SX_OPT_QSA_MTP_PAGE4_CAPTURE=0 path), both dtypes keep separate buffers, and
  the MTP lane's pre-capture reserve allocates the E4M3 scratch a captured
  launch then finds;
* the grouped page4 planner padding is repointed at the group's first real
  microblock for E4M3 K/V only (FP16 and VLLM_SM70_QSA_GROUPED_PAD_FIX=0 keep
  the planner output);
* the SM70 Qwen3.8 lane admits an E4M3 KV cache only for the MTP lane with
  VLLM_QWEN4EXP_QSA_E4M3_MTP=1, and then applies the lane defaults;
* the drafter's load finalizes its two K/V scales from ``mtp.layers.0.*``,
  fails without them, and is a no-op for an FP16 cache; the target keeps its
  unit-scale fallback and drops the loading slots.
Expected: all pass.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e4m3_boot  # noqa: E402

ops = e4m3_boot.install()
from vllm import envs  # noqa: E402
from vllm.config import vllm as config_vllm  # noqa: E402
from vllm.models.qwen4_exp.nvidia import model as qsa_model  # noqa: E402
from vllm.models.qwen4_exp.nvidia import mtp, qsa  # noqa: E402

E4M3, FP16 = "fp8_e4m3", "auto"
SELECTION = 2051  # token_topk 2048 + compress_ratio 4 - 1
OPT_IN = "VLLM_QWEN4EXP_QSA_E4M3_MTP"
STRICT = "VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES"


def _set_flag(monkeypatch, name: str, enabled: bool) -> None:
    """Set a vllm.envs switch through the environment.

    Other test files pin such switches as module attributes with
    monkeypatch.setattr; drop that pin so the registry is read again.
    """
    monkeypatch.setenv(name, "1" if enabled else "0")
    monkeypatch.delitem(envs.__dict__, name, raising=False)
    assert getattr(envs, name) is enabled


@pytest.fixture
def host(monkeypatch):
    """CPU stand-in for the CUDA stream state the workspaces are keyed by."""
    state = SimpleNamespace(capturing=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )
    for name in (
        "_SM70_QSA_XQA_PAGE4_WORKSPACES",
        "_SM70_QSA_GROUPED_PAGE4_WORKSPACES",
        "_SM70_QSA_XQA_PAGE4_PARTITION_COUNTS",
        "_SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES",
        "_SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES",
    ):
        monkeypatch.setattr(ops, name, {})
    monkeypatch.setattr(ops, "_SM70_QSA_PAGE4_GRAPH_RETIRED", [])
    monkeypatch.setattr(ops, "_SM70_QSA_PAGE4_GRAPH_RESERVED", set())
    monkeypatch.setattr(ops, "_SX_OPT_QSA_MTP_PAGE4_CAPTURE", True)
    return state


def _query(rows: int = 3) -> torch.Tensor:
    return torch.zeros((rows, 6, 256), dtype=torch.float16)


@pytest.mark.parametrize("path", ["eager", "captured", "dev2"])
@pytest.mark.parametrize(
    ("kv_cache_dtype", "expected"),
    [(E4M3, torch.float32), (FP16, torch.float16), ("float16", torch.float16)],
)
def test_xqa_scratch_dtype_follows_kv_dtype(
    host, monkeypatch, path, kv_cache_dtype, expected
):
    host.capturing = path == "captured"
    monkeypatch.setattr(ops, "_SX_OPT_QSA_MTP_PAGE4_CAPTURE", path != "dev2")
    scratch, max_logits, exp_sums, active = ops._qsa_xqa_page4_workspace(
        _query(), 9, kv_cache_dtype
    )
    assert scratch.dtype == expected and scratch.shape == (3, 6, 9, 256)
    assert max_logits.dtype == exp_sums.dtype == torch.float32
    assert active.dtype == torch.int32 and active.tolist() == [9]


@pytest.mark.parametrize("path", ["eager", "captured", "dev2"])
def test_fp16_and_e4m3_scratch_are_separate(host, monkeypatch, path):
    """Same rows and partition count, so only the dtype tells the two apart."""
    host.capturing = path == "captured"
    monkeypatch.setattr(ops, "_SX_OPT_QSA_MTP_PAGE4_CAPTURE", path != "dev2")
    q = _query()
    e4m3 = ops._qsa_xqa_page4_workspace(q, 8, E4M3)[0]
    fp16 = ops._qsa_xqa_page4_workspace(q, 8, FP16)[0]
    assert (e4m3.dtype, fp16.dtype) == (torch.float32, torch.float16)
    assert ops._qsa_xqa_page4_workspace(q, 8, E4M3)[0].data_ptr() == e4m3.data_ptr()
    assert ops._qsa_xqa_page4_workspace(q, 8, FP16)[0].data_ptr() == fp16.data_ptr()
    cache = (
        ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES
        if path == "captured"
        else ops._SM70_QSA_XQA_PAGE4_WORKSPACES
    )
    assert sorted(key[-1] for key in cache) == [False, True]


@pytest.mark.parametrize(
    ("kv_cache_dtype", "grouped", "partitions", "capacity", "dtype"),
    [
        (E4M3, True, 9, 8, torch.float32),  # 7 remainder rows
        (E4M3, False, 9, 16, torch.float32),  # 16-row XQA slices
        (FP16, True, 3, 8, torch.float16),
        (FP16, False, 3, 128, torch.float16),  # every row
    ],
)
def test_reserve_allocates_the_scratch_a_capture_finds(
    host, kv_cache_dtype, grouped, partitions, capacity, dtype
):
    graph_rows = 120  # 24 requests x (1 + 4) verify rows
    ops._qsa_page4_reserve_graph_workspaces(
        _query(64), kv_cache_dtype, SELECTION, grouped, graph_rows
    )
    ((key, entry),) = ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES.items()
    assert key[3:] == (partitions, kv_cache_dtype == E4M3)
    assert entry[0] == capacity and entry[1].dtype == dtype
    assert bool(ops._SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES) == grouped

    host.capturing = True
    rows = min(capacity, 7)
    scratch, _, _, active = ops._qsa_xqa_page4_workspace(
        _query(rows), partitions, kv_cache_dtype
    )
    assert scratch.dtype == dtype and scratch.data_ptr() == entry[1].data_ptr()
    assert active.tolist() == [partitions]
    assert len(ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES) == 1
    assert not ops._SM70_QSA_PAGE4_GRAPH_RETIRED


def _grouped_pages_seen_by_forward(monkeypatch, kv_cache_dtype, pad_fix=True):
    """Run the grouped launch host code; return the pages the kernel is given."""
    real = [(40, 0x0000000F), (41, 0x000000FF), (57, 0x00000003)]
    seen = {}

    def plan(indices, table, token_to_req, positions, lengths, pages, masks, *rest):
        pages.zero_()
        masks.zero_()
        for column, (page, mask) in enumerate(real):
            pages[:, column] = page
            masks[:, column] = mask
        rest[0].fill_(8 * 4)

    def forward(q, k, v, out, pages, masks, *rest):
        seen["pages"] = pages.clone()
        seen["dtype"] = rest[3]

    flash = SimpleNamespace(
        grouped_sparse_page4_abi_version=lambda: 2,
        grouped_sparse_page4_plan_fwd=plan,
        grouped_sparse_page4_fwd=forward,
    )
    monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAGE4_ABI_CACHE", None)
    monkeypatch.setattr(ops, "_SM70_QSA_GROUPED_PAD_FIX", pad_fix)
    q = _query(16)
    cache = torch.zeros(
        (2, 8, 1, 256), dtype=torch.uint8 if kv_cache_dtype == E4M3 else q.dtype
    )
    rows = torch.zeros(16, dtype=torch.int32)
    ops._qsa_sparse_paged_attention_sm70_grouped_page4(
        q,
        cache,
        cache.clone(),
        torch.zeros((16, SELECTION), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
        rows,
        rows.long(),
        torch.zeros(1, dtype=torch.int32),
        torch.empty_like(q),
        kv_cache_dtype,
        0.5,
        0.25,
        flash,
    )
    assert seen["dtype"] == kv_cache_dtype
    return seen["pages"], [page for page, _ in real]


def test_e4m3_padding_is_repointed_at_the_first_real_microblock(host, monkeypatch):
    pages, real = _grouped_pages_seen_by_forward(monkeypatch, E4M3)
    assert pages.shape == (2, ops._SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES)
    assert pages[:, : len(real)].tolist() == [real, real]
    # Nothing points at the null block (microblock 0) any more.
    assert (pages[:, len(real) :] == real[0]).all()


@pytest.mark.parametrize(
    ("kv_cache_dtype", "pad_fix"), [(FP16, True), (E4M3, False)]
)
def test_planner_padding_is_kept_outside_the_e4m3_fix(
    host, monkeypatch, kv_cache_dtype, pad_fix
):
    pages, real = _grouped_pages_seen_by_forward(monkeypatch, kv_cache_dtype, pad_fix)
    assert pages[:, : len(real)].tolist() == [real, real]
    assert (pages[:, len(real) :] == 0).all()


# ---------------------------------------------------------------------------
# Lane admission (vllm/config/vllm.py)
# ---------------------------------------------------------------------------
def _lane_config(cache_dtype: str, mtp_lane: bool, **overrides):
    model_config = SimpleNamespace(
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
        architectures=["Qwen4ExpForCausalLM"],
        multimodal_config=None,
        dtype=torch.float16,
        quantization="modelopt_fp4",
    )
    speculative_config = None
    if mtp_lane:
        speculative_config = SimpleNamespace(
            method="mtp",
            num_speculative_tokens=4,
            num_speculative_state_tokens=lambda: 4,
            parallel_drafting=False,
            rejection_sample_method="standard",
            use_qwen4_exp_mtp=lambda: True,
            target_model_config=model_config,
        )
    config = SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_dbo=False,
            data_parallel_size=1,
            nnodes_within_dp=1,
        ),
        lora_config=None,
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype, mamba_ssm_cache_dtype="auto"
        ),
    )
    config.__dict__.update(overrides)
    return config


@pytest.fixture
def lane_env(monkeypatch):
    """A private os.environ: the lane defaults are written into it."""
    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SX_OPT_", "VLLM_"))
    }
    monkeypatch.setattr(os, "environ", environ)
    _set_flag(monkeypatch, OPT_IN, False)
    return environ


@pytest.mark.parametrize(
    ("cache_dtype", "mtp_lane", "opt_in", "qualified"),
    [
        ("auto", False, False, True),
        ("auto", True, False, True),
        ("float16", True, True, True),
        ("fp8_e4m3", True, False, False),
        ("fp8_e4m3", True, True, True),
        ("fp8", True, True, True),
        ("fp8_e4m3", False, False, False),
        ("fp8_e4m3", False, True, False),  # the opt-in is for the MTP lane only
        ("fp8_e5m2", True, True, False),
        ("bfloat16", True, True, False),
    ],
)
def test_lane_admits_e4m3_only_with_mtp_and_the_opt_in(
    lane_env, cache_dtype, mtp_lane, opt_in, qualified
):
    lane_env[OPT_IN] = str(int(opt_in))
    config = _lane_config(cache_dtype, mtp_lane)
    assert config_vllm._sm70_qwen38_lane_qualified(config, is_sm70=True) is qualified
    assert not config_vllm._sm70_qwen38_lane_qualified(config, is_sm70=False)


def test_unset_switch_keeps_e4m3_with_mtp_rejected(lane_env, monkeypatch):
    monkeypatch.setattr(
        qsa,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda capability: capability == 70),
    )
    config = _lane_config(E4M3, True)
    gate = (config, config.model_config, config.cache_config)
    del lane_env[OPT_IN]
    assert envs.VLLM_QWEN4EXP_QSA_E4M3_MTP is False
    with pytest.raises(NotImplementedError, match="requires MTP0"):
        qsa._verify_e4m3_kv_requirements(*gate)
    assert not config_vllm._sm70_qwen38_lane_qualified(config, is_sm70=True)
    lane_env[OPT_IN] = "1"
    qsa._verify_e4m3_kv_requirements(*gate)


def test_opt_in_does_not_loosen_the_other_lane_checks(lane_env):
    lane_env[OPT_IN] = "1"
    qualified = config_vllm._sm70_qwen38_lane_qualified
    assert qualified(_lane_config(E4M3, True), is_sm70=True)
    config = _lane_config(E4M3, True)
    config.model_config.quantization = "awq"
    assert not qualified(config, is_sm70=True)
    config = _lane_config(E4M3, True)
    config.cache_config.mamba_ssm_cache_dtype = "float16"
    assert not qualified(config, is_sm70=True)
    config = _lane_config(E4M3, True, lora_config=object())
    assert not qualified(config, is_sm70=True)
    lane_env["SX_OPT_MTP_LANE"] = "0"
    assert not qualified(_lane_config(E4M3, True), is_sm70=True)


def test_e4m3_mtp_lane_receives_the_lane_defaults(lane_env):
    apply = config_vllm._apply_sm70_qwen38_nomtp_defaults
    assert apply(_lane_config(E4M3, True), is_sm70=True) == ()
    assert "VLLM_SM70_QWEN38_FP16_GEMV" not in lane_env

    lane_env[OPT_IN] = "1"
    applied = apply(_lane_config(E4M3, True), is_sm70=True)
    e4m3_lane = dict(lane_env)
    assert "VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS" in applied
    assert lane_env["VLLM_SM70_QWEN38_FP16_GEMV"] == "1"

    # The same defaults as the FP16 MTP lane.
    for name in applied:
        del lane_env[name]
    assert apply(_lane_config(FP16, True), is_sm70=True) == applied
    assert dict(lane_env) == e4m3_lane


# ---------------------------------------------------------------------------
# K/V scale loading
# ---------------------------------------------------------------------------
class FakeLoader:
    """AutoWeightsLoader stand-in: copies each weight whose name is a parameter."""

    def __init__(self, module, **kwargs) -> None:
        self.module = module

    def load_weights(self, weights, mapper=None) -> set[str]:
        parameters = dict(self.module.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name in parameters:
                parameters[name].data.copy_(weight)
                loaded.add(name)
        return loaded


def _qsa_owner(kv_cache_dtype: str = E4M3):
    """A QSA owner in the state its constructor leaves the K/V scales in."""
    owner = qsa.Qwen4ExpQSAAttention.__new__(qsa.Qwen4ExpQSAAttention)
    nn.Module.__init__(owner)
    owner.kv_cache_dtype = kv_cache_dtype
    owner.layer_name = "layers.0.self_attn.attn"
    qsa.set_default_quant_scales(owner, register_buffer=True)
    owner._qsa_kv_scales_finalized = kv_cache_dtype not in ("fp8", E4M3)
    if not owner._qsa_kv_scales_finalized:
        owner.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        owner.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
    return owner


def _drafter(monkeypatch, kv_cache_dtype: str = E4M3):
    monkeypatch.setattr(mtp, "AutoWeightsLoader", FakeLoader)
    monkeypatch.setattr(qsa_model, "is_offload_process", lambda: False)
    layer = nn.Module()
    layer.self_attn = _qsa_owner(kv_cache_dtype)
    predictor = nn.Module()
    predictor.layers = nn.ModuleList([layer])
    predictor._kv_cache_dtype = kv_cache_dtype
    predictor.fp8_mtp_checkpoint_prefixes = set()
    predictor.fp8_mtp_tp_size = 4
    predictor.config = SimpleNamespace(num_experts=512)
    drafter = mtp.Qwen4ExpMTP.__new__(mtp.Qwen4ExpMTP)
    nn.Module.__init__(drafter)
    drafter.model = predictor
    return drafter, layer.self_attn


def _checkpoint(*names: str):
    values = {
        "mtp.layers.0.self_attn.k_scale": 0.03125,
        "mtp.layers.0.self_attn.v_scale": 0.0625,
        # A target scale: the drafter must not pick it up.
        "model.language_model.layers.3.self_attn.k_scale": 9.0,
    }
    return [(name, torch.tensor(values[name])) for name in names]


def test_drafter_finalizes_its_two_calibrated_scales(monkeypatch):
    drafter, owner = _drafter(monkeypatch)
    loaded = drafter.load_weights(
        _checkpoint(
            "model.language_model.layers.3.self_attn.k_scale",
            "mtp.layers.0.self_attn.k_scale",
            "mtp.layers.0.self_attn.v_scale",
        )
    )
    assert loaded == {
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
    }
    assert owner._qsa_kv_scales_finalized
    assert (owner._k_scale_float, owner._v_scale_float) == (0.03125, 0.0625)
    assert (float(owner._k_scale), float(owner._v_scale)) == (0.03125, 0.0625)
    assert not hasattr(owner, "k_scale") and not hasattr(owner, "v_scale")


@pytest.mark.parametrize("strict", [False, True])
def test_drafter_fails_without_calibrated_scales(monkeypatch, strict):
    """Unlike the target, the drafter has no unit-scale fallback in any mode."""
    _set_flag(monkeypatch, STRICT, strict)
    drafter, owner = _drafter(monkeypatch)
    with pytest.raises(ValueError, match="draft scale overlay is incomplete.*1/2"):
        drafter.load_weights(_checkpoint("mtp.layers.0.self_attn.k_scale"))
    assert not owner._qsa_kv_scales_finalized


def test_drafter_scale_gate_is_a_noop_for_an_fp16_cache(monkeypatch):
    drafter, owner = _drafter(monkeypatch, "auto")
    assert drafter.load_weights(_checkpoint()) == set()
    assert (owner._k_scale_float, owner._v_scale_float) == (1.0, 1.0)


def test_target_unit_scale_fallback_drops_the_loading_slots(monkeypatch):
    _set_flag(monkeypatch, STRICT, False)
    monkeypatch.setattr(qsa_model, "is_offload_process", lambda: False)
    layer = nn.Module()
    layer.self_attn = owner = _qsa_owner()
    target = nn.Module()
    target.layers = nn.ModuleList([layer])
    qsa_model._finalize_qsa_e4m3_scale_load(target, set(), E4M3)
    assert owner._qsa_kv_scales_finalized
    assert (owner._k_scale_float, owner._v_scale_float) == (1.0, 1.0)
    assert dict(owner.named_parameters()) == {}


def test_upstream_overlay_gate_test(monkeypatch):
    """The upstream test of both halves of the target's scale gate.

    Its file, tests/models/qwen4_exp/test_weight_loading.py, imports the whole
    model stack, so only that one function is run here.
    """
    path = os.path.join(
        e4m3_boot.REPO, "tests", "models", "qwen4_exp", "test_weight_loading.py"
    )
    name = "test_qsa_e4m3_loader_requires_all_24_scales"
    namespace = {
        "pytest": pytest,
        "qwen4_exp_model": qsa_model,
        "_validate_qsa_e4m3_scale_load": qsa_model._validate_qsa_e4m3_scale_load,
    }
    exec(compile(e4m3_boot._cut(path, (name,)), path, "exec"), namespace)  # noqa: S102
    namespace[name](monkeypatch)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
