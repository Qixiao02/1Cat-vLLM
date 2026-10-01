# SPDX-License-Identifier: Apache-2.0
"""b3 lane-core (design_1 MTP-0/MTP-1/MTP-6, design_2 PF2, design_4 MTP-K5):
config, contract and semantics checks for the SM70 Qwen3.8 native-MTP lane.

GPU: none (no CUDA context). Runs inside image 1.8.0-dev2 with the overlay
bind-mounted:

  /opt/venv/bin/python -m pytest -q sx_tests/b3-lane-core/test_mtp_lane_config_cpu.py

Asserts:
  * _is_sm70_qwen38_mtp_lane_contract admits method "mtp" with the Qwen4Exp
    drafter, 1 <= k <= 7, standard rejection, no parallel drafting, on the
    exact Qwen3.8 TP4 target (also when called with the draft model config,
    resolved through speculative_config.target_model_config); rejects
    dflash/eagle3/other MTP drafters, k=0/8, tree state tokens, synthetic
    rejection, parallel drafting, TP2/PP2/other shapes, SX_OPT_MTP_LANE=0,
    and the legacy SimpleNamespace(method="mtp") fixtures of older tests.
  * _apply_sm70_qwen38_nomtp_defaults: no-MTP lane unchanged (5 defaults, no
    split flag); MTP lane gets the same 5 + VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS
    + the upstream d30469863 port defaults (PORT_KEYS, each only when the
    loaded _C carries its op and its SX_OPT_MTP_* switch is on), explicit
    overrides win, unqualified deployments and SX_OPT_MTP_LANE=0 get nothing.
    The _C capability probes are pinned (absent by default) so the expected
    lists do not depend on the image's _C.
  * MTP-K5 verify widths B*(k+1) for k in 1..4 and max_num_seqs 1..64,
    SX_OPT_MTP_GRAPH_MAX_REQS / SX_OPT_MTP_GRAPH_REQS, the untouched legacy
    helper (cap 16 for other models), and the auto-list override decision
    (auto split / bounded lists replaced, operator lists and maxima kept);
    widths never exceed max_num_batched_tokens (the widest schedulable B is
    the cap), and re-applying the override to its own output never changes
    it (the worker re-runs VllmConfig.__post_init__ on the shared
    compilation_config via vllm.config.replace() for the MTP drafter).
  * PW-1 sizes in the MTP lane: grid strictly above the largest verify
    width, [] with SX_OPT_MTP_PW=0 / SX_OPT_MTP_LANE=0 / other methods; the
    no-MTP grid unchanged.
  * sm70_fp16_gemv._exact_runtime_contract: MTP lane admitted only with the
    dual-compile lane; no-MTP unchanged; other spec methods rejected.
  * use_sm70_decode_graph_semantics truth table: no-MTP unchanged; MTP lane
    installed -> True for the drafter, False inside the target main
    backbone, True inside FULL decode-graph capture; SX_OPT_MTP_ROWS gate of
    _sx_decode_graph_active; model.py _sx_sm70_qwen38_mtp_lane_installed.
  * Qwen4ExpForCausalLM.forward wiring (stand-in self, no model build): the
    main backbone sees main-compile semantics inside the marker context, the
    decode backbone decode semantics, code after the forward (the drafter)
    decode semantics again, the marker is reset on exceptions, and the
    no-MTP lane calls the main backbone exactly as before.
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
from vllm.compilation import sm70_decode_graph as dg
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode

K_VALUES = (1, 2, 3, 4)
LANE_REQS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)
NOMTP_GRID = [
    32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
    576, 640, 704, 768, 832, 896, 960, 1024,
]  # fmt: skip
DEFAULT_KEYS = (
    "VLLM_SM70_QWEN38_FP16_GEMV",
    "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
    "VLLM_SM70_QWEN38_FUSED_HC_FP16",
    "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP",
    "VLLM_SM70_MOE_ADD_ALLREDUCE",
)
SPLIT_KEY = "VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS"
# MTP-lane defaults of the upstream d30469863 port (_sx_mtp_lane_port_defaults)
# and the _C capability probe each one needs.
PORT_KEYS = ("VLLM_SM70_RMSNORM_GATED_EXACT", "VLLM_SM70_MTP_PLE_CONV")
PORT_PROBES = (
    "_sm70_rmsnorm_gated_exact_available",
    "_sm70_ple_spec_conv_available",
)
PORT_SWITCHES = ("SX_OPT_MTP_RMSNORM_GATED_EXACT", "SX_OPT_MTP_PLE_CONV")
SX_KEYS = (
    "SX_OPT_MTP_LANE",
    "SX_OPT_MTP_ROWS",
    "SX_OPT_MTP_PW",
    "SX_OPT_MTP_PW_DRAFT",
    "SX_OPT_MTP_GRAPH_MAX_REQS",
    "SX_OPT_MTP_GRAPH_REQS",
    "SX_OPT_PIECEWISE_MIXED",
    "SX_OPT_PIECEWISE_MAX_TOKENS",
    "SX_OPT_PIECEWISE_SIZES",
    *PORT_SWITCHES,
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_sx_piecewise_platform_ok", lambda: True)
    for probe in PORT_PROBES:
        monkeypatch.setattr(cfg_mod, probe, lambda: False)
    saved_installed = dg.sm70_mtp_lane_installed()
    with sx_env(
        **{key: None for key in SX_KEYS},
        VLLM_SM70_QWEN38_DUAL_COMPILE="1",
        VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH="1",
        VLLM_SM70_QWEN38_HYBRID_PLE="0",
        VLLM_PLE_CPU_OFFLOAD="0",
    ):
        dg.set_sm70_mtp_lane_installed(False)
        try:
            yield
        finally:
            dg.set_sm70_mtp_lane_installed(saved_installed)


def text_config(**overrides):
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


def model_config(arch="Qwen4ExpForConditionalGeneration", **overrides):
    return SimpleNamespace(
        architectures=[arch],
        multimodal_config=SimpleNamespace(language_model_only=True),
        dtype=torch.float16,
        hf_text_config=text_config(**overrides),
        quantization="modelopt_fp4",
    )


TARGET = model_config()


def mtp_spec(k=4, method="mtp", qwen4=True, target=TARGET, state_tokens=None):
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        parallel_drafting=False,
        rejection_sample_method="standard",
        target_model_config=target,
    )
    spec.use_qwen4_exp_mtp = lambda: bool(qwen4 and method == "mtp")
    spec.num_speculative_state_tokens = lambda: k if state_tokens is None else state_tokens
    return spec


def parallel(tp=4, pp=1):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        enable_expert_parallel=False,
        enable_dbo=False,
        data_parallel_size=1,
        nnodes_within_dp=1,
    )


def serving_config(spec=None, **overrides):
    cfg = SimpleNamespace(
        model_config=model_config(),
        speculative_config=spec,
        parallel_config=parallel(),
        lora_config=None,
        cache_config=SimpleNamespace(
            cache_dtype="float16", mamba_ssm_cache_dtype="float32"
        ),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def lane_ok(spec, model=None, par=None):
    return cfg_mod._is_sm70_qwen38_mtp_lane_contract(
        model if model is not None else TARGET, spec, par or parallel()
    )


# ----------------------------------------------------------------------------
# contract
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("k", range(1, 8))
def test_contract_admits_k(k):
    assert lane_ok(mtp_spec(k=k))


def test_contract_resolves_draft_config_through_target():
    draft = model_config(arch="Qwen4ExpMTP")
    assert lane_ok(mtp_spec(), model=draft)
    assert not lane_ok(mtp_spec(target=None), model=draft)


def _rejection_cases():
    synthetic = mtp_spec()
    synthetic.rejection_sample_method = "synthetic"
    parallel_drafting = mtp_spec()
    parallel_drafting.parallel_drafting = True
    return [
        ("none", None, {}),
        ("k0", mtp_spec(k=0), {}),
        ("k8", mtp_spec(k=8), {}),
        ("dflash", mtp_spec(method="dflash"), {}),
        ("eagle3", mtp_spec(method="eagle3"), {}),
        ("other_mtp_drafter", mtp_spec(qwen4=False), {}),
        ("tree_state_tokens", mtp_spec(k=4, state_tokens=6), {}),
        ("synthetic", synthetic, {}),
        ("parallel_drafting", parallel_drafting, {}),
        ("legacy_fixture", SimpleNamespace(method="mtp", num_speculative_tokens=1), {}),
        ("tp2", mtp_spec(), dict(par=parallel(tp=2))),
        ("pp2", mtp_spec(), dict(par=parallel(pp=2))),
        (
            "hidden4096",
            mtp_spec(target=model_config(hidden_size=4096)),
            dict(model=model_config(hidden_size=4096)),
        ),
    ]


@pytest.mark.parametrize("label, spec, kwargs", _rejection_cases())
def test_contract_rejections(label, spec, kwargs):
    assert not lane_ok(spec, **kwargs), label


def test_contract_master_switch():
    with sx_env(SX_OPT_MTP_LANE="0"):
        assert not lane_ok(mtp_spec())
    assert lane_ok(mtp_spec())


def test_lane_contract_union():
    par = parallel()
    assert cfg_mod._is_sm70_qwen38_lane_contract(TARGET, None, par)
    assert cfg_mod._is_sm70_qwen38_lane_contract(TARGET, mtp_spec(), par)
    assert not cfg_mod._is_sm70_qwen38_lane_contract(TARGET, mtp_spec(method="dflash"), par)
    # The no-MTP contract itself is unchanged: any speculative config rejects.
    assert not cfg_mod._is_sm70_qwen38_nomtp_dual_compile_contract(
        TARGET, mtp_spec(), par
    )


# ----------------------------------------------------------------------------
# auto defaults
# ----------------------------------------------------------------------------
def test_defaults_nomtp_unchanged():
    with sx_env(**{key: None for key in (*DEFAULT_KEYS, SPLIT_KEY, *PORT_KEYS)}):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(), is_sm70=True
        )
        assert applied == DEFAULT_KEYS
        assert SPLIT_KEY not in os.environ
    with sx_env(
        **{key: None for key in (*DEFAULT_KEYS, SPLIT_KEY, *PORT_KEYS)},
        SX_OPT_MTP_LANE="0",
    ):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(), is_sm70=True
        )
        assert applied == DEFAULT_KEYS  # the MTP switch never touches no-MTP


@pytest.mark.parametrize("k", K_VALUES)
def test_defaults_mtp_lane(k):
    with sx_env(**{key: None for key in (*DEFAULT_KEYS, SPLIT_KEY, *PORT_KEYS)}):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(mtp_spec(k=k)), is_sm70=True
        )
        assert applied == (*DEFAULT_KEYS, SPLIT_KEY)
        assert all(os.environ[key] == "1" for key in applied)
        assert cfg_mod._sm70_qwen38_lane_qualified(
            serving_config(mtp_spec(k=k)), is_sm70=True
        )
    # Explicit overrides win.
    # Validation fix (b3a): DEFAULT_KEYS already holds the FP16_GEMV key, so
    # merge into one dict instead of passing it twice as a keyword.
    with sx_env(
        **{
            **{key: None for key in DEFAULT_KEYS},
            "VLLM_SM70_QWEN38_FP16_GEMV": "0",
            "VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS": "0",
        }
    ):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(mtp_spec(k=k)), is_sm70=True
        )
        assert "VLLM_SM70_QWEN38_FP16_GEMV" not in applied
        assert SPLIT_KEY not in applied
        assert os.environ["VLLM_SM70_QWEN38_FP16_GEMV"] == "0"
        assert os.environ[SPLIT_KEY] == "0"


@pytest.mark.parametrize(
    "label, spec, overrides, is_sm70, env",
    [
        ("switch_off", mtp_spec(), {}, True, dict(SX_OPT_MTP_LANE="0")),
        ("not_sm70", mtp_spec(), {}, False, {}),
        ("dflash", mtp_spec(method="dflash"), {}, True, {}),
        ("lora", mtp_spec(), dict(lora_config=SimpleNamespace()), True, {}),
        (
            "fp8_kv",
            mtp_spec(),
            dict(cache_config=SimpleNamespace(cache_dtype="fp8_e4m3", mamba_ssm_cache_dtype="float32")),
            True,
            {},
        ),
    ],
)
def test_defaults_mtp_rejections(label, spec, overrides, is_sm70, env):
    with sx_env(**{key: None for key in (*DEFAULT_KEYS, SPLIT_KEY)}, **env):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(spec, **overrides), is_sm70=is_sm70
        )
        assert applied == (), label
        assert not any(key in os.environ for key in (*DEFAULT_KEYS, SPLIT_KEY)), label


def test_defaults_mtp_quantization_rejected():
    cfg = serving_config(mtp_spec())
    cfg.model_config.quantization = "awq"
    with sx_env(**{key: None for key in (*DEFAULT_KEYS, SPLIT_KEY)}):
        assert cfg_mod._apply_sm70_qwen38_nomtp_defaults(cfg, is_sm70=True) == ()


@pytest.mark.parametrize("k", K_VALUES)
def test_defaults_mtp_lane_port(monkeypatch, k):
    """Upstream d30469863 port: lane-only defaults, own switches, _C probes."""
    for probe in PORT_PROBES:
        monkeypatch.setattr(cfg_mod, probe, lambda: True)
    keys = (*DEFAULT_KEYS, SPLIT_KEY, *PORT_KEYS)
    with sx_env(**{key: None for key in keys}):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(mtp_spec(k=k)), is_sm70=True
        )
        assert applied == (*DEFAULT_KEYS, SPLIT_KEY, *PORT_KEYS)
        assert all(os.environ[key] == "1" for key in PORT_KEYS)
    for key, switch in zip(PORT_KEYS, PORT_SWITCHES, strict=True):
        with sx_env(**{key: None for key in keys}, **{switch: "0"}):
            applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
                serving_config(mtp_spec(k=k)), is_sm70=True
            )
            assert key not in applied and key not in os.environ
        with sx_env(**{**{name: None for name in keys}, key: "0"}):
            applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
                serving_config(mtp_spec(k=k)), is_sm70=True
            )
            assert key not in applied and os.environ[key] == "0"
    # The no-MTP lane keeps exactly its previous list: the exact gated norm
    # (already a no-MTP default when _C has it) and nothing else of the port.
    with sx_env(**{key: None for key in keys}):
        applied = cfg_mod._apply_sm70_qwen38_nomtp_defaults(
            serving_config(), is_sm70=True
        )
        assert applied == (*DEFAULT_KEYS, "VLLM_SM70_RMSNORM_GATED_EXACT")


# ----------------------------------------------------------------------------
# MTP-K5 capture sizes
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("max_num_seqs", [1, 2, 3, 4, 5, 8, 12, 16, 17, 20, 23, 24, 32, 64])
def test_lane_capture_sizes(k, max_num_seqs):
    q = k + 1
    sizes = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(max_num_seqs, q)
    cap = min(max_num_seqs, 24)
    reqs = sorted({r for r in LANE_REQS if r <= cap} | {1, cap})
    assert sizes == [q * r for r in reqs]
    assert all(size % q == 0 for size in sizes)
    # Every request count 1..cap has a FULL verify graph (no eager verify).
    for b in range(1, cap + 1):
        assert any(size >= b * q for size in sizes), (k, max_num_seqs, b)


def test_lane_capture_sizes_production_table():
    table = {
        k: cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(24, k + 1) for k in K_VALUES
    }
    print("\n[MTP-K5] verify widths at max_num_seqs=24:", table)
    assert table[4] == [5, 10, 15, 20, 30, 40, 60, 80, 100, 120]
    assert table[1] == [2, 4, 6, 8, 12, 16, 24, 32, 40, 48]


def test_lane_capture_sizes_env():
    f = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes
    with sx_env(SX_OPT_MTP_GRAPH_MAX_REQS="16"):
        assert f(24, 5) == [5, 10, 15, 20, 30, 40, 60, 80]  # the previous cap
    with sx_env(SX_OPT_MTP_GRAPH_REQS="1,2,4,8,12,16,20,24"):
        assert f(24, 5) == [5, 10, 20, 40, 60, 80, 100, 120]
    with sx_env(SX_OPT_MTP_GRAPH_REQS="2,4"):
        assert f(24, 5) == [5, 10, 20, 120]  # B=1 and the cap are always kept
    with sx_env(SX_OPT_MTP_GRAPH_REQS="junk,,0,999"):
        assert f(24, 5) == [5 * r for r in LANE_REQS]
    with sx_env(SX_OPT_MTP_GRAPH_MAX_REQS="junk"):
        assert f(24, 5)[-1] == 120
    with sx_env(SX_OPT_MTP_GRAPH_MAX_REQS="999"):
        assert f(100, 5)[-1] == 5 * 64


def test_legacy_helpers_unchanged():
    # Other SM70 MTP models keep the previous shapes (cap 16).
    assert cfg_mod._sm70_mtp_cudagraph_capture_sizes(32, 5) == [
        5, 10, 15, 20, 30, 40, 60, 80,
    ]  # fmt: skip
    assert cfg_mod._sm70_speculative_cudagraph_capture_sizes(24, 5) == [
        1, 2, 4, 5, 8, 9, 10, 15, 18, 20, 30, 40, 60, 80,
    ]  # fmt: skip


def _capture_config(sizes, max_size, k=4, max_num_seqs=24):
    return SimpleNamespace(
        speculative_config=mtp_spec(k=k),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=None if sizes is None else list(sizes),
            max_cudagraph_capture_size=max_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


@pytest.mark.parametrize("k", K_VALUES)
def test_capture_size_override(k):
    q = k + 1
    ov = cfg_mod._sx_mtp_lane_capture_sizes_override
    lane = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(24, q)
    bounded = cfg_mod._sm70_speculative_cudagraph_capture_sizes(24, q)
    split = cfg_mod._sm70_mtp_cudagraph_capture_sizes(24, q)
    with sx_env(VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS="1"):
        # arg_utils default (bounded, env unset at that time) and deferred.
        assert ov(_capture_config(bounded, max(bounded), k), True) == lane
        assert ov(_capture_config(split, max(split), k), True) == lane
        assert ov(_capture_config(split, None, k), True) == lane
        assert ov(_capture_config(None, None, k), True) == lane
        # Not admitted / operator lists / explicit maxima are kept.
        assert ov(_capture_config(bounded, max(bounded), k), False) is None
        operator = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 60, 72]
        assert ov(_capture_config(operator, 72, k), True) is None
        assert ov(_capture_config(bounded, 2 * q, k), True) is None
        assert ov(_capture_config(None, 2 * q, k), True) is None
    with sx_env(VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS="0"):
        got = ov(_capture_config(bounded, max(bounded), k), True)
        assert got == sorted({1, 2, 4, 8, 9, 18} | set(lane))


def test_lane_capture_sizes_token_budget():
    """Widths never exceed max_num_batched_tokens (_set_cudagraph_sizes would
    reject such a maximum at startup); the cap is the widest schedulable B."""
    f = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes
    assert f(24, 5, None) == f(24, 5) == f(24, 5, 8192) == f(24, 5, 0)
    assert f(24, 5, 64) == [5, 10, 15, 20, 30, 40, 60]  # B <= 12
    assert f(24, 5, 70) == [5, 10, 15, 20, 30, 40, 60, 70]  # cap B=14 kept
    assert f(24, 5, 120) == f(24, 5)
    assert f(24, 5, 4) == []  # not even one verify request fits
    for k in K_VALUES:
        for budget in (2, 7, 16, 33, 64, 100, 119, 120, 512):
            sizes = f(24, k + 1, budget)
            assert all(size <= budget for size in sizes), (k, budget, sizes)
            if sizes:
                assert sizes[-1] == (k + 1) * min(24, budget // (k + 1))


def _capture_config_budget(sizes, k, max_num_seqs, budget):
    cfg = _capture_config(sizes, max(sizes), k, max_num_seqs)
    cfg.scheduler_config.max_num_batched_tokens = budget
    return cfg


@pytest.mark.parametrize("k", K_VALUES)
def test_capture_size_override_token_budget(k):
    q = k + 1
    ov = cfg_mod._sx_mtp_lane_capture_sizes_override
    bounded = cfg_mod._sm70_speculative_cudagraph_capture_sizes(24, q)
    with sx_env(VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS="1"):
        got = ov(_capture_config_budget(bounded, k, 24, 64), True)
        assert got == cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(24, q, 64)
        assert max(got) <= 64
        # Not even one verify request fits: keep the current list.
        assert ov(_capture_config_budget(bounded, k, 24, q - 1), True) is None
    with sx_env(VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS="0"):
        got = ov(_capture_config_budget(bounded, k, 24, 16), True)
        assert max(got) <= 16
        assert set(got) >= {s for s in (1, 2, 4, 8, 9) if s <= 16}
        assert 18 not in got


@pytest.mark.parametrize("split", ["0", "1"])
@pytest.mark.parametrize("k", K_VALUES)
def test_capture_size_override_idempotent(k, split):
    """The worker re-runs VllmConfig.__post_init__ on the shared
    compilation_config when the MTP drafter builds its config through
    vllm.config.replace(); re-applying the override to its own output must
    keep the list (never flip it), for every max_num_seqs / budget / env."""
    q = k + 1
    ov = cfg_mod._sx_mtp_lane_capture_sizes_override
    envs_cases = (
        {},
        dict(SX_OPT_MTP_GRAPH_MAX_REQS="16"),
        dict(SX_OPT_MTP_GRAPH_REQS="1,2,4,8,16"),
    )
    with sx_env(VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=split):
        for env in envs_cases:
            with sx_env(**env):
                for max_num_seqs in (1, 4, 8, 16, 24, 32):
                    for budget in (None, 64, 8192):
                        for legacy in (
                            cfg_mod._sm70_mtp_cudagraph_capture_sizes(max_num_seqs, q),
                            cfg_mod._sm70_speculative_cudagraph_capture_sizes(
                                max_num_seqs, q
                            ),
                        ):
                            first = ov(
                                _capture_config_budget(legacy, k, max_num_seqs, budget),
                                True,
                            )
                            if first is None:
                                continue
                            again = ov(
                                _capture_config_budget(first, k, max_num_seqs, budget),
                                True,
                            )
                            assert again is None or again == first, (
                                k, split, env, max_num_seqs, budget, first, again,
                            )  # fmt: skip


# ----------------------------------------------------------------------------
# PW-1 sizes (MTP-6 / PF2)
# ----------------------------------------------------------------------------
def pw_config(spec, max_num_batched_tokens=8192):
    return SimpleNamespace(
        model_config=TARGET,
        parallel_config=parallel(),
        speculative_config=spec,
        lora_config=None,
        scheduler_config=SimpleNamespace(
            max_num_seqs=24, max_num_batched_tokens=max_num_batched_tokens
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_inductor_graph_partition=False,
            pass_config=SimpleNamespace(enable_sp=False),
            splitting_ops=list(CompilationConfig._attention_ops)
            + ["vllm::unified_kv_cache_update"],
        ),
    )


@pytest.mark.parametrize("k", K_VALUES)
def test_pw_sizes_mtp_lane(k):
    q = k + 1
    verify = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(24, q)
    sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(pw_config(mtp_spec(k=k)), verify)
    assert sizes == [s for s in NOMTP_GRID if s > max(verify)], (k, sizes)
    assert set(sizes).isdisjoint(verify)
    with sx_env(SX_OPT_MTP_PW="0"):
        assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
            pw_config(mtp_spec(k=k)), verify
        ) == []
    with sx_env(SX_OPT_MTP_LANE="0"):
        assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
            pw_config(mtp_spec(k=k)), verify
        ) == []
    with sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="0"):
        assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
            pw_config(mtp_spec(k=k)), verify
        ) == []
    assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
        pw_config(mtp_spec(k=k, method="dflash")), verify
    ) == []


def test_pw_sizes_nomtp_unchanged():
    for env in ({}, dict(SX_OPT_MTP_PW="0"), dict(SX_OPT_MTP_LANE="0")):
        with sx_env(**env):
            assert cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
                pw_config(None), [1, 2, 4, 8, 16, 24]
            ) == NOMTP_GRID


def test_pw_switch_helpers():
    assert cfg_mod._sx_mtp_pw_enabled() and cfg_mod._sx_mtp_pw_draft_enabled()
    with sx_env(SX_OPT_MTP_PW_DRAFT="0"):
        assert cfg_mod._sx_mtp_pw_enabled() and not cfg_mod._sx_mtp_pw_draft_enabled()
    with sx_env(SX_OPT_MTP_PW="0"):
        assert not cfg_mod._sx_mtp_pw_draft_enabled()


# ----------------------------------------------------------------------------
# FP16 route contract (sm70_fp16_gemv / sm70_fp16_hc)
# ----------------------------------------------------------------------------
def _gemv():
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv

    return sm70_fp16_gemv


def test_fp16_route_contract():
    gemv = _gemv()
    nomtp = SimpleNamespace(
        model_config=TARGET, parallel_config=parallel(), speculative_config=None
    )
    mtp = SimpleNamespace(
        model_config=TARGET, parallel_config=parallel(), speculative_config=mtp_spec()
    )
    other = SimpleNamespace(
        model_config=TARGET,
        parallel_config=parallel(),
        speculative_config=mtp_spec(method="dflash"),
    )
    assert gemv._exact_runtime_contract(nomtp)
    assert gemv._exact_runtime_contract(mtp)
    assert not gemv._exact_runtime_contract(other)
    with sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="0"):
        assert gemv._exact_runtime_contract(nomtp)  # unchanged
        assert not gemv._exact_runtime_contract(mtp)
    with sx_env(SX_OPT_MTP_LANE="0"):
        assert gemv._exact_runtime_contract(nomtp)
        assert not gemv._exact_runtime_contract(mtp)
    tp2 = SimpleNamespace(
        model_config=TARGET, parallel_config=parallel(tp=2), speculative_config=mtp_spec()
    )
    assert not gemv._exact_runtime_contract(tp2)


# ----------------------------------------------------------------------------
# decode-graph semantics
# ----------------------------------------------------------------------------
def test_semantics_nomtp_unchanged():
    assert not dg.sm70_mtp_lane_installed()
    assert not dg.use_sm70_decode_graph_semantics()
    with dg.sm70_decode_graph_compilation():
        assert dg.use_sm70_decode_graph_semantics()
    with dg.sm70_target_main_backbone():
        assert not dg.use_sm70_decode_graph_semantics()
    with sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="0"):
        assert dg.use_sm70_decode_graph_semantics()


def test_semantics_mtp_lane():
    dg.set_sm70_mtp_lane_installed(True)
    # Drafter (and anything else outside the target main backbone).
    assert dg.is_sm70_mtp_drafter_decode_semantics()
    assert dg.use_sm70_decode_graph_semantics()
    assert not dg.is_sm70_decode_graph_compiling()
    # Target main compile (prefill/mixed/eager).
    with dg.sm70_target_main_backbone():
        assert not dg.use_sm70_decode_graph_semantics()
        assert not dg.is_sm70_mtp_drafter_decode_semantics()
    # Target FULL verify graph capture (decode compiler).
    with dg.sm70_decode_graph_compilation():
        assert dg.use_sm70_decode_graph_semantics()
        assert dg.is_sm70_decode_graph_compiling()
    with sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="0"):
        with dg.sm70_target_main_backbone():
            assert dg.use_sm70_decode_graph_semantics()  # legacy lane
    dg.set_sm70_mtp_lane_installed(False)
    assert not dg.use_sm70_decode_graph_semantics()


def test_rows_gate_mtp_switch(monkeypatch):
    gemv = _gemv()
    monkeypatch.setattr(
        gemv.current_platform, "is_device_capability", lambda *a, **k: True
    )
    with dg.sm70_decode_graph_compilation():
        assert gemv._sx_decode_graph_active()  # no-MTP lane
        with sx_env(SX_OPT_MTP_ROWS="0"):
            assert gemv._sx_decode_graph_active()  # switch is MTP-lane only
        dg.set_sm70_mtp_lane_installed(True)
        assert gemv._sx_decode_graph_active()
        with sx_env(SX_OPT_MTP_ROWS="0"):
            assert not gemv._sx_decode_graph_active()
        dg.set_sm70_mtp_lane_installed(False)
    dg.set_sm70_mtp_lane_installed(True)
    assert not gemv._sx_decode_graph_active()  # never outside FULL capture
    with dg.sm70_target_main_backbone():
        assert not gemv._sx_decode_graph_active()


def test_model_lane_installed_helper():
    from vllm.models.qwen4_exp.nvidia.model import _sx_sm70_qwen38_mtp_lane_installed

    def vcfg(spec):
        return SimpleNamespace(
            model_config=TARGET, parallel_config=parallel(), speculative_config=spec
        )

    assert _sx_sm70_qwen38_mtp_lane_installed(vcfg(mtp_spec()))
    assert not _sx_sm70_qwen38_mtp_lane_installed(vcfg(None))
    assert not _sx_sm70_qwen38_mtp_lane_installed(vcfg(mtp_spec(method="dflash")))
    with sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="0"):
        assert not _sx_sm70_qwen38_mtp_lane_installed(vcfg(mtp_spec()))
    with sx_env(SX_OPT_MTP_LANE="0"):
        assert not _sx_sm70_qwen38_mtp_lane_installed(vcfg(mtp_spec()))


def _forward_probe(lane_main_backbone: bool, fail: bool = False):
    """A stand-in `self` for Qwen4ExpForCausalLM.forward: the main and decode
    backbones record the semantics they observe (what their Dynamo trace
    would bake in)."""
    seen: list[tuple[str, bool, bool]] = []

    def backbone(name):
        def run(*args, **kwargs):
            seen.append(
                (
                    name,
                    dg.use_sm70_decode_graph_semantics(),
                    dg._sm70_target_main_backbone.get(),
                )
            )
            if fail:
                raise RuntimeError("backbone failed")
            return name

        return run

    probe = SimpleNamespace(
        model=backbone("main"),
        _sm70_decode_graph_model=backbone("decode"),
        _sx_mtp_lane_main_backbone=lane_main_backbone,
    )
    return probe, seen


def test_model_forward_marks_main_backbone():
    """model.py wiring of the lane: the target main backbone (prefill, mixed,
    eager and PIECEWISE steps) runs with the no-MTP main-compile semantics,
    FULL verify captures run the decode backbone with decode semantics, and
    code running after the target forward (the MTP drafter) keeps decode
    semantics. The no-MTP lane is unchanged."""
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM

    forward = Qwen4ExpForCausalLM.forward
    positions = torch.zeros(4, dtype=torch.long)

    dg.set_sm70_mtp_lane_installed(True)
    probe, seen = _forward_probe(True)
    assert forward(probe, None, positions) == "main"
    assert seen[-1] == ("main", False, True)
    # Drafter after the target forward returned: context reset.
    assert not dg._sm70_target_main_backbone.get()
    assert dg.use_sm70_decode_graph_semantics()
    with dg.sm70_decode_graph_compilation():
        assert forward(probe, None, positions) == "decode"
    assert seen[-1] == ("decode", True, False)
    # The context is reset even when the backbone raises.
    probe, _ = _forward_probe(True, fail=True)
    with pytest.raises(RuntimeError):
        forward(probe, None, positions)
    assert not dg._sm70_target_main_backbone.get()

    # No-MTP lane: flag off, the main backbone is called exactly as before.
    dg.set_sm70_mtp_lane_installed(False)
    probe, seen = _forward_probe(False)
    assert forward(probe, None, positions) == "main"
    assert seen[-1] == ("main", False, False)
    with dg.sm70_decode_graph_compilation():
        assert forward(probe, None, positions) == "decode"
    assert seen[-1] == ("decode", True, False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
