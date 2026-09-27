# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the batch-3a "moe-verify" gates (SX_OPT_MTP_MOE_ROUTES).

GPUs: 0 (runs anywhere the image's vllm imports):
  /opt/venv/bin/python -m pytest -q sx_tests/b3-moe-verify/test_mtp_verify_dispatch_cpu.py

Asserts:
  * lane contract (sx_sm70_qwen38_mtp_verify_q): q = k + 1 only for method
    "mtp" + Qwen4Exp drafter, 1 <= k <= 7, chain drafting, standard
    rejection, the exact Qwen3.8 TP4 target contract (resolved through
    speculative_config.target_model_config when the draft is being built),
    no DBO, SM70, master switch on; every other config -> 0.
  * router: runtime-M M<=32 route armed in the lane (target and draft), not
    for other speculative configs, unchanged without spec.
  * shared gate: rows gate constructed in the lane (probe runs), not for
    other speculative configs; unchanged without spec.
  * NVFP4 MoE: load-time lane stamping (_sx_mtp_moe_verify_q, grouped v2
    contract, eager iota; PERSIST32 unchanged at 18); uniform-verify
    metadata classification (prefill, plain decode, mixed, non-uniform
    drafts, non-int counters rejected); verify context cached per forward;
    live-row view with q rows per request; grouped route/split per (q, W) in
    verify / decode / mixed contexts incl. v2-absent and switch attributes;
    direct QPN dynamic / MTP5 admission only in the verify context; route
    table for k = 1..7; layers without the lane attributes keep exactly the
    1.8.0-dev2 routes in every context.
  * V2 runner invariants the verify masking relies on (FULL uniform capture
    uses num_reqs = W / q, make_dummy writes tail = W).
  * custom all-reduce: the 25-KiB MTP5 push env is defaulted only in the
    k = 4 lane, never over an explicit value or with the switch off.
Review additions:
  * no-MTP lane: grouped split / QPN batch split / MTP5 / fused-W2 decisions
    of production-stamped and attribute-free layers equal a frozen restatement
    of the 1.8.0-dev2 code in every context (none, decode, verify, mixed,
    attention-only) for all 16 QPN env combinations, widths 1..40;
  * the real FlashAttention / GDN / PLE metadata dataclasses carry the fields
    the verify classification reads, classify a uniform verify, reject
    prefill / plain-decode / fewer-draft / other-q / non-spec PLE builds, and
    a private query_start_loc copy disables masking (never a wrong count);
  * a non-int counter rejects without any comparison or conversion (no
    device read-back) and is logged once;
  * PIECEWISE captures run without attention metadata and FULL uniform
    descriptors pad requests to W / q (cudagraph_utils source tripwires);
  * the W5 route-table label says "+batch-w2" only when apply() really takes
    the fused W2 reduce.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.model_executor.layers.fused_moe.router import (  # noqa: E402
    fused_topk_router as router,
)
from vllm.model_executor.layers.quantization import (  # noqa: E402
    nvfp4_sm70_moe as moe,
)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _sm70_platform(monkeypatch):
    monkeypatch.setattr(
        router,
        "current_platform",
        NS(is_device_capability=lambda cap, *a, **k: cap == 70),
    )
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", True)
    monkeypatch.setattr(router, "_SX_OPT_ROUTER32", True)
    monkeypatch.delenv("SX_OPT_MTP_LANE", raising=False)


# ----------------------------------------------------------------------------
# lane contract
@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("draft", [False, True])
def test_lane_contract_admits(k, draft):
    assert router.sx_sm70_qwen38_mtp_verify_q(C.lane_config(k, draft=draft)) == k + 1


def test_lane_contract_target_from_config_model(monkeypatch):
    cfg = C.lane_config(4, spec=C.mtp_spec(4, target_model_config=None,
                                           target_parallel_config=None))
    assert router.sx_sm70_qwen38_mtp_verify_q(cfg) == 5
    # ... but the drafter's own model config never satisfies the contract.
    cfg.model_config = C.draft_model_config()
    assert router.sx_sm70_qwen38_mtp_verify_q(cfg) == 0


def _bad(**spec_over):
    return C.lane_config(4, spec=C.mtp_spec(4, **spec_over))


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        NS(speculative_config=None),
        NS(speculative_config=object()),
        _bad(method="eagle"),
        _bad(method="dflash"),
        _bad(use_qwen4_exp_mtp=lambda: False),
        _bad(use_qwen4_exp_mtp=None),
        _bad(num_speculative_tokens=0, num_speculative_state_tokens=lambda: 0),
        _bad(num_speculative_tokens=8, num_speculative_state_tokens=lambda: 8),
        _bad(num_speculative_state_tokens=lambda: 16),  # tree verification
        _bad(parallel_drafting=True),
        _bad(rejection_sample_method="probabilistic"),
        _bad(target_parallel_config=C.parallel_config(tp=2)),
        _bad(target_parallel_config=C.parallel_config(enable_dbo=True)),
        _bad(target_parallel_config=C.parallel_config(ubatch_size=2)),
        _bad(target_model_config=C.target_model_config(hidden_size=2048)),
        _bad(target_model_config=C.target_model_config(num_experts=256)),
        _bad(target_model_config=C.target_model_config(arch="Qwen3NextForCausalLM")),
        _bad(target_model_config=NS(hf_text_config=C.qwen38_text_config(),
                                    architectures=["Qwen4ExpForCausalLM"],
                                    multimodal_config=None,
                                    dtype=torch.bfloat16)),
        _bad(use_qwen4_exp_mtp=Mock(side_effect=RuntimeError("partial"))),
    ],
)
def test_lane_contract_rejects(cfg):
    assert router.sx_sm70_qwen38_mtp_verify_q(cfg) == 0


def test_lane_contract_switch_and_platform(monkeypatch):
    cfg = C.lane_config(4)
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", False)
    assert router.sx_sm70_qwen38_mtp_verify_q(cfg) == 0
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", True)
    # The lane-wide master (vllm/config/vllm.py) disables this group too.
    for raw, expected in (("0", 0), (" 0 ", 0), ("1", 5), ("", 5)):
        monkeypatch.setenv("SX_OPT_MTP_LANE", raw)
        assert router.sx_sm70_qwen38_mtp_verify_q(cfg) == expected, raw
    monkeypatch.delenv("SX_OPT_MTP_LANE")
    monkeypatch.setattr(
        router, "current_platform", NS(is_device_capability=lambda cap: cap == 80)
    )
    assert router.sx_sm70_qwen38_mtp_verify_q(cfg) == 0


# ----------------------------------------------------------------------------
# router (SX_OPT_ROUTER32 in the lane)
E, K = 512, 10


def _router_flag(monkeypatch, cfg):
    import vllm.config.vllm as vcfg

    monkeypatch.setattr(vcfg, "get_current_vllm_config_or_none", lambda: cfg)
    return router.FusedTopKRouter(
        top_k=K, global_num_experts=E, scoring_func="softmax", renormalize=True
    )._sm70_qwen38_router_runtime_m


def test_router_lane_flag(monkeypatch):
    assert _router_flag(monkeypatch, C.lane_config(4)) is True
    assert _router_flag(monkeypatch, C.lane_config(2, draft=True)) is True
    assert _router_flag(monkeypatch, _bad(method="eagle")) is False
    assert _router_flag(monkeypatch, NS(speculative_config=NS(method="mtp"))) is False
    assert _router_flag(monkeypatch, C.nomtp_config()) is True  # unchanged
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", False)
    assert _router_flag(monkeypatch, C.lane_config(4)) is False
    assert _router_flag(monkeypatch, C.nomtp_config()) is True  # unchanged
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", True)
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    assert _router_flag(monkeypatch, C.lane_config(4)) is False
    assert _router_flag(monkeypatch, C.nomtp_config()) is True  # unchanged
    monkeypatch.delenv("SX_OPT_MTP_LANE")
    monkeypatch.setattr(router, "_SX_OPT_ROUTER32", False)
    assert _router_flag(monkeypatch, C.lane_config(4)) is False


# ----------------------------------------------------------------------------
# shared gate rows (qwen2_moe)
def _construct_shared_expert(monkeypatch, cfg, prefix):
    from vllm.model_executor.models import qwen2_moe

    monkeypatch.setattr(
        qwen2_moe, "envs", NS(VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION=True)
    )
    monkeypatch.setattr(qwen2_moe, "MergedColumnParallelLinear", lambda *a, **k: NS())
    monkeypatch.setattr(qwen2_moe, "RowParallelLinear", lambda *a, **k: NS())
    monkeypatch.setattr(qwen2_moe, "SiluAndMul", lambda **k: NS())
    monkeypatch.setattr(
        qwen2_moe, "_sm70_force_shared_expert_silu_custom_op", lambda prefix: True
    )
    monkeypatch.setattr(
        qwen2_moe,
        "_sm70_fused_shared_expert_gate_module_supported",
        lambda gate_up, down: True,
    )
    monkeypatch.setattr(qwen2_moe, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(qwen2_moe, "_SX_OPT_SHARED_GATE_ROWS", True)
    capable = Mock(return_value=True)
    monkeypatch.setattr(qwen2_moe, "_sx_shared_gate_rows_capable", capable)
    layer = qwen2_moe.Qwen2MoeMLP(
        hidden_size=2560,
        intermediate_size=160,
        hidden_act="silu",
        reduce_results=False,
        expert_gate=NS(weight=torch.zeros(1, 2560, dtype=torch.float16)),
        prefix=prefix,
    )
    return layer, capable


@pytest.mark.parametrize(
    "cfg_name,prefix,expected",
    [
        ("lane_target", "model.layers.3.mlp.shared_expert", True),
        ("lane_draft", "mtp.layers.48.mlp.shared_expert", True),
        ("nomtp", "model.layers.3.mlp.shared_expert", True),
        ("eagle", "model.layers.3.mlp.shared_expert", False),
        ("mtp_other_model", "model.layers.3.mlp.shared_expert", False),
        ("opaque_spec", "model.layers.3.mlp.shared_expert", False),
    ],
)
def test_shared_gate_rows_lane(monkeypatch, cfg_name, prefix, expected):
    cfg = {
        "lane_target": C.lane_config(4),
        "lane_draft": C.lane_config(4, draft=True),
        "nomtp": C.nomtp_config(),
        "eagle": _bad(method="eagle"),
        "mtp_other_model": _bad(
            target_model_config=C.target_model_config(hidden_size=2048)
        ),
        "opaque_spec": NS(speculative_config=object()),
    }[cfg_name]
    layer, capable = _construct_shared_expert(monkeypatch, cfg, prefix)
    assert layer._sm70_exact_shared_expert_gate is True
    assert layer._sx_shared_gate_rows is expected
    assert capable.call_count == int(expected)


def test_shared_gate_rows_lane_switch_off(monkeypatch):
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", False)
    layer, capable = _construct_shared_expert(
        monkeypatch, C.lane_config(4), "model.layers.3.mlp.shared_expert"
    )
    assert layer._sx_shared_gate_rows is False and capable.call_count == 0


def test_shared_gate_rows_lane_master_off(monkeypatch):
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    layer, capable = _construct_shared_expert(
        monkeypatch, C.lane_config(4), "model.layers.3.mlp.shared_expert"
    )
    assert layer._sx_shared_gate_rows is False and capable.call_count == 0
    layer, capable = _construct_shared_expert(
        monkeypatch, C.nomtp_config(), "model.layers.3.mlp.shared_expert"
    )
    assert layer._sx_shared_gate_rows is True  # no-MTP lane unchanged


# ----------------------------------------------------------------------------
# NVFP4 MoE: load-time stamping
def _contract_layer(tp=4, experts=512, hidden=2560, inter=160, top_k=10):
    return NS(
        moe_config=NS(tp_size=tp),
        sm70_nvfp4_num_experts=experts,
        sm70_nvfp4_hidden_size=hidden,
        sm70_nvfp4_intermediate_size=inter,
        sm70_nvfp4_top_k=top_k,
        global_num_experts=experts,
    )


def _patch_cfg(monkeypatch, cfg):
    monkeypatch.setattr(moe, "get_current_vllm_config_or_none", lambda: cfg)


def test_mtp_moe_verify_q_load_time(monkeypatch):
    _patch_cfg(monkeypatch, C.lane_config(4))
    assert moe._sx_mtp_moe_verify_q(_contract_layer()) == 5
    assert moe._sx_mtp_moe_verify_q(_contract_layer(tp=2, inter=320)) == 0
    _patch_cfg(monkeypatch, C.lane_config(3))
    assert moe._sx_mtp_moe_verify_q(_contract_layer()) == 4
    _patch_cfg(monkeypatch, C.nomtp_config())
    assert moe._sx_mtp_moe_verify_q(_contract_layer()) == 0
    cfg = C.lane_config(4)
    cfg.parallel_config = C.parallel_config(use_ubatching=True)
    _patch_cfg(monkeypatch, cfg)
    assert moe._sx_mtp_moe_verify_q(_contract_layer()) == 0
    _patch_cfg(monkeypatch, None)
    assert moe._sx_mtp_moe_verify_q(_contract_layer()) == 0
    _patch_cfg(monkeypatch, C.lane_config(4))
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    assert moe._sx_mtp_moe_verify_q(_contract_layer()) == 0


def test_grouped_v2_contract_lane(monkeypatch):
    layer = _contract_layer()
    _patch_cfg(monkeypatch, C.lane_config(4))
    assert moe._grouped_v2_contract(layer)
    monkeypatch.setattr(moe, "_SX_OPT_MTP_MOE_GROUPED", False)
    assert not moe._grouped_v2_contract(layer)
    monkeypatch.setattr(moe, "_SX_OPT_MTP_MOE_GROUPED", True)
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    assert not moe._grouped_v2_contract(layer)  # dev2 MTP lane: no v2
    monkeypatch.delenv("SX_OPT_MTP_LANE")
    _patch_cfg(monkeypatch, _bad(method="eagle"))
    assert not moe._grouped_v2_contract(layer)
    _patch_cfg(monkeypatch, C.nomtp_config())
    assert moe._grouped_v2_contract(layer)  # unchanged no-MTP lane


def test_eager_iota_and_persist_in_lane(monkeypatch):
    monkeypatch.setattr(moe, "_SX_OPT_MOE_EAGER_IOTA", True)
    monkeypatch.setattr(moe, "_SX_OPT_MOE_PERSIST32", True)
    monkeypatch.setattr(moe, "_qwen38_eager_iotas", {})
    layer = _contract_layer()
    cpu0 = torch.device("cpu", 0)
    _patch_cfg(monkeypatch, C.lane_config(4, max_tokens=64))
    iota = moe._get_qwen38_eager_iota(layer, cpu0)
    assert iota is not None and torch.equal(
        iota, torch.arange(64 * K + 1, dtype=torch.int32)
    )
    # PERSIST32 stays off in the MTP lane (design_1 / design_4 rejected).
    assert moe._persistent_max_tokens_for(layer) == 18
    monkeypatch.setattr(moe, "_qwen38_eager_iotas", {})
    _patch_cfg(monkeypatch, _bad(method="eagle"))
    assert moe._get_qwen38_eager_iota(layer, cpu0) is None
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", False)
    _patch_cfg(monkeypatch, C.lane_config(4, max_tokens=64))
    assert moe._get_qwen38_eager_iota(layer, cpu0) is None
    monkeypatch.setattr(router, "_SX_OPT_MTP_MOE_ROUTES", True)
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    assert moe._get_qwen38_eager_iota(layer, cpu0) is None


# ----------------------------------------------------------------------------
# uniform verify metadata
def _qsl(n=40):
    return torch.arange(n, dtype=torch.int32)


@pytest.mark.parametrize("q", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("reqs", [1, 3, 6])
def test_uniform_verify_metadata_accepts(q, reqs):
    view = _qsl()[: reqs + 1]
    assert moe._uniform_verify_metadata(C.verify_metadata(view, reqs * q, q), q)
    # Attention-only metadata (no Mamba groups) is a verify too.
    assert moe._uniform_verify_metadata({"a": NS(max_query_len=q)}, q)
    # Backends that count verify requests as decodes (threshold q).
    assert moe._uniform_verify_metadata(
        {"a": NS(max_query_len=q, num_decodes=reqs, num_decode_tokens=reqs * q)}, q
    )


@pytest.mark.parametrize(
    "case",
    ["other_q", "decode", "plain_decodes", "prefill", "non_uniform", "no_spec",
     "empty", "not_int", "decode_threshold_non_uniform", "no_verify_fields"],
)
def test_uniform_verify_metadata_rejects(case):
    q, view = 5, _qsl()[:4]
    meta = {
        "other_q": C.verify_metadata(view, 12, 4),
        "decode": C.decode_metadata(_qsl()[:16], 15),
        "plain_decodes": C.verify_metadata(view, 15, q, plain_decodes=2),
        "prefill": C.mixed_metadata(view, q),
        "non_uniform": C.verify_metadata(view, 15, q, spec_tokens=13),
        "no_spec": C.verify_metadata(view, 15, q, spec_reqs=0),
        "empty": {},
        "not_int": {"a": NS(max_query_len=torch.tensor(5))},
        "decode_threshold_non_uniform": {
            "a": NS(max_query_len=q, num_decodes=3, num_decode_tokens=13)
        },
        "no_verify_fields": {"a": NS(foo=1)},
    }[case]
    assert not moe._uniform_verify_metadata(meta, q)


class _NoReadback:
    """A counter that must never be compared or converted (a device tensor
    would synchronize the host on __bool__/__int__/__index__/comparisons)."""

    def _fail(self, *args):
        raise AssertionError("verify classification read a non-int counter")

    __bool__ = __int__ = __index__ = __eq__ = __ne__ = _fail
    __lt__ = __le__ = __gt__ = __ge__ = __mul__ = __rmul__ = _fail
    __hash__ = object.__hash__


@pytest.mark.parametrize(
    "field",
    ["num_prefills", "num_prefill_tokens", "max_query_len", "num_spec_decodes",
     "num_spec_decode_tokens", "num_decodes", "num_decode_tokens"],
)
def test_non_int_counter_rejects_without_readback(monkeypatch, field):
    seen = []
    monkeypatch.setattr(
        moe, "logger", NS(warning_once=lambda *a, **k: seen.append(a))
    )
    q, reqs = 5, 3
    view = _qsl()[: reqs + 1]
    meta = C.verify_metadata(view, reqs * q, q)
    assert moe._uniform_verify_metadata(meta, q) and not seen
    target = meta["layers.3.self_attn"] if field == "max_query_len" else (
        meta["layers.0.linear_attn"])
    setattr(target, field, _NoReadback())
    assert not moe._uniform_verify_metadata(meta, q)
    assert [a[2:] for a in seen] == [(field, "_NoReadback")]
    # Decode-threshold backends (no spec counters) are covered too.
    seen.clear()
    meta = {"a": NS(max_query_len=q, num_decodes=reqs, num_decode_tokens=_NoReadback())}
    assert not moe._uniform_verify_metadata(meta, q)
    assert [a[2:] for a in seen] == [("num_decode_tokens", "_NoReadback")]


def test_verify_context_cached(monkeypatch):
    calls = []
    real = moe._uniform_verify_metadata
    monkeypatch.setattr(
        moe, "_uniform_verify_metadata", lambda m, q: calls.append(q) or real(m, q)
    )
    view = _qsl()[:5]
    with C.forward_context(moe, C.verify_metadata(view, 20, 5)):
        assert moe._grouped_verify_context_ok(5)
        assert moe._grouped_verify_context_ok(5)
        assert calls == [5]
        assert not moe._grouped_verify_context_ok(4)  # other q re-classifies
        assert calls == [5, 4]
        assert not moe._grouped_verify_context_ok(1)
    monkeypatch.setattr(moe, "is_forward_context_available", lambda: False)
    assert not moe._grouped_verify_context_ok(5)


# ----------------------------------------------------------------------------
# live rows with q rows per request
@pytest.mark.parametrize("q,reqs", [(5, 1), (5, 4), (5, 6), (4, 8), (3, 10), (2, 16)])
def test_live_rows_view_verify(q, reqs):
    buf = torch.zeros(40, dtype=torch.int32)
    tokens = reqs * q
    C.write_live(buf, tokens - q if reqs > 1 else tokens, q)
    meta = C.verify_metadata(buf[: reqs + 1], tokens, q)
    view = moe._find_live_rows_view(meta, tokens, CPU, q)
    assert view is not None and view.data_ptr() == buf[reqs:].data_ptr()
    assert int(view[0]) == (tokens - q if reqs > 1 else tokens)
    # The q = 1 interpretation of the same metadata fails closed.
    if q > 1:
        assert moe._find_live_rows_view(meta, tokens, CPU) is None
    # Width not a multiple of q, or a different view length: fail closed.
    assert moe._find_live_rows_view(meta, tokens + 1, CPU, q) is None
    assert moe._find_live_rows_view(
        C.verify_metadata(buf[: reqs + 2], tokens, q), tokens, CPU, q
    ) is None


def test_live_rows_cache_keys_rows_per_request(monkeypatch):
    buf = torch.zeros(40, dtype=torch.int32)
    C.write_live(buf, 20, 5)
    meta = C.verify_metadata(buf[:5], 20, 5)
    with C.forward_context(moe, meta) as ctx:
        x = torch.empty(20, 2560, dtype=torch.float16)
        live = moe._grouped_decode_live_rows(x, 5)
        assert live is not None and int(live[0]) == 20
        assert ctx.additional_kwargs[moe._GROUPED_LIVE_ROWS_KEY][0] == 20
        # Same width, q = 1: re-discovers (and fails closed on this view).
        assert moe._grouped_decode_live_rows(x) is None
        assert moe._grouped_decode_live_rows(x, 5) is not None


# ----------------------------------------------------------------------------
# grouped route per (q, width)
V2_SENTINEL = ("w13_v2", "w2_v2")


def _lane_layer(q, *, v2=True, grouped=True, min_tokens=8, split_small=4,
                split32=8, max_tokens=32, dynamic=True, mtp5=None):
    return NS(
        moe_config=NS(tp_size=4),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_grouped_decode=True,
        sm70_nvfp4_grouped_max_tokens=max_tokens if v2 else 16,
        sm70_nvfp4_grouped32_split=split32,
        sm70_nvfp4_grouped_mask=v2,
        _nvfp4_grouped_v2_ops=V2_SENTINEL if v2 else None,
        sx_mtp_verify_q=q,
        sx_mtp_grouped=grouped,
        sx_mtp_grouped_min_tokens=min_tokens,
        sx_mtp_grouped_split_small=split_small,
        sx_mtp_qpn_dynamic=dynamic,
        sx_mtp_qpn_mtp5=(q == 5) if mtp5 is None else mtp5,
    )


def _dev2_layer():
    """A layer as 1.8.0-dev2 built it (no batch-3a attributes)."""
    return NS(
        moe_config=NS(tp_size=4),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_grouped_decode=True,
        sm70_nvfp4_grouped_max_tokens=16,  # MTP lane: v2 contract was refused
        sm70_nvfp4_grouped32_split=8,
    )


def _xi(tokens):
    return (
        torch.empty(tokens, 2560, dtype=torch.float16),
        torch.empty(tokens, 10, dtype=torch.int32),
    )


def _expected_verify(q, w, *, v2=True, min_tokens=8, split_small=4, max_tokens=32):
    if w % q or not min_tokens <= w <= (max_tokens if v2 else 16):
        return None
    if w not in (8, 16) and not v2:
        return None
    if w in (8, 16):
        split = {8: 4, 16: 8}[w]
    elif w > 16:
        split = 8
    elif w <= 12:
        split = split_small
    else:
        split = 8
    return split, q


@pytest.mark.parametrize("q", range(2, 9))
def test_grouped_route_verify_context(q):
    layer = _lane_layer(q)
    buf = torch.zeros(80, dtype=torch.int32)
    for w in range(1, 41):
        x, ids = _xi(w)
        meta = (C.verify_metadata(buf[: w // q + 1], w, q) if w % q == 0
                else {"a": NS(max_query_len=q)})
        with C.forward_context(moe, meta):
            got = moe._grouped_decode_route(layer, x, ids)
        exp = _expected_verify(q, w) if w % q == 0 else None
        assert got == exp, (q, w, got, exp)
        with C.forward_context(moe, meta):
            assert moe._grouped_decode_split(layer, x, ids) == (
                None if exp is None else exp[0]
            )


@pytest.mark.parametrize("q", [2, 4, 5])
def test_grouped_route_decode_and_mixed_context(q):
    layer = _lane_layer(q)
    buf = torch.zeros(80, dtype=torch.int32)
    for w in range(1, 41):
        x, ids = _xi(w)
        # Pure decode in the lane: the no-MTP admission, 1 row per request.
        with C.forward_context(moe, C.decode_metadata(buf[: w + 1], w)):
            got = moe._grouped_decode_route(layer, x, ids)
        exp = ({8: 4, 16: 8}[w], 1) if w in (8, 16) else (
            (8, 1) if 16 < w <= 32 else None
        )
        assert got == exp, (q, w)
        # Mixed prefill + verify: never grouped.
        with C.forward_context(moe, C.mixed_metadata(buf[:3], q)):
            assert moe._grouped_decode_route(layer, x, ids) is None


@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_grouped_route_knobs(q):
    buf = torch.zeros(80, dtype=torch.int32)
    for kw in (dict(v2=False), dict(min_tokens=16), dict(min_tokens=5),
               dict(split_small=8), dict(grouped=False)):
        layer = _lane_layer(q, **kw)
        for w in range(q, 41, q):
            x, ids = _xi(w)
            with C.forward_context(moe, C.verify_metadata(buf[: w // q + 1], w, q)):
                got = moe._grouped_decode_route(layer, x, ids)
            if kw.get("grouped") is False:
                exp = None
            else:
                exp = _expected_verify(
                    q, w,
                    v2=kw.get("v2", True),
                    min_tokens=kw.get("min_tokens", 8),
                    split_small=kw.get("split_small", 4),
                )
            assert got == exp, (q, w, kw, got, exp)


@pytest.mark.parametrize("q", [2, 4, 5])
def test_dev2_layer_routes_unchanged(plain_envs, q):
    """A layer without the batch-3a attributes (the no-MTP lane, or
    SX_OPT_MTP_MOE_ROUTES=0) keeps the 1.8.0-dev2 decisions everywhere."""
    layer = _dev2_layer()
    buf = torch.zeros(80, dtype=torch.int32)
    for w in range(1, 41):
        x, ids = _xi(w)
        if w % q == 0:
            with C.forward_context(moe, C.verify_metadata(buf[: w // q + 1], w, q)):
                assert moe._grouped_decode_route(layer, x, ids) is None
                assert moe._qwen38_qpn_batch_split(layer, w) == {
                    2: 10, 4: 5, 8: 4, 16: 1}.get(w)
        with C.forward_context(moe, C.decode_metadata(buf[: w + 1], w)):
            assert moe._grouped_decode_split(layer, x, ids) == {8: 4, 16: 8}.get(w)


# ----------------------------------------------------------------------------
# No-MTP lane: every dispatch decision equals a frozen 1.8.0-dev2 reference
# (review addition). The reference below is the dev2 code of
# _grouped_decode_context_ok / _use_grouped_decode / _grouped_decode_split /
# _use_qwen38_qpn_batch_decode / _use_qwen38_qpn_mtp5_decode /
# _use_qwen38_qpn_batch_fused_w2, restated on a metadata dict, so the check
# does not depend on the module under test.
def _dev2_decode_context_ok(metadata) -> bool:
    if not isinstance(metadata, dict) or not metadata:
        return False
    seen_decode = False
    for meta in metadata.values():
        prefills = getattr(meta, "num_prefills", 0)
        prefill_tokens = getattr(meta, "num_prefill_tokens", 0)
        if (
            not isinstance(prefills, int)
            or not isinstance(prefill_tokens, int)
            or prefills != 0
            or prefill_tokens != 0
        ):
            return False
        max_query = getattr(meta, "max_query_len", None)
        if max_query is not None:
            if not isinstance(max_query, int) or max_query != 1:
                return False
            seen_decode = True
        num_decodes = getattr(meta, "num_decodes", None)
        if num_decodes is not None:
            decode_tokens = getattr(meta, "num_decode_tokens", None)
            if (
                not isinstance(num_decodes, int)
                or not isinstance(decode_tokens, int)
                or decode_tokens != num_decodes
            ):
                return False
            seen_decode |= num_decodes > 0
    return seen_decode


def _dev2_routes(env, layer, x, ids, metadata):
    """(grouped split, qpn batch split, mtp5, fused w2) as dev2 decides."""
    w = int(x.shape[0])
    shape_ok = bool(
        x.ndim == 2 and x.shape[1] == 2560 and x.dtype == torch.float16
        and x.is_contiguous() and ids.shape == (w, 10)
        and ids.dtype == torch.int32 and ids.is_contiguous()
    )
    width_ok = w in (8, 16) or 16 < w <= int(
        getattr(layer, "sm70_nvfp4_grouped_max_tokens", 16))
    grouped = None
    if (
        getattr(layer, "sm70_nvfp4_grouped_decode", False)
        and width_ok and shape_ok and _dev2_decode_context_ok(metadata)
    ):
        grouped = {8: 4, 16: 8}.get(
            w, int(getattr(layer, "sm70_nvfp4_grouped32_split", 8)))
    contract = bool(
        int(layer.moe_config.tp_size) == 4
        and int(layer.sm70_nvfp4_num_experts) == 512
        and int(layer.sm70_nvfp4_hidden_size) == 2560
        and int(layer.sm70_nvfp4_intermediate_size) == 160
        and int(layer.sm70_nvfp4_top_k) == 10
    )
    table = DYNAMIC if env.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE else STATIC
    batch = bool(
        env.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE and w in table
        and shape_ok and contract
    )
    mtp5 = bool(env.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE and w == 5
                and shape_ok and contract)
    fused_w2 = bool(env.VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2 and batch)
    return grouped, (table[w] if batch else None), mtp5, fused_w2


def _new_routes(layer, x, ids):
    batch = moe._use_qwen38_qpn_batch_decode(layer, x, ids)
    return (
        moe._grouped_decode_split(layer, x, ids),
        moe._qwen38_qpn_batch_split(layer, int(x.shape[0])) if batch else None,
        moe._use_qwen38_qpn_mtp5_decode(layer, x, ids),
        moe._use_qwen38_qpn_batch_fused_w2(layer, x, ids),
    )


def _nomtp_layer(**over):
    """A production no-MTP layer as the new load code stamps it: batch-2
    GROUPED32/MASK admitted, every batch-3a attribute inert."""
    fields = dict(
        moe_config=NS(tp_size=4), sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560, sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10, sm70_nvfp4_grouped_decode=True,
        sm70_nvfp4_grouped_max_tokens=32, sm70_nvfp4_grouped32_split=8,
        sm70_nvfp4_grouped_mask=True, _nvfp4_grouped_v2_ops=V2_SENTINEL,
        sx_mtp_verify_q=0, sx_mtp_grouped=False, sx_mtp_grouped_min_tokens=8,
        sx_mtp_grouped_split_small=4, sx_mtp_qpn_mtp5=False,
        sx_mtp_qpn_dynamic=False,
    )
    fields.update(over)
    return NS(**fields)


def _contexts(w, q, buf):
    out = [("none", None), ("decode", C.decode_metadata(buf[: w + 1], w)),
           ("mixed", C.mixed_metadata(buf[:3], q)),
           ("attn_only", {"a": NS(max_query_len=q)})]
    if w % q == 0:
        out.append(("verify", C.verify_metadata(buf[: w // q + 1], w, q)))
    return out


@pytest.mark.parametrize("batch,dynamic,mtp5,fused_w2", [
    (b, d, m, f) for b in (True, False) for d in (True, False)
    for m in (True, False) for f in (True, False)
])
def test_nomtp_dispatch_equals_dev2(monkeypatch, batch, dynamic, mtp5, fused_w2):
    from vllm import envs as real

    env = C.EnvsProxy(
        real,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE=batch,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=dynamic,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE=mtp5,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2=fused_w2,
    )
    monkeypatch.setattr(moe, "envs", env)
    monkeypatch.setattr(
        moe.sm70_ops, "has_nvfp4_qpn_w2_reduce_dispatch", lambda: True
    )
    layers = {
        "production": _nomtp_layer(),
        "v2_absent": _nomtp_layer(sm70_nvfp4_grouped_max_tokens=16,
                                  sm70_nvfp4_grouped_mask=False,
                                  _nvfp4_grouped_v2_ops=None),
        "grouped_off": _nomtp_layer(sm70_nvfp4_grouped_decode=False),
        "split32_4": _nomtp_layer(sm70_nvfp4_grouped32_split=4),
        "dev2_object": _dev2_layer(),  # no batch-3a attributes at all
    }
    buf = torch.zeros(48, dtype=torch.int32)
    for w in range(1, 41):
        x, ids = _xi(w)
        for q in (2, 5):
            for label, meta in _contexts(w, q, buf):
                expected = {name: _dev2_routes(env, layer, x, ids, meta)
                            for name, layer in layers.items()}
                # One forward context shared by every layer, as in a forward.
                ctx = (contextlib.nullcontext() if meta is None
                       else C.forward_context(moe, meta))
                with ctx:
                    got = {name: _new_routes(layer, x, ids)
                           for name, layer in layers.items()}
                assert got == expected, (w, q, label, got, expected)


# ----------------------------------------------------------------------------
# Real metadata dataclasses (review addition): the verify classification
# reads fields by name; a rename would silently disable every lane route.
def _real_meta(cls, **values):
    import dataclasses

    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in values:
            kwargs[field.name] = values.pop(field.name)
        elif (field.default is dataclasses.MISSING
              and field.default_factory is dataclasses.MISSING):
            kwargs[field.name] = None
    assert not values, f"unknown fields for {cls.__name__}: {sorted(values)}"
    return cls(**kwargs)


def _metadata_classes():
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.short_conv_attn import (
            PleShortConvAttentionMetadata,
        )
    except Exception as exc:  # noqa: BLE001 - stubbed/partial vllm
        pytest.skip(f"attention metadata classes unavailable: {exc}")
    return FlashAttentionMetadata, GDNAttentionMetadata, PleShortConvAttentionMetadata


def test_real_metadata_field_contract():
    import dataclasses

    flash, gdn, ple = _metadata_classes()

    def names(cls):
        return {f.name for f in dataclasses.fields(cls)}

    counters = {"num_prefills", "num_prefill_tokens", "num_decodes",
                "num_decode_tokens", "num_spec_decodes", "num_spec_decode_tokens"}
    # QSA main attention (Qwen4ExpQSAMetadataBuilder builds FlashAttention
    # metadata): max_query_len + the runner's query_start_loc view, no
    # decode/prefill counters (so it cannot veto on its own).
    assert {"max_query_len", "query_start_loc"} <= names(flash)
    assert not counters & names(flash)
    # GDN: spec counters, no query_start_loc / max_query_len of its own.
    assert counters <= names(gdn)
    assert not {"query_start_loc", "max_query_len"} & names(gdn)
    # PLE short conv: spec counters + the shared query_start_loc view.
    assert counters | {"query_start_loc"} <= names(ple)
    assert "max_query_len" not in names(ple)
    # A non-spec PLE build leaves num_spec_decodes at 0, which never
    # classifies as a verify (plain-decode steps of the lane stay q = 1).
    spec_field = next(f for f in dataclasses.fields(ple) if f.name == "num_spec_decodes")
    assert spec_field.default == 0


@pytest.mark.parametrize("q", [2, 4, 5])
@pytest.mark.parametrize("reqs,live_reqs", [(1, 1), (4, 4), (6, 5), (8, 3)])
def test_real_metadata_verify_classification(q, reqs, live_reqs):
    flash, gdn_cls, ple_cls = _metadata_classes()
    width = reqs * q
    buf = torch.zeros(64, dtype=torch.int32)
    C.write_live(buf, live_reqs * q, q)
    view = buf[: reqs + 1]

    def attn(max_query_len=q, qsl=view):
        return _real_meta(flash, num_actual_tokens=width, max_query_len=max_query_len,
                          query_start_loc=qsl, max_seq_len=64, use_cascade=False,
                          common_prefix_len=0)

    def gdn(**over):
        vals = dict(num_prefills=0, num_prefill_tokens=0, num_decodes=0,
                    num_decode_tokens=0, num_spec_decodes=reqs,
                    num_spec_decode_tokens=width, num_actual_tokens=width)
        vals.update(over)
        return _real_meta(gdn_cls, **vals)

    def ple(**over):
        vals = dict(num_prefills=0, num_prefill_tokens=0, num_decodes=0,
                    num_decode_tokens=0, num_reqs=reqs, num_spec_decodes=reqs,
                    num_spec_decode_tokens=width, num_actual_tokens=width,
                    query_start_loc=view)
        vals.update(over)
        return _real_meta(ple_cls, **vals)

    def meta(a=None, g=None, p=None):
        a, g, p = a or attn(), g or gdn(), p or ple()
        return {"model.layers.3.self_attn.attn": a, "model.layers.0.linear_attn": g,
                "model.layers.1.linear_attn": g, "model.layers.2.ple": p}

    ok = meta()
    assert moe._uniform_verify_metadata(ok, q)
    live = moe._find_live_rows_view(ok, width, CPU, q)
    assert live is not None and int(live[0]) == live_reqs * q
    # Rejections: prefill in the batch, plain decodes mixed in, fewer drafts,
    # another verify width, a plain (non-spec) PLE build.
    assert not moe._uniform_verify_metadata(
        meta(g=gdn(num_prefills=1, num_prefill_tokens=q)), q)
    assert not moe._uniform_verify_metadata(
        meta(g=gdn(num_decodes=1, num_decode_tokens=1)), q)
    assert not moe._uniform_verify_metadata(
        meta(g=gdn(num_spec_decode_tokens=width - 1)), q)
    assert not moe._uniform_verify_metadata(meta(a=attn(max_query_len=q + 1)), q)
    assert not moe._uniform_verify_metadata(
        meta(p=ple(num_spec_decodes=0, num_spec_decode_tokens=0,
                   num_decodes=reqs, num_decode_tokens=reqs)), q)
    # A private copy of query_start_loc anywhere disables masking (fail
    # closed: every row computed), never a wrong live count.
    private = torch.zeros(reqs + 1, dtype=torch.int32)
    assert moe._find_live_rows_view(meta(p=ple(query_start_loc=private)),
                                    width, CPU, q) is None


def test_cudagraph_capture_invariants():
    """PIECEWISE captures run without attention metadata (so no verify route
    or live-row view is ever baked into a PIECEWISE graph, which also serves
    mixed batches); FULL uniform-decode descriptors pad requests to W / q."""
    src = _source("vllm.v1.worker.gpu.cudagraph_utils",
                  "vllm/v1/worker/gpu/cudagraph_utils.py")
    assert "skip_attn=(desc.cg_mode==CUDAGraphMode.PIECEWISE)" in src
    assert "ifcg_mode==CUDAGraphMode.PIECEWISE:assertattn_metadataisNone" in src
    assert "num_reqs=num_tokens//query_len" in src
    assert "andnum_tokens%query_len==0" in src


# ----------------------------------------------------------------------------
# direct QPN / MTP5
@pytest.fixture
def plain_envs(monkeypatch):
    from vllm import envs as real

    proxy = C.EnvsProxy(
        real,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE=True,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=False,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE=False,
        VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2=True,
    )
    monkeypatch.setattr(moe, "envs", proxy)
    # The fused W2 reduce op exists in 1.8.0-dev2; pin it so the route-table
    # labels do not depend on the extension the test process loaded.
    monkeypatch.setattr(
        moe.sm70_ops, "has_nvfp4_qpn_w2_reduce_dispatch", lambda: True
    )
    return proxy


DYNAMIC = {2: 10, 3: 8, 4: 5, 5: 4, 6: 8, 7: 8, 8: 4, 9: 4, 10: 4, 11: 4,
           12: 4, 13: 5, 14: 5, 15: 4, 16: 1}
STATIC = {2: 10, 4: 5, 8: 4, 16: 1}


@pytest.mark.parametrize("q", range(2, 9))
def test_qpn_batch_split_lane(plain_envs, q):
    layer = _lane_layer(q)
    buf = torch.zeros(80, dtype=torch.int32)
    for w in range(1, 33):
        if w % q == 0:
            with C.forward_context(moe, C.verify_metadata(buf[: w // q + 1], w, q)):
                assert moe._qwen38_qpn_batch_split(layer, w) == DYNAMIC.get(w), w
        with C.forward_context(moe, C.decode_metadata(buf[: w + 1], w)):
            assert moe._qwen38_qpn_batch_split(layer, w) == STATIC.get(w), w
        with C.forward_context(moe, C.mixed_metadata(buf[:3], q)):
            assert moe._qwen38_qpn_batch_split(layer, w) == STATIC.get(w), w
        assert moe._qwen38_qpn_batch_split(layer, w) == STATIC.get(w)  # no ctx
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe, "envs", C.EnvsProxy(
            plain_envs, VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=True))
        # An explicit global dynamic opt-in keeps its shape-only meaning.
        for w in range(1, 33):
            assert moe._qwen38_qpn_batch_split(_dev2_layer(), w) == DYNAMIC.get(w)


def test_qpn_mtp5_lane(plain_envs, monkeypatch):
    layer = _lane_layer(5)
    buf = torch.zeros(80, dtype=torch.int32)
    x, ids = _xi(5)
    with C.forward_context(moe, C.verify_metadata(buf[:2], 5, 5)):
        assert moe._use_qwen38_qpn_mtp5_decode(layer, x, ids)
        assert moe._use_qwen38_qpn_batch_decode(layer, x, ids)  # fused W2 too
        assert not moe._use_qwen38_qpn_mtp5_decode(_dev2_layer(), x, ids)
        assert not moe._use_qwen38_qpn_mtp5_decode(_lane_layer(5, mtp5=False), x, ids)
        unstamped = _lane_layer(5)
        unstamped.sx_mtp_verify_q = 0  # SX_OPT_MTP_MOE_ROUTES=0 build
        assert not moe._use_qwen38_qpn_mtp5_decode(unstamped, x, ids)
        assert not moe._use_qwen38_qpn_batch_decode(unstamped, x, ids)
    with C.forward_context(moe, C.mixed_metadata(buf[:3], 5)):
        assert not moe._use_qwen38_qpn_mtp5_decode(layer, x, ids)
        assert not moe._use_qwen38_qpn_batch_decode(layer, x, ids)
    assert not moe._use_qwen38_qpn_mtp5_decode(layer, x, ids)  # no context
    monkeypatch.setattr(moe, "envs", C.EnvsProxy(
        plain_envs, VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE=True))
    assert moe._use_qwen38_qpn_mtp5_decode(_dev2_layer(), x, ids)  # global opt-in
    x6, ids6 = _xi(6)
    assert not moe._use_qwen38_qpn_mtp5_decode(layer, x6, ids6)


EXPECTED_TABLES = {
    2: {2: "qpn-split10", 4: "qpn-split5", 6: "qpn-split8", 8: "grouped-split4",
        10: "grouped-split4", 12: "grouped-split4", 14: "grouped-split8",
        16: "grouped-split8", **{w: "grouped-split8" for w in range(18, 33, 2)},
        34: "turbomind", 36: "turbomind", 38: "turbomind", 40: "turbomind"},
    3: {3: "qpn-split8", 6: "qpn-split8", 9: "grouped-split4",
        12: "grouped-split4", 15: "grouped-split8",
        **{w: "grouped-split8" for w in range(18, 31, 3)},
        33: "turbomind", 36: "turbomind", 39: "turbomind"},
    4: {4: "qpn-split5", 8: "grouped-split4", 12: "grouped-split4",
        16: "grouped-split8", **{w: "grouped-split8" for w in range(20, 33, 4)},
        36: "turbomind", 40: "turbomind"},
    5: {5: "qpn-mtp5-split4+batch-w2", 10: "grouped-split4", 15: "grouped-split8",
        20: "grouped-split8", 25: "grouped-split8", 30: "grouped-split8",
        35: "turbomind", 40: "turbomind"},
    6: {6: "qpn-split8", 12: "grouped-split4", 18: "grouped-split8",
        24: "grouped-split8", 30: "grouped-split8", 36: "turbomind"},
    7: {7: "qpn-split8", 14: "grouped-split8", 21: "grouped-split8",
        28: "grouped-split8", 35: "turbomind"},
    8: {8: "grouped-split4", 16: "grouped-split8", 24: "grouped-split8",
        32: "grouped-split8", 40: "turbomind"},
}


@pytest.mark.parametrize("q", range(2, 9))
def test_route_table(plain_envs, q):
    assert moe._mtp_verify_route_table(_lane_layer(q)) == EXPECTED_TABLES[q]
    assert moe._mtp_verify_route_table(_dev2_layer()) == {}


def test_route_table_direct_off(plain_envs):
    table = moe._mtp_verify_route_table(_lane_layer(5, dynamic=False, mtp5=False))
    assert table[5] == "turbomind"
    table = moe._mtp_verify_route_table(_lane_layer(5, dynamic=False))
    assert table[5] == "qpn-mtp5-split4"  # 1Cat's MTP5 with the separate W2


def test_route_table_w5_label_follows_fused_w2(plain_envs, monkeypatch):
    """'+batch-w2' only when apply() really takes the fused W2 reduce (same
    gates as _use_qwen38_qpn_batch_fused_w2); otherwise W5 is 1Cat's MTP5
    route (MTP5 op for W2 plus the Triton weighted reduce)."""
    x, ids = _xi(5)
    buf = torch.zeros(8, dtype=torch.int32)
    layer = _lane_layer(5)
    for fused_env, fused_op in ((True, True), (False, True), (True, False)):
        monkeypatch.setattr(moe, "envs", C.EnvsProxy(
            plain_envs, VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2=fused_env))
        monkeypatch.setattr(
            moe.sm70_ops, "has_nvfp4_qpn_w2_reduce_dispatch", lambda v=fused_op: v
        )
        label = moe._mtp_verify_route_table(layer)[5]
        with C.forward_context(moe, C.verify_metadata(buf[:2], 5, 5)):
            fused = moe._use_qwen38_qpn_batch_fused_w2(layer, x, ids)
            assert moe._use_qwen38_qpn_mtp5_decode(layer, x, ids)
        assert fused == (fused_env and fused_op)
        assert label == ("qpn-mtp5-split4+batch-w2" if fused else "qpn-mtp5-split4")


def test_env_parsers(monkeypatch):
    for raw, exp in (("8", 8), (" 5 ", 5), ("1", 8), ("33", 8), ("x", 8), ("32", 32)):
        monkeypatch.setenv("SX_OPT_MTP_MOE_GROUPED_MIN_TOKENS", raw)
        assert moe._mtp_grouped_min_tokens_from_env() == exp, raw
    for raw, exp in (("4", 4), ("8", 8), ("5", 5), ("3", 4), ("", 4)):
        monkeypatch.setenv("SX_OPT_MTP_MOE_GROUPED_SPLIT_SMALL", raw)
        assert moe._mtp_grouped_split_small_from_env() == exp, raw


# ----------------------------------------------------------------------------
# V2 runner invariants for the verify masking
def _source(module: str, rel: str) -> str:
    path = None
    try:
        spec = importlib.util.find_spec(module)
        if spec is not None and spec.origin:
            path = Path(spec.origin)
    except (ImportError, ValueError):
        path = None
    if path is None or not path.exists():
        path = C.REPO / rel
    return re.sub(r"\s+", "", path.read_text(encoding="utf-8"))


def test_runner_verify_capture_invariants():
    src = _source("vllm.v1.worker.gpu.model_runner", "vllm/v1/worker/gpu/model_runner.py")
    # FULL uniform verify dummies/captures use W / q requests ...
    assert "num_reqs=num_tokens//self.decode_query_len" in src
    # ... and every step pads the tail with the live token count.
    assert "query_start_loc_np[num_reqs+1:]=num_tokens" in src
    assert (
        "query_start_loc=self.input_buffers.query_start_loc[:num_reqs_padded+1]"
        in src
    )


@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_make_dummy_verify_tail(q):
    try:
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    except Exception as exc:  # noqa: BLE001 - stubbed/partial vllm
        pytest.skip(f"V2 runner InputBatch unavailable: {exc}")
    buffers = InputBuffers(max_num_reqs=24, max_num_tokens=128, device=CPU)
    buffers.query_start_loc.fill_(-1)
    for reqs in (6, 4, 1, 3):
        tokens = reqs * q
        InputBatch.make_dummy(reqs, tokens, buffers)
        qsl = buffers.query_start_loc
        assert qsl[: reqs + 1].tolist() == [i * q for i in range(reqs + 1)]
        view = moe._find_live_rows_view(
            C.verify_metadata(qsl[: reqs + 1], tokens, q), tokens, CPU, q
        )
        assert view is not None and int(view[0]) == tokens


# ----------------------------------------------------------------------------
# custom all-reduce gating (25-KiB MTP5 push)
@pytest.fixture
def car(monkeypatch):
    from vllm.distributed.device_communicators import custom_all_reduce as mod

    monkeypatch.delenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5", raising=False)
    monkeypatch.delenv("SX_OPT_MTP_MOE_ROUTES", raising=False)
    monkeypatch.delenv("SX_OPT_MTP_LANE", raising=False)
    return mod


def _set_current(monkeypatch, cfg):
    import vllm.config.vllm as vcfg

    monkeypatch.setattr(vcfg, "get_current_vllm_config_or_none", lambda: cfg)


def test_push_ar_mtp5_gate(car, monkeypatch):
    _set_current(monkeypatch, C.lane_config(4))
    assert car._sx_mtp_lane_push_ar_mtp5() is True
    _set_current(monkeypatch, C.lane_config(2))
    assert car._sx_mtp_lane_push_ar_mtp5() is False  # no M5 verify
    _set_current(monkeypatch, _bad(method="eagle"))
    assert car._sx_mtp_lane_push_ar_mtp5() is False
    _set_current(monkeypatch, C.nomtp_config())
    assert car._sx_mtp_lane_push_ar_mtp5() is False
    _set_current(monkeypatch, None)
    assert car._sx_mtp_lane_push_ar_mtp5() is False
    _set_current(monkeypatch, C.lane_config(4))
    monkeypatch.setenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5", "0")
    assert car._sx_mtp_lane_push_ar_mtp5() is False  # explicit value wins
    monkeypatch.delenv("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5")
    monkeypatch.setenv("SX_OPT_MTP_MOE_ROUTES", "0")
    assert car._sx_mtp_lane_push_ar_mtp5() is False
    monkeypatch.delenv("SX_OPT_MTP_MOE_ROUTES")
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    assert car._sx_mtp_lane_push_ar_mtp5() is False
    monkeypatch.delenv("SX_OPT_MTP_LANE")
    assert car._sx_mtp_lane_push_ar_mtp5() is True
