# SPDX-License-Identifier: Apache-2.0
"""Upstream 1Cat ports, group "mtp-port-moe": host-side gates, no GPU.

Run (CPU is enough, no vLLM install needed; see ``port_boot.py``):

    python -m pytest -q sx_tests/mtp-port-moe/test_port_cpu.py

The same stubs run the ported upstream CPU tests unchanged (their GPU cases
skip; ``--noconftest`` keeps tests/conftest.py, which imports vLLM, out):

    PYTHONPATH=sx_tests/mtp-port-moe python -m pytest -q --noconftest \\
        -p port_boot \\
        tests/quantization/test_sm70_nvfp4_grouped_decode_dispatch.py

Inside the image (vLLM installed, the group's files bind-mounted) the real
modules are used and the same commands work.

Asserted here, item 1 (grouped MTP5 experts, SX_OPT_MTP_MOE_GROUPED_MTP5):
* load-time admission: only lane layers built for k = 4 whose W5 verify is
  on the QPN-MTP5 route, TP4 and the grouped-kernel contract; the switch,
  k != 4, a missing MTP5 route, TP2 or an unsupported layer refuse; an
  explicitly set VLLM_SM70_NVFP4_MOE_GROUPED_MTP5 keeps upstream's global
  meaning ("1" admits a no-MTP layer only together with the MTP5 env, "0"
  refuses the lane);
* dispatch: W5 in a uniform verify forward only (not mixed / decode / no
  context, not W10 or any other width, not an unstamped layer);
* route table: W5 says "grouped-mtp5-split4+batch-reduce" exactly when the
  layer carries the admission, every other width is unchanged (also with
  SX_OPT_MTP_MOE_GROUPED_MIN_TOKENS <= 5, where the ported route wins W5);
* apply(): the W5 verify calls the grouped W13 at split 4 (the QPN-MTP5
  split, interleaved flag passed through) and the batch-reduce W2 on the
  layer-owned group metadata, writes the layer's output buffer and calls
  nothing else; W10 keeps the v2 grouped route.
Expected: all pass.
"""

from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import port_boot  # noqa: E402

MODULES = port_boot.install()
moe = MODULES["moe"]
from vllm import envs  # noqa: E402

MTP5_ENV = "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE"
GROUPED_MTP5_ENV = "VLLM_SM70_NVFP4_MOE_GROUPED_MTP5"


@pytest.fixture(autouse=True)
def plain_envs(monkeypatch):
    """Every switch of this group at its default; env caches off."""
    for name in (
        MTP5_ENV,
        GROUPED_MTP5_ENV,
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE",
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE",
        "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2",
    ):
        monkeypatch.delenv(name, raising=False)
    getattr(envs, "disable_envs_cache", lambda: None)()
    monkeypatch.setattr(moe, "_SX_OPT_MTP_MOE_GROUPED_MTP5", True)
    yield


# ----------------------------------------------------------------------------
# metadata (V2 runner semantics, as in sx_tests/b3-moe-verify/_mtp_common.py)
def verify_metadata(qsl_view, tokens: int, q: int, *, prefills: int = 0):
    reqs = tokens // q
    qsa = NS(max_query_len=q, query_start_loc=qsl_view)
    gdn = NS(
        num_prefills=prefills, num_prefill_tokens=prefills * 7,
        num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=reqs, num_spec_decode_tokens=reqs * q,
    )
    return {"layers.3.self_attn": qsa, "layers.0.linear_attn": gdn}


def decode_metadata(qsl_view, tokens: int):
    gdn = NS(
        num_prefills=0, num_prefill_tokens=0, num_decodes=tokens,
        num_decode_tokens=tokens, num_spec_decodes=0, num_spec_decode_tokens=0,
    )
    return {"layers.3.self_attn": NS(max_query_len=1, query_start_loc=qsl_view),
            "layers.0.linear_attn": gdn}


def qsl(tokens: int, q: int) -> torch.Tensor:
    return (torch.arange(tokens // q + 1, dtype=torch.int32) * q).clamp_(max=tokens)


@contextlib.contextmanager
def forward_context(metadata):
    ctx = NS(attn_metadata=metadata, additional_kwargs={})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe, "is_forward_context_available", lambda: True)
        mp.setattr(moe, "get_forward_context", lambda: ctx)
        yield ctx


def xi(tokens: int):
    return (
        torch.empty(tokens, 2560, dtype=torch.float16),
        torch.empty(tokens, 10, dtype=torch.int32),
    )


def lane_layer(q: int = 5, **over):
    """A layer as process_weights_after_loading stamps it in the k = q-1 lane
    (VLLM_SM70_NVFP4_MOE_GROUPED_DECODE=1, v2 ops present)."""
    fields = dict(
        moe_config=NS(tp_size=4),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_grouped_decode=True,
        sm70_nvfp4_grouped_max_tokens=32,
        sm70_nvfp4_grouped32_split=8,
        sm70_nvfp4_grouped_mask=True,
        _nvfp4_grouped_v2_ops=("w13_v2", "w2_v2"),
        sx_mtp_verify_q=q,
        sx_mtp_grouped=True,
        sx_mtp_grouped_min_tokens=8,
        sx_mtp_grouped_split_small=4,
        sx_mtp_qpn_dynamic=True,
        sx_mtp_qpn_mtp5=q == 5,
        sm70_nvfp4_grouped_mtp5=q == 5,
    )
    fields.update(over)
    return NS(**fields)


def nomtp_layer(**over):
    fields = dict(
        moe_config=NS(tp_size=4),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
        sx_mtp_verify_q=0,
        sx_mtp_qpn_mtp5=False,
    )
    fields.update(over)
    return NS(**fields)


# ----------------------------------------------------------------------------
# item 1: load-time admission
def test_grouped_mtp5_lane_default():
    assert moe._grouped_mtp5_requested(lane_layer(), True) == (True, False)


@pytest.mark.parametrize(
    "case",
    ["switch_off", "k3", "k5", "no_mtp5_route", "tp2", "unsupported", "nomtp"],
)
def test_grouped_mtp5_lane_refusals(monkeypatch, case):
    layer, supported = lane_layer(), True
    if case == "switch_off":
        monkeypatch.setattr(moe, "_SX_OPT_MTP_MOE_GROUPED_MTP5", False)
    elif case == "k3":
        layer = lane_layer(4)
    elif case == "k5":
        layer = lane_layer(6)
    elif case == "no_mtp5_route":
        layer.sx_mtp_qpn_mtp5 = False  # SX_OPT_MTP_MOE_DIRECT=0 / op absent
    elif case == "tp2":
        layer.moe_config.tp_size = 2
    elif case == "unsupported":
        supported = False  # raw scales, SwiGLU limit or another shape
    else:
        layer = nomtp_layer()
    assert moe._grouped_mtp5_requested(layer, supported) == (False, False)


@pytest.mark.parametrize(
    "grouped_env,mtp5_env,lane,expected",
    [
        ("1", "1", False, True),  # upstream: global opt-in with MTP5
        ("1", None, False, False),  # upstream: needs the MTP5 route
        ("1", None, True, True),  # lane MTP5 default satisfies it
        ("0", None, True, False),  # explicit off wins over the lane
        ("0", "1", True, False),
    ],
)
def test_grouped_mtp5_explicit_env(monkeypatch, grouped_env, mtp5_env, lane,
                                   expected):
    monkeypatch.setenv(GROUPED_MTP5_ENV, grouped_env)
    if mtp5_env is not None:
        monkeypatch.setenv(MTP5_ENV, mtp5_env)
    layer = lane_layer() if lane else nomtp_layer()
    if mtp5_env is not None:
        layer.sx_mtp_qpn_mtp5 = False  # set only when the env is absent
    assert moe._grouped_mtp5_requested(layer, True) == (expected, True)


# ----------------------------------------------------------------------------
# item 1: dispatch
def test_grouped_mtp5_dispatch_verify_only():
    layer = lane_layer()
    x, ids = xi(5)
    with forward_context(verify_metadata(qsl(5, 5), 5, 5)):
        assert moe._use_grouped_mtp5(layer, x, ids)
        assert not moe._use_grouped_mtp5(lane_layer(sm70_nvfp4_grouped_mtp5=False),
                                         x, ids)
        assert not moe._use_grouped_mtp5(nomtp_layer(), x, ids)
        assert not moe._use_grouped_mtp5(NS(), x, ids)
    with forward_context(verify_metadata(qsl(10, 5), 10, 5, prefills=1)):
        assert not moe._use_grouped_mtp5(layer, x, ids)  # mixed step
    with forward_context(decode_metadata(qsl(5, 1), 5)):
        assert not moe._use_grouped_mtp5(layer, x, ids)  # 5 plain decodes
    assert not moe._use_grouped_mtp5(layer, x, ids)  # no forward context
    for width in (1, 2, 4, 8, 10, 15, 16, 20):
        xw, idw = xi(width)
        with forward_context(verify_metadata(qsl(width, 5), width, 5)):
            assert not moe._use_grouped_mtp5(layer, xw, idw), width


def test_grouped_mtp5_tensor_contract():
    layer = lane_layer()
    x, ids = xi(5)
    bad = [
        (x.float(), ids),
        (x, ids.long()),
        (torch.empty(2560, 5, dtype=torch.float16).t(), ids),
        (x, torch.empty(10, 5, dtype=torch.int32).t()),
    ]
    with forward_context(verify_metadata(qsl(5, 5), 5, 5)):
        for bx, bids in bad:
            assert not moe._use_grouped_mtp5(layer, bx, bids)


# ----------------------------------------------------------------------------
# item 1: route table
EXPECTED_K4 = {
    5: "grouped-mtp5-split4+batch-reduce", 10: "grouped-split4",
    15: "grouped-split8", 20: "grouped-split8", 25: "grouped-split8",
    30: "grouped-split8", 35: "turbomind", 40: "turbomind",
}


def test_route_table_k4(monkeypatch):
    # The fused W2 reduce op of the previous W5 label exists in the image.
    monkeypatch.setattr(
        moe.sm70_ops, "has_nvfp4_qpn_w2_reduce_dispatch", lambda: True
    )
    assert moe._mtp_verify_route_table(lane_layer()) == EXPECTED_K4
    table = moe._mtp_verify_route_table(lane_layer(sm70_nvfp4_grouped_mtp5=False))
    assert table[5] == "qpn-mtp5-split4+batch-w2"  # the previous lane route
    assert {w: r for w, r in table.items() if w != 5} == {
        w: r for w, r in EXPECTED_K4.items() if w != 5
    }


def test_route_table_low_grouped_min_tokens():
    """With SX_OPT_MTP_MOE_GROUPED_MIN_TOKENS <= 5 the fork's v2 grouped route
    would also claim W5; the ported route is checked first."""
    layer = lane_layer(sx_mtp_grouped_min_tokens=5)
    assert moe._mtp_verify_route_table(layer)[5] == EXPECTED_K4[5]
    x, ids = xi(5)
    with forward_context(verify_metadata(qsl(5, 5), 5, 5)):
        assert moe._use_grouped_mtp5(layer, x, ids)
        assert moe._grouped_decode_route(layer, x, ids) == (4, 5)


@pytest.mark.parametrize("q", [2, 3, 4, 6, 7, 8])
def test_route_table_other_k_unchanged(q):
    with_attr = moe._mtp_verify_route_table(lane_layer(q, sm70_nvfp4_grouped_mtp5=True))
    without = moe._mtp_verify_route_table(lane_layer(q, sm70_nvfp4_grouped_mtp5=False))
    assert with_attr == without
    assert not any("mtp5" in route for route in with_attr.values())


# ----------------------------------------------------------------------------
# item 1: apply()
class FakeCuda(torch.Tensor):
    """CPU tensor that passes apply()'s ``x.is_cuda`` guard."""

    @property
    def is_cuda(self):  # type: ignore[override]
        return True


def fake_cuda(t: torch.Tensor) -> torch.Tensor:
    return torch.Tensor._make_subclass(FakeCuda, t)


class Recorder:
    """Stands in for vllm._sm70_ops: records every op call."""

    def __init__(self, real):
        self.real = real
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        if name.startswith("has_"):
            return getattr(self.real, name)

        def record(*args, **kwargs):
            self.calls.append((name, args))

        return record


def _apply(monkeypatch, layer, tokens, metadata):
    ops = Recorder(moe.sm70_ops)
    monkeypatch.setattr(moe, "sm70_ops", ops)
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **k: True)
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    slots = tokens * 10
    bufs = {
        "output": torch.empty(tokens, 2560, dtype=torch.float16),
        "intermediate": torch.empty(slots, 160, dtype=torch.float16),
        "sorted_output": torch.empty(slots, 2560, dtype=torch.float16),
        "gate_up": torch.empty(slots, 320, dtype=torch.float16),
    }
    monkeypatch.setattr(
        moe.ModelOptNvFp4SM70MoEMethod, "_get_buffers",
        lambda self, layer, num_tokens, indexed: bufs,
    )
    v2_calls = []
    layer._nvfp4_grouped_v2_ops = (
        lambda *a: v2_calls.append(("w13_v2", a)),
        lambda *a: v2_calls.append(("w2_v2", a)),
    )
    for name, shape in (("rows", (320, 8)), ("experts", (320,)),
                        ("sizes", (320,)), ("total", (1,))):
        setattr(layer, f"_nvfp4_grouped_{name}", torch.empty(shape, dtype=torch.int32))
    layer.w13_tm_weight = torch.empty(1, dtype=torch.int32)
    layer.w13_tm_scales = torch.empty(1, dtype=torch.float16)
    layer.w2_tm_weight = torch.empty(1, dtype=torch.int32)
    layer.w2_tm_scales = torch.empty(1, dtype=torch.float16)
    layer.sm70_nvfp4_qwen38_fused_swiglu_prefill = True  # interleaved W13
    x = fake_cuda(torch.randn(tokens, 2560).half())
    ids = fake_cuda(torch.randint(0, 512, (tokens, 10), dtype=torch.int32))
    weights = fake_cuda(torch.rand(tokens, 10))
    with forward_context(metadata):
        out = method.apply(layer, x, weights, ids, None, None)
    return out, bufs, ops.calls, v2_calls, (x, weights, ids)


def test_apply_w5_grouped_mtp5(monkeypatch):
    layer = lane_layer()
    out, bufs, calls, v2_calls, (x, weights, ids) = _apply(
        monkeypatch, layer, 5, verify_metadata(qsl(5, 5), 5, 5)
    )
    assert out is bufs["output"]
    assert [name for name, _ in calls] == [
        "nvfp4_grouped_w13_sm70_out",
        "nvfp4_grouped_w2_batch_reduce_sm70_out",
    ]
    assert not v2_calls
    w13 = calls[0][1]
    assert w13[0] is bufs["intermediate"] and w13[1] is x
    assert w13[2] is layer.w13_tm_weight and w13[3] is layer.w13_tm_scales
    assert torch.equal(w13[4], ids.view(-1))
    assert w13[5:9] == (layer._nvfp4_grouped_rows, layer._nvfp4_grouped_experts,
                        layer._nvfp4_grouped_sizes, layer._nvfp4_grouped_total)
    assert w13[9] == 4 == moe._QWEN38_QPN_MTP5_W13_SPLIT_K
    assert w13[10] is True
    w2 = calls[1][1]
    assert w2[0] is bufs["output"] and w2[1] is bufs["sorted_output"]
    assert w2[2] is bufs["intermediate"]
    assert w2[3] is layer.w2_tm_weight and w2[4] is layer.w2_tm_scales
    assert w2[5] is weights
    assert w2[6:] == w13[5:9]


def test_apply_w10_keeps_v2_grouped(monkeypatch):
    layer = lane_layer()
    _, _, calls, v2_calls, _ = _apply(
        monkeypatch, layer, 10, verify_metadata(qsl(10, 5), 10, 5)
    )
    assert not calls
    assert [name for name, _ in v2_calls] == ["w13_v2", "w2_v2"]
    assert v2_calls[0][1][9] == 4  # SX_OPT_MTP_MOE_GROUPED_SPLIT_SMALL
