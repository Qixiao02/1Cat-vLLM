# SPDX-License-Identifier: Apache-2.0
"""Upstream 1Cat ports, group "mtp-port-moe": host-side gates, no GPU.

Run (CPU is enough, no vLLM install needed; see ``port_boot.py``):

    python -m pytest -q sx_tests/mtp-port-moe/test_port_cpu.py

The same stubs run the ported upstream CPU tests unchanged (their GPU cases
skip; ``--noconftest`` keeps tests/conftest.py, which imports vLLM, out):

    PYTHONPATH=sx_tests/mtp-port-moe python -m pytest -q --noconftest \\
        -p port_boot \\
        tests/quantization/test_sm70_nvfp4_grouped_decode_dispatch.py \\
        tests/kernels/moe/test_sm70_mtp_moe_fp16.py

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
Item 2 (exact FP16 draft MoE projections, SX_OPT_MTP_MOE_FP16_EXACT):
* arming: only the native-MTP lane drafter with the switch on and
  VLLM_SM70_MTP_MOE_FP16_EXACT unset; an explicit "1"/"0" decides globally;
* the native op runs for M1/M5 W13 and W2 with upstream's operand order, and
  is refused (nothing launched) for M2/M10, any other tile, sorted
  assignment, quantized/biased/blocked experts, BF16 compute, missing
  weights, int64 ids, the legacy-config warmup,
  VLLM_SM70_MTP_MOE_TUNED_CONFIG=0, batch invariance, a non-SM70 device or a
  build without the op;
* dispatch_fused_moe_kernel launches exactly one of native / Triton;
* TritonExperts tries the native op first with the same operands and keeps
  its Triton calls (W13 still without routing weights); the drafter arms
  the op only under _is_sm70_qwen38_mtp_lane_contract (SX_OPT_MTP_LANE=0,
  other methods, k > 7, TP2 and partial configs refuse).
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


# ----------------------------------------------------------------------------
# item 2: exact FP16 draft MoE projections (SX_OPT_MTP_MOE_FP16_EXACT)
fused_moe = MODULES["fused_moe"]
FP16_EXACT_ENV = "VLLM_SM70_MTP_MOE_FP16_EXACT"
TILE = dict(BLOCK_SIZE_M=2, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1,
            SPLIT_K=1, num_warps=4, num_stages=3)


@pytest.fixture
def draft_moe(monkeypatch):
    """SM70 platform, a recording native op, nothing armed, envs unset."""
    for name in (FP16_EXACT_ENV, "VLLM_BATCH_INVARIANT",
                 "VLLM_SM70_MTP_MOE_TUNED_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    getattr(envs, "disable_envs_cache", lambda: None)()
    monkeypatch.setattr(fused_moe, "_SX_OPT_MTP_MOE_FP16_EXACT", True)
    monkeypatch.setattr(fused_moe, "_sx_mtp_moe_fp16_exact_armed", False)
    monkeypatch.setattr(fused_moe, "_force_sm70_mtp_moe_legacy_config", False)
    monkeypatch.setattr(fused_moe, "current_platform",
                        NS(is_device_capability=lambda cap, *a, **k: cap == 70))
    calls = []
    monkeypatch.setattr(torch.ops._C, "sm70_mtp_moe_fp16_out",
                        lambda *args: calls.append(args), raising=False)
    return calls


def _meta(*shape, dtype=torch.float16):
    return fake_cuda(torch.empty(*shape, device="meta", dtype=dtype))


def draft_args(m: int, down: bool, **over):
    """Positional/keyword arguments of sm70_mtp_moe_fp16_dispatch for the
    drafter's TritonExperts call (naive block assignment)."""
    n, k = (2560, 160) if down else (320, 2560)
    args = dict(
        A=_meta(m * 10 if down else m, k),
        B=_meta(512, n, k),
        C=_meta(m, 10, n),
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=_meta(m, 10, dtype=torch.float32),
        sorted_token_ids=None,
        expert_ids=_meta(m * 10, dtype=torch.int32),
        num_tokens_post_padded=_meta(1, dtype=torch.int32),
        mul_routed_weight=down,
        top_k=1 if down else 10,
        config=dict(TILE),
        compute_type=fused_moe.tl.float16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        block_shape=None,
        B_bias=None,
    )
    args.update(over)
    return args


def test_fp16_exact_arm(monkeypatch, draft_moe):
    assert fused_moe.arm_sm70_mtp_draft_moe_fp16(True)
    assert not fused_moe.arm_sm70_mtp_draft_moe_fp16(False)  # not the lane
    monkeypatch.setattr(fused_moe, "_SX_OPT_MTP_MOE_FP16_EXACT", False)
    assert not fused_moe.arm_sm70_mtp_draft_moe_fp16(True)
    monkeypatch.setattr(fused_moe, "_SX_OPT_MTP_MOE_FP16_EXACT", True)
    for value in ("0", "1"):  # an explicit value decides globally
        monkeypatch.setenv(FP16_EXACT_ENV, value)
        assert not fused_moe.arm_sm70_mtp_draft_moe_fp16(True)


@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("down", [False, True])
def test_fp16_exact_lane_admits_m1_m5(draft_moe, m, down):
    args = draft_args(m, down)
    assert not fused_moe.sm70_mtp_moe_fp16_dispatch(**args)  # not armed
    assert not draft_moe
    fused_moe.arm_sm70_mtp_draft_moe_fp16(True)
    assert fused_moe.sm70_mtp_moe_fp16_dispatch(**args)
    (call,) = draft_moe
    assert call[0] is args["C"] and call[1] is args["A"] and call[2] is args["B"]
    assert call[3] is args["expert_ids"] and call[4] is args["topk_weights"]
    assert call[5] is args["num_tokens_post_padded"] and call[6] is down


@pytest.mark.parametrize(
    "case",
    ["m2", "m10", "tile_bm16", "tile_bk32", "tile_w8", "sorted", "fp8",
     "int4", "a_scale", "bias", "block_shape", "bf16_compute", "no_weights",
     "ids_int64", "legacy_config", "tuned_off", "batch_invariant", "platform",
     "no_op"],
)
def test_fp16_exact_refusals(monkeypatch, draft_moe, case):
    fused_moe.arm_sm70_mtp_draft_moe_fp16(True)
    m, down, over = 5, False, {}
    if case == "m2":
        m = 2
    elif case == "m10":
        m = 10
    elif case.startswith("tile_"):
        key, value = {"tile_bm16": ("BLOCK_SIZE_M", 16),
                      "tile_bk32": ("BLOCK_SIZE_K", 32),
                      "tile_w8": ("num_warps", 8)}[case]
        over["config"] = dict(TILE, **{key: value})
    elif case == "sorted":
        over["sorted_token_ids"] = _meta(64, dtype=torch.int32)
    elif case == "fp8":
        over["use_fp8_w8a8"] = True
    elif case == "int4":
        over["use_int4_w4a16"] = True
    elif case == "a_scale":
        over["A_scale"] = _meta(1, dtype=torch.float32)
    elif case == "bias":
        over["B_bias"] = _meta(512, 320)
    elif case == "block_shape":
        over["block_shape"] = [128, 128]
    elif case == "bf16_compute":
        over["compute_type"] = fused_moe.tl.bfloat16
    elif case == "no_weights":
        over["topk_weights"] = None
    elif case == "ids_int64":
        over["expert_ids"] = _meta(50, dtype=torch.int64)
    elif case == "legacy_config":
        monkeypatch.setattr(fused_moe, "_force_sm70_mtp_moe_legacy_config", True)
    elif case == "tuned_off":
        monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "0")
    elif case == "batch_invariant":
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    elif case == "platform":
        monkeypatch.setattr(fused_moe, "current_platform",
                            NS(is_device_capability=lambda cap, *a, **k: False))
    else:
        monkeypatch.delattr(torch.ops._C, "sm70_mtp_moe_fp16_out")
    getattr(envs, "disable_envs_cache", lambda: None)()
    args = draft_args(m, down, **over)
    assert not fused_moe.sm70_mtp_moe_fp16_dispatch(**args)
    assert not draft_moe


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False)])
def test_fp16_exact_explicit_env_is_global(monkeypatch, draft_moe, value, expected):
    monkeypatch.setenv(FP16_EXACT_ENV, value)
    getattr(envs, "disable_envs_cache", lambda: None)()
    fused_moe.arm_sm70_mtp_draft_moe_fp16(True)  # the lane does not override
    assert fused_moe.sm70_mtp_moe_fp16_dispatch(**draft_args(1, False)) is expected


def test_fp16_exact_dispatch_fused_moe_kernel(monkeypatch, draft_moe):
    triton_calls = []
    monkeypatch.setattr(fused_moe, "invoke_fused_moe_triton_kernel",
                        lambda *a, **k: triton_calls.append(a))
    fused_moe.arm_sm70_mtp_draft_moe_fp16(True)
    for m, expected in ((1, 1), (5, 1), (2, 0)):
        draft_moe.clear()
        triton_calls.clear()
        args = draft_args(m, True)
        fused_moe.dispatch_fused_moe_kernel(**args, per_channel_quant=False)
        assert len(draft_moe) == expected and len(triton_calls) == 1 - expected


def _function(path: str, name: str, cls: str | None = None):
    import ast

    tree = ast.parse(open(os.path.join(port_boot.REPO, path), encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path}: {name} not found")


@pytest.mark.parametrize("fn,weights_arg", [("_base_w13_fn", None),
                                            ("_base_w2_fn", "topk_weights")])
def test_triton_experts_try_native_first(fn, weights_arg):
    """TritonExperts (upstream 6bcffbb79): the exact dispatch is tried first
    with the routing weights; the unchanged Triton call follows it, W13 still
    without routing weights."""
    import ast

    node = _function("vllm/model_executor/layers/fused_moe/experts/triton_moe.py", fn)
    first, second = node.body[0], node.body[1]
    assert isinstance(first, ast.If)
    assert first.test.func.id == "sm70_mtp_moe_fp16_dispatch"
    assert first.test.args[6].id == "topk_weights"
    assert isinstance(first.body[0], ast.Return) and not first.orelse
    assert second.value.func.id == "invoke_fused_moe_triton_kernel"
    arg = second.value.args[5]
    if weights_arg is None:
        assert isinstance(arg, ast.Constant) and arg.value is None
    else:
        assert arg.id == weights_arg
    # Same operands for both calls (A, B, C, scales; assignment metadata).
    native, triton = first.test.args, second.value.args
    assert [ast.unparse(a) for a in native[:5]] == [ast.unparse(a) for a in triton[:5]]
    assert [ast.unparse(a) for a in native[7:12]] == [
        ast.unparse(a) for a in triton[6:11]
    ]


def test_mtp_draft_arms_fp16_exact_in_lane_only():
    import ast

    node = _function("vllm/models/qwen4_exp/nvidia/mtp.py", "_sx_prepare_mtp_draft_sm70")
    calls = [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Call)]
    assert "arm_sm70_mtp_draft_moe_fp16(_sx_mtp_lane_contract(vllm_config))" in calls
    source = ast.unparse(_function("vllm/models/qwen4_exp/nvidia/mtp.py",
                                   "_sx_mtp_lane_contract"))
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102
    lane = namespace["_sx_mtp_lane_contract"]
    torch_fp16 = torch.float16
    text = NS(hidden_size=2560, num_hidden_layers=48, num_experts=512,
              num_experts_per_tok=10, moe_intermediate_size=640, hc_count=4,
              hc_lowrank=320, num_attention_heads=24, num_key_value_heads=2,
              indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4)
    target = NS(hf_text_config=text, architectures=["Qwen4ExpForCausalLM"],
                multimodal_config=None, dtype=torch_fp16)
    parallel = NS(tensor_parallel_size=4, pipeline_parallel_size=1)

    def spec(method="mtp", k=4):
        return NS(method=method, num_speculative_tokens=k,
                  use_qwen4_exp_mtp=lambda: True,
                  num_speculative_state_tokens=lambda: k,
                  parallel_drafting=False, rejection_sample_method="standard",
                  target_model_config=target)

    assert lane(NS(model_config=target, speculative_config=spec(),
                   parallel_config=parallel))
    assert not lane(NS(model_config=target, speculative_config=spec("eagle"),
                       parallel_config=parallel))
    assert not lane(NS(model_config=target, speculative_config=spec(k=8),
                       parallel_config=parallel))
    assert not lane(NS(model_config=target, speculative_config=spec(),
                       parallel_config=NS(tensor_parallel_size=2,
                                          pipeline_parallel_size=1)))
    assert not lane(NS())  # partial config fails closed
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SX_OPT_MTP_LANE", "0")
        assert not lane(NS(model_config=target, speculative_config=spec(),
                           parallel_config=parallel))
