# SPDX-License-Identifier: Apache-2.0
"""Batch 3a "moe-verify": router top-k and shared-expert gate in the MTP lane.

GPUs: 1 x V100 (SM70). Needs the image's vllm._C (qwen38_shared_gate_exact_out
accepting (M, 2560), topk_softmax) with this group's files bind-mounted.

  /opt/venv/bin/python -m pytest -q sx_tests/b3-moe-verify/test_mtp_router_gate.py
  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-moe-verify/test_mtp_router_gate.py -k bench

Router (SX_OPT_ROUTER32 admitted in the lane, target and draft):
  * FusedTopKRouter built under the MTP-lane config (the real lane contract)
    arms the runtime-M route; built with SX_OPT_MTP_MOE_ROUTES=0 it does not
    (the 1.8.0-dev2 MTP lane).
  * at verify / draft widths 5, 10, 15, 20, 25, 30, 32 (and 40, above the
    route): expert ids and source rows equal ops.topk_softmax, weights within
    the admitted 3e-7; every row bitwise equal to the legacy constexpr-M
    kernel on <= 16-row chunks (the no-MTP lane's per-row result); widths
    <= 16 bitwise equal to the dev2 lane router; width 40 bitwise equal to
    topk_softmax (unchanged).
  * a CUDA graph of 48 routing calls at 20/30/32 replays bitwise equal to
    eager with new logits.
Shared-expert gate (SX_OPT_SHARED_GATE_ROWS admitted in the lane):
  * Qwen2MoeMLP constructed in the lane for the target and for the MTP draft
    prefix arms the rows gate (real capability probe);
  * forward at 5..32 rows uses the fused rows op and is bitwise equal to the
    M == 1 gate per row; 33 rows keeps the unfused path; the dev2 build keeps
    the unfused path for every M > 1;
  * CUDA-graph replay of the forward at 20/30 rows equals per-row M1.
Prints (-s): 48-call CUDA-graph rounds at the verify widths for dev2 vs lane
(router: topk_softmax/legacy vs runtime-M; gate: unfused vs fused rows).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.skipif(not C.ON_SM70, reason="requires an SM70 GPU")

E, K, LAYERS = 512, 10, 48
WIDTHS = (5, 10, 15, 20, 25, 30, 32)
TS_TOL = 3e-7


def _router_mod():
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router

    return fused_topk_router


def _build_router(monkeypatch, *, lane_switch=True, cfg=None):
    import vllm.config.vllm as vcfg

    mod = _router_mod()
    monkeypatch.setattr(
        vcfg, "get_current_vllm_config_or_none",
        lambda: C.lane_config(4) if cfg is None else cfg,
    )
    monkeypatch.setattr(mod, "_SX_OPT_MTP_MOE_ROUTES", lane_switch)
    monkeypatch.setattr(mod, "_SX_OPT_ROUTER32", True)
    monkeypatch.delenv("SX_OPT_MTP_LANE", raising=False)
    return mod.FusedTopKRouter(
        top_k=K, global_num_experts=E, scoring_func="softmax", renormalize=True
    )


@pytest.fixture
def routers(monkeypatch):
    lane = _build_router(monkeypatch)
    draft = _build_router(monkeypatch, cfg=C.lane_config(4, draft=True))
    dev2 = _build_router(monkeypatch, lane_switch=False)
    return NS(lane=lane, draft=draft, dev2=dev2)


def _logits(case, m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, E, generator=gen, device="cuda", dtype=torch.float32)
    if case == "scaled_small":
        x = x * 1e-3
    elif case == "scaled_large":
        x = x * 10.0
    x = x.half()
    if case == "ties":
        x[:, :16] = 4.0
        x[:, 200:230] = 4.0
    return x.contiguous()


def _route(router, logits):
    h = torch.empty(logits.shape[0], 2560, dtype=torch.float16, device="cuda")
    w, ids = router._compute_routing(h, logits, None)
    return w, ids


def _topk_softmax(logits):
    import vllm._custom_ops as ops

    m = logits.shape[0]
    w = torch.empty(m, K, dtype=torch.float32, device="cuda")
    ids = torch.empty(m, K, dtype=torch.int32, device="cuda")
    rows = torch.empty(m, K, dtype=torch.int32, device="cuda")
    ops.topk_softmax(w, ids, rows, logits, True)
    return w, ids, rows


def _legacy(logits):
    mod = _router_mod()
    m = logits.shape[0]
    w = torch.empty(m, K, dtype=torch.float32, device="cuda")
    ids = torch.empty(m, K, dtype=torch.int32, device="cuda")
    rows = torch.empty(m, K, dtype=torch.int32, device="cuda")
    mod._sm70_qwen38_router_topk(w, ids, rows, logits)
    return w, ids, rows


def _bits(a, b):
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def test_router_flags(routers):
    assert routers.lane._sm70_qwen38_router_runtime_m is True
    assert routers.draft._sm70_qwen38_router_runtime_m is True
    assert routers.dev2._sm70_qwen38_router_runtime_m is False


@pytest.mark.parametrize("case", ["random", "scaled_small", "scaled_large", "ties"])
@pytest.mark.parametrize("m", WIDTHS + (40,))
def test_router_ids_vs_topk_softmax(routers, m, case):
    x = _logits(case, m, seed=1000 + m)
    lw, lids = _route(routers.lane, x)
    dw, dids = _route(routers.draft, x)
    ow, oids = _route(routers.dev2, x)
    sw, sids, _ = _topk_softmax(x)
    torch.accelerator.synchronize()
    assert torch.equal(lids, sids), (m, case)
    torch.testing.assert_close(lw, sw, atol=TS_TOL, rtol=TS_TOL)
    assert _bits(lw, dw) and torch.equal(lids, dids)  # draft == target route
    assert torch.equal(oids, sids)
    if m <= 16:
        assert _bits(lw, ow) and torch.equal(lids, oids)  # bitwise vs dev2
    if m > 32:
        assert _bits(lw, sw) and _bits(lw, ow)  # unchanged generic route
    if m <= 32:
        half = max(1, m // 2) if m > 16 else m
        for lo, hi in ((0, half), (half, m)):
            if lo >= hi:
                continue
            cw, cids, _ = _legacy(x[lo:hi].contiguous())
            assert _bits(lw[lo:hi], cw) and torch.equal(lids[lo:hi], cids), (
                m, case, lo)


@pytest.mark.parametrize("m", [20, 30, 32])
def test_router_graph_replay(routers, m):
    x = torch.randn(LAYERS, m, E, device="cuda").half()
    h = torch.empty(m, 2560, dtype=torch.float16, device="cuda")

    def run():
        return [routers.lane._compute_routing(h, x[i], None) for i in range(LAYERS)]

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    for trial in range(6):
        case = ("random", "ties", "scaled_large")[trial % 3]
        x.copy_(torch.stack([_logits(case, m, 5000 + 97 * trial + i)
                             for i in range(LAYERS)]))
        for w, ids in outs:
            w.fill_(float("nan"))
            ids.fill_(-777)
        graph.replay()
        torch.accelerator.synchronize()
        for i in range(LAYERS):
            rw, rids = _route(routers.lane, x[i])
            assert _bits(outs[i][0], rw) and torch.equal(outs[i][1], rids), (
                trial, i)
    del graph


@pytest.mark.skipif(os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0"),
                    reason="SX_TEST_NO_BENCH")
def test_router_bench(routers):
    print("\n[b3-router] 48 routing calls in one CUDA graph, median of 60 (ms)")
    print(f"  {'M':>4}{'dev2':>9}{'lane':>9}{'saved':>9}")
    h = {m: torch.empty(m, 2560, dtype=torch.float16, device="cuda") for m in WIDTHS}
    for m in WIDTHS:
        x = _logits("random", m, 77 + m)
        old = C.graph_ms(lambda: [routers.dev2._compute_routing(h[m], x, None)
                                  for _ in range(LAYERS)])
        new = C.graph_ms(lambda: [routers.lane._compute_routing(h[m], x, None)
                                  for _ in range(LAYERS)])
        print(f"  {m:>4}{old:>9.4f}{new:>9.4f}{old - new:>+9.4f}")


# ----------------------------------------------------------------------------
# shared-expert gate
def _gate_ops():
    from vllm import _sm70_ops

    if not _sm70_ops.has_qwen38_shared_gate_exact():
        pytest.fail("qwen38_shared_gate_exact_out is not available in this build")
    return _sm70_ops


def _construct(monkeypatch, cfg, prefix, lane_switch=True):
    from vllm.model_executor.models import qwen2_moe

    mod = _router_mod()
    monkeypatch.setattr(mod, "_SX_OPT_MTP_MOE_ROUTES", lane_switch)
    monkeypatch.delenv("SX_OPT_MTP_LANE", raising=False)
    monkeypatch.setattr(
        qwen2_moe, "envs", NS(VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION=True)
    )
    monkeypatch.setattr(qwen2_moe, "MergedColumnParallelLinear", lambda *a, **k: NS())
    monkeypatch.setattr(qwen2_moe, "RowParallelLinear", lambda *a, **k: NS())
    monkeypatch.setattr(qwen2_moe, "SiluAndMul", lambda **k: NS())
    monkeypatch.setattr(
        qwen2_moe,
        "_sm70_fused_shared_expert_gate_module_supported",
        lambda gate_up, down: True,
    )
    monkeypatch.setattr(qwen2_moe, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(qwen2_moe, "_SX_OPT_SHARED_GATE_ROWS", True)
    monkeypatch.setattr(qwen2_moe, "_SX_SHARED_GATE_ROWS_CAPABLE", None)
    monkeypatch.setattr(qwen2_moe, "_sm70_dump_qwen_mlp_tensor",
                        lambda label, idx, t: t)
    gen = torch.Generator(device="cuda").manual_seed(4242)
    weight = (torch.randn(1, 2560, generator=gen, device="cuda") * 0.02).half()
    layer = qwen2_moe.Qwen2MoeMLP(
        hidden_size=2560,
        intermediate_size=160,
        hidden_act="silu",
        reduce_results=False,
        expert_gate=NS(weight=weight),
        prefix=prefix,
    )
    return qwen2_moe, layer, weight


class _Gate:
    """ReplicatedLinear stand-in: (F.linear(x, w), None) plus .weight."""

    def __init__(self, weight, calls):
        self.weight = weight
        self.calls = calls

    def __call__(self, inp):
        self.calls.append("unfused")
        return F.linear(inp, self.weight), None


def _stub(layer, weight, down):
    calls = []
    layer.gate_up_proj = NS(forward_fused_silu_and_mul=lambda t: t)
    layer.down_proj = lambda t: (down.clone(), None)
    layer.expert_gate = _Gate(weight, calls)
    return calls


def _rows(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, 2560, generator=gen, device="cuda")).half().contiguous()


def _m1_per_row(down, x, weight):
    ops = _gate_ops()
    out = down.clone()
    for r in range(x.shape[0]):
        row = down[r:r + 1].clone()
        ops.qwen38_shared_gate_exact_out(row, x[r:r + 1].contiguous(), weight)
        out[r].copy_(row[0])
    return out


@pytest.mark.parametrize("prefix", ["model.layers.3.mlp.shared_expert",
                                    "mtp.layers.48.mlp.shared_expert"])
@torch.inference_mode()
def test_gate_rows_in_lane(monkeypatch, prefix):
    cfg = C.lane_config(4, draft=prefix.startswith("mtp"))
    qwen2_moe, layer, weight = _construct(monkeypatch, cfg, prefix)
    assert layer._sm70_exact_shared_expert_gate and layer._sx_shared_gate_rows
    for m in WIDTHS + (1, 33):
        x = _rows(m, 100 + m)
        down = _rows(m, 200 + m)
        calls = _stub(layer, weight, down)
        got = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
        torch.cuda.synchronize()
        if m <= 32:
            assert calls == [], m
            assert torch.equal(got.view(torch.int16),
                               _m1_per_row(down, x, weight).view(torch.int16)), m
        else:
            assert calls == ["unfused"], m
            ref = torch.sigmoid(F.linear(x, weight)) * down
            assert torch.equal(got, ref), m


@torch.inference_mode()
def test_gate_dev2_build(monkeypatch):
    qwen2_moe, layer, weight = _construct(
        monkeypatch, C.lane_config(4), "model.layers.3.mlp.shared_expert",
        lane_switch=False,
    )
    assert layer._sm70_exact_shared_expert_gate and not layer._sx_shared_gate_rows
    for m in (1, 5, 20):
        x, down = _rows(m, 300 + m), _rows(m, 400 + m)
        calls = _stub(layer, weight, down)
        qwen2_moe.Qwen2MoeMLP.forward(layer, x)
        assert calls == ([] if m == 1 else ["unfused"]), m


@pytest.mark.parametrize("m", [20, 30])
@torch.inference_mode()
def test_gate_graph_replay(monkeypatch, m):
    qwen2_moe, layer, weight = _construct(
        monkeypatch, C.lane_config(4), "model.layers.3.mlp.shared_expert"
    )
    x = torch.zeros(m, 2560, dtype=torch.float16, device="cuda")
    down = torch.zeros(m, 2560, dtype=torch.float16, device="cuda")
    _stub(layer, weight, down)
    qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    for trial in range(8):
        new_x, new_down = _rows(m, 600 + trial), _rows(m, 700 + trial)
        x.copy_(new_x)
        down.copy_(new_down)
        graph.replay()
        torch.cuda.synchronize()
        expected = _m1_per_row(new_down, new_x, weight)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), trial
    del graph


@pytest.mark.skipif(os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0"),
                    reason="SX_TEST_NO_BENCH")
@torch.inference_mode()
def test_gate_bench(monkeypatch):
    ops = _gate_ops()
    gen = torch.Generator(device="cuda").manual_seed(9)
    weights = [(torch.randn(1, 2560, generator=gen, device="cuda") * 0.02).half()
               for _ in range(LAYERS)]
    print("\n[b3-gate] 48 shared-gate calls in one CUDA graph, median of 60 (ms)")
    print(f"  {'M':>4}{'unfused':>9}{'rows':>9}{'saved':>9}")
    for m in WIDTHS:
        x = _rows(m, 1)
        outs = [_rows(m, 2 + i) for i in range(LAYERS)]

        def unfused():
            for i in range(LAYERS):
                outs[i].copy_(torch.sigmoid(F.linear(x, weights[i])) * outs[i])

        def fused():
            for i in range(LAYERS):
                ops.qwen38_shared_gate_exact_out(outs[i], x, weights[i])

        old = C.graph_ms(unfused)
        new = C.graph_ms(fused)
        print(f"  {m:>4}{old:>9.4f}{new:>9.4f}{old - new:>+9.4f}")
