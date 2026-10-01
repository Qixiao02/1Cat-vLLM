# SPDX-License-Identifier: Apache-2.0
"""Item 1 of "mtp-port-moe": grouped W5 MTP verify experts (upstream 45248dc8d).

GPUs: 1 x V100 (SM70), ~4 GB free (the b3-moe-verify synthetic TP4-local
Qwen3.8 MoE layers: E512, hidden 2560, local intermediate 160, top-k 10).
Needs a vllm._C rebuilt from this branch (nvfp4_grouped_w2_batch_reduce_sm70_out
next to the legacy grouped and QPN-MTP5 ops); the module skips otherwise.

  /opt/venv/bin/python -m pytest -q sx_tests/mtp-port-moe/test_grouped_mtp5_gpu.py
  /opt/venv/bin/python -m pytest -q -s sx_tests/mtp-port-moe/test_grouped_mtp5_gpu.py -k bench

The layers are built by ModelOptNvFp4SM70MoEMethod.process_weights_after_loading
under the MTP-lane (k = 4) and no-MTP configs of sx_tests/b3-moe-verify
(builders, contexts and route generators are imported from there).

Asserts:
  * load time: the k = 4 lane layer is stamped (sm70_nvfp4_grouped_mtp5) and
    its route table says W5 = grouped-mtp5-split4+batch-reduce; the no-MTP
    layer is not stamped;
  * W5 uniform verify, five route shapes x five activation scales: output
    bitwise equal to 1Cat's QPN-MTP5 route (split-4 W13, separate W2,
    Triton weighted reduce: upstream's reference) and to the previous lane
    route (QPN-MTP5 W13 + fused batch W2 reduce); invalid expert ids
    (-1 / 512 / 99999) bitwise equal to the QPN-MTP5 route;
  * a FULL CUDA graph of apply() at W5 replays with new x / ids / weights and
    NaN-poisoned scratch bitwise equal to the eager QPN-MTP5 route;
  * W10 and wider, and W5 in mixed / non-uniform / decode contexts, are
    bitwise unchanged by the stamp (the ported route is never taken).
Prints (-s, -k bench): 48 x apply() in one CUDA graph at W5: grouped MTP5 vs
the previous lane route vs 1Cat's QPN-MTP5 route.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.pop("VLLM_SM70_NVFP4_MOE_GROUPED_MTP5", None)  # lane default
os.environ.pop("SX_OPT_MTP_MOE_GROUPED_MTP5", None)

B3 = Path(__file__).resolve().parents[1] / "b3-moe-verify"
sys.path.insert(0, str(B3))

import _mtp_common as C  # noqa: E402
import pytest  # noqa: E402
import test_mtp_verify_moe_apply as base  # noqa: E402
import torch  # noqa: E402
from test_mtp_verify_moe_apply import layers  # noqa: E402,F401  (fixture)

pytestmark = pytest.mark.skipif(not C.ON_SM70, reason="requires an SM70 GPU")

KINDS = ("random", "pool113", "shared10", "reversed", "same")
SCALES = (0.0, 0.001, 0.1, 1.0, 3.0)


@pytest.fixture(scope="module", autouse=True)
def _native_op():
    from vllm import _sm70_ops as sm70_ops

    if not sm70_ops.has_nvfp4_grouped_batch_reduce_dispatch():
        pytest.skip("vllm._C lacks nvfp4_grouped_w2_batch_reduce_sm70_out; "
                    "rebuild the native extension from this branch")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _three_routes(layers, x, w, ids):
    lane = layers.lane
    got = base._apply_verify(layers.lane_method, lane, x, w, ids, 5)
    with C.layer_attrs(lane, sm70_nvfp4_grouped_mtp5=False):
        batch_w2 = base._apply_verify(layers.lane_method, lane, x, w, ids, 5)
        with C.layer_attrs(lane, sx_mtp_qpn_dynamic=False):
            mtp5_only = base._apply_verify(layers.lane_method, lane, x, w, ids, 5)
    torch.accelerator.synchronize()
    return got, batch_w2, mtp5_only


def test_load_time(layers):
    from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

    assert layers.lane.sm70_nvfp4_grouped_mtp5
    assert not layers.nomtp.sm70_nvfp4_grouped_mtp5
    table = moe._mtp_verify_route_table(layers.lane)
    assert table[5] == "grouped-mtp5-split4+batch-reduce"
    assert table[10] == "grouped-split4"


@pytest.mark.parametrize("kind", KINDS)
def test_w5_bitwise_vs_qpn_mtp5(layers, kind):
    x0, w, ids = base._routes(kind, 5, seed=4100 + len(kind))
    for scale in SCALES:
        x = (x0.float() * (scale / 0.1)).half() if scale else torch.zeros_like(x0)
        got, batch_w2, mtp5_only = _three_routes(layers, x, w, ids)
        assert torch.equal(_bits(got), _bits(mtp5_only)), (kind, scale)
        assert torch.equal(_bits(got), _bits(batch_w2)), (kind, scale)


def test_w5_invalid_expert_ids(layers):
    x, w, ids = base._routes("invalid", 5, seed=4200)
    got, batch_w2, mtp5_only = _three_routes(layers, x, w, ids)
    assert torch.equal(_bits(got), _bits(mtp5_only))
    print(f"\n[mtp-port-moe] W5 invalid ids: grouped vs previous lane route "
          f"bitwise {torch.equal(_bits(got), _bits(batch_w2))}")


def test_w5_full_graph_replay(layers):
    from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

    lane = layers.lane
    x0, w0, ids0 = base._routes("pool113", 5, seed=4300)
    sx, sw, sids = x0.clone(), w0.clone(), ids0.clone()
    buf = C.qsl_buffer()
    C.write_live(buf, 5, 5)
    meta = C.verify_metadata(buf[:2], 5, 5)
    with C.forward_context(moe, meta):
        layers.lane_method.apply(lane, sx, sw, sids, None, None)  # warm
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with C.forward_context(moe, meta), torch.cuda.graph(graph):
        out = layers.lane_method.apply(lane, sx, sw, sids, None, None)
    for trial, kind in enumerate(KINDS):
        x, w, ids = base._routes(kind, 5, seed=4400 + trial)
        sx.copy_(x)
        sw.copy_(w)
        sids.copy_(ids)
        base._poison(lane)
        graph.replay()
        torch.accelerator.synchronize()
        got = out.clone()
        _, _, mtp5_only = _three_routes(layers, x, w, ids)
        assert torch.equal(_bits(got), _bits(mtp5_only)), kind
    del graph


@pytest.mark.parametrize("width", [10, 15, 20, 30])
def test_wider_verify_unchanged(layers, width):
    lane = layers.lane
    x, w, ids = base._routes("pool113", width, seed=4500 + width)
    got = base._apply_verify(layers.lane_method, lane, x, w, ids, 5)
    with C.layer_attrs(lane, sm70_nvfp4_grouped_mtp5=False):
        previous = base._apply_verify(layers.lane_method, lane, x, w, ids, 5)
    torch.accelerator.synchronize()
    assert torch.equal(_bits(got), _bits(previous))


def test_w5_other_contexts_unchanged(layers):
    lane = layers.lane
    x, w, ids = base._routes("pool113", 5, seed=4600)
    buf = C.qsl_buffer()
    C.write_live(buf, 5, 5)
    view = buf[:2]
    cases = {
        "mixed": C.mixed_metadata(buf[:3], 5),
        "non_uniform": C.verify_metadata(view, 5, 5, spec_tokens=4),
        "plain_decode_mix": C.verify_metadata(view, 5, 5, plain_decodes=1),
    }
    for name, meta in cases.items():
        got = base._apply_verify(layers.lane_method, lane, x, w, ids, 5,
                                 metadata=meta)
        with C.layer_attrs(lane, sm70_nvfp4_grouped_mtp5=False):
            previous = base._apply_verify(layers.lane_method, lane, x, w, ids, 5,
                                          metadata=meta)
        torch.accelerator.synchronize()
        assert torch.equal(_bits(got), _bits(previous)), name
    got = base._apply_decode(layers.lane_method, lane, x, w, ids)
    with C.layer_attrs(lane, sm70_nvfp4_grouped_mtp5=False):
        previous = base._apply_decode(layers.lane_method, lane, x, w, ids)
    torch.accelerator.synchronize()
    assert torch.equal(_bits(got), _bits(previous)), "plain decode"


def test_bench(layers):
    direct = dict(sm70_nvfp4_grouped_mtp5=False)
    rows = {
        "grouped-mtp5 (new)": base._graph_round(layers, 5, 5, 5),
        "qpn-mtp5+batch-w2 (previous lane)": base._graph_round(
            layers, 5, 5, 5, **direct),
        "qpn-mtp5+sep-w2 (1Cat)": base._graph_round(
            layers, 5, 5, 5, **direct, sx_mtp_qpn_dynamic=False),
    }
    print("\n[mtp-port-moe] W5, 48 x apply() in one CUDA graph, median of 60 "
          "(ms), pool113 routes:")
    for name, ms in rows.items():
        print(f"  {name:<36}{ms:>8.3f}")
