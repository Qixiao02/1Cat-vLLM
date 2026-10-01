# SPDX-License-Identifier: Apache-2.0
"""Item 3 of "mtp-port-moe": lane routers' packed key (M<=16, upstream
8a99ccb4e) and top-16 key selection (M5/M10, upstream 1ef9f45a5).

GPU: 1 x V100 (SM70). Needs a Triton with ``tl.topk`` (upstream ran 3.6.0)
and this group's fused_topk_router.py in the imported vllm (rebuilt image or
bind mount).

  /opt/venv/bin/python -m pytest -q sx_tests/mtp-port-moe/test_router_key_gpu.py
  /opt/venv/bin/python -m pytest -q -s sx_tests/mtp-port-moe/test_router_key_gpu.py -k bench

Routers are FusedTopKRouter objects built under the MTP-lane configs of
sx_tests/b3-moe-verify/_mtp_common.py (target k = 4, draft k = 4) and the
no-MTP config; the "previous" router is the lane router with its key
arguments cleared (exactly the launch before this port).

Asserts:
  * construction: lane target and draft routers carry (16, True); the no-MTP
    router carries () and so launches as before;
  * all 63,488 finite FP16 logit payloads (both zeros, subnormals), shuffled
    into 124 rows and routed at every width 1..32: expert ids, FP32 weights
    and rank-major source rows of the
    lane router are bitwise equal to the previous router (packed key at
    M2..16, top-16 at M5/M10, the 64-bit key unchanged at M17..32);
  * NaN / +Inf / some -Inf / all -Inf / signed-zero / tie / huge rows
    bitwise equal as well;
  * CUDA-graph replay at W5 / W10 / W20 with changing logits.
Prints (-s, -k bench): 48 router calls in one CUDA graph at W5 / W10:
previous (64-bit sort), packed key, packed key + top-16.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.pop("VLLM_SM70_MTP_ROUTER_TOP16", None)  # lane default

B3 = Path(__file__).resolve().parents[1] / "b3-moe-verify"
sys.path.insert(0, str(B3))

import _mtp_common as C  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.ON_SM70, reason="requires an SM70 GPU")
E, K = 512, 10


def _mod():
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router

    return fused_topk_router


@pytest.fixture(scope="module")
def routers():
    import triton.language as tl

    if not hasattr(tl, "topk"):
        pytest.skip("this Triton has no tl.topk (upstream used 3.6.0)")
    import vllm.config.vllm as vcfg

    mod = _mod()
    built = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("SX_OPT_MTP_LANE", raising=False)
        for name in ("_SX_OPT_MTP_MOE_ROUTES", "_SX_OPT_ROUTER32",
                     "_SX_OPT_MTP_ROUTER_PACKED_KEY", "_SX_OPT_MTP_ROUTER_TOP16"):
            mp.setattr(mod, name, True)
        for label, cfg in (("lane", C.lane_config(4)),
                           ("draft", C.lane_config(4, draft=True)),
                           ("nomtp", C.nomtp_config())):
            mp.setattr(vcfg, "get_current_vllm_config_or_none", lambda c=cfg: c)
            built[label] = mod.FusedTopKRouter(
                top_k=K, global_num_experts=E, scoring_func="softmax",
                renormalize=True)
    return built


def _route(r, logits, key_args=None):
    mod = _mod()
    hidden = torch.empty(logits.shape[0], 2560, device="cuda", dtype=torch.float16)
    return mod.fused_topk(
        hidden, logits, K, True,
        sm70_qwen38_router_runtime_m=r._sm70_qwen38_router_runtime_m,
        sm70_qwen38_router_key_args=(
            r._sm70_qwen38_router_key_args if key_args is None else key_args),
    )


def _assert_bitwise(got, ref, label):
    for name, a, b in zip(("weights", "ids", "rows"), got, ref):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (label, name)


def test_construction(routers):
    assert routers["lane"]._sm70_qwen38_router_key_args == (16, True)
    assert routers["draft"]._sm70_qwen38_router_key_args == (16, True)
    assert routers["nomtp"]._sm70_qwen38_router_key_args == ()
    assert routers["lane"]._sm70_qwen38_router_runtime_m


@pytest.mark.parametrize("label", ["lane", "draft"])
@pytest.mark.parametrize("width", range(1, 33))
def test_all_fp16_payloads(routers, label, width):
    r = routers[label]
    payloads = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32)
    halfs = payloads.to(torch.int16).view(torch.float16)
    # The 63,488 finite payloads (both zeros, subnormals) fill 124 rows
    # exactly; NaN / +-Inf rows take the degenerate path, covered below.
    rows = halfs[torch.isfinite(halfs)].view(-1, E)
    # Shuffle columns per row so equal payloads do not sit at fixed experts.
    gen = torch.Generator(device="cuda").manual_seed(width)
    perm = torch.argsort(torch.rand(rows.shape, generator=gen, device="cuda"), -1)
    rows = torch.gather(rows, 1, perm)
    for start in range(0, rows.shape[0], width):
        chunk = rows[start:start + width]
        if chunk.shape[0] < width:
            chunk = torch.cat([chunk, rows[: width - chunk.shape[0]]])
        chunk = chunk.contiguous()
        got = _route(r, chunk)
        ref = _route(r, chunk, key_args=())
        torch.accelerator.synchronize()
        _assert_bitwise(got, ref, (label, width, start))


@pytest.mark.parametrize("width", [1, 2, 5, 10, 16, 20, 32])
def test_degenerate_rows(routers, width):
    r = routers["lane"]
    base = torch.randn(width, E, device="cuda").half()
    cases = {
        "zeros_signed": base.zero_().clone().index_fill_(
            1, torch.arange(0, E, 2, device="cuda"), -0.0),
        "ties": (torch.arange(E, device="cuda") % 7).half().expand(width, E),
        "nan": torch.randn(width, E, device="cuda").half().index_fill_(
            1, torch.tensor([0], device="cuda"), float("nan")),
        "pinf": torch.randn(width, E, device="cuda").half().index_fill_(
            1, torch.tensor([3], device="cuda"), float("inf")),
        "some_ninf": torch.randn(width, E, device="cuda").half().index_fill_(
            1, torch.arange(1, E, 3, device="cuda"), -float("inf")),
        "all_ninf": torch.full((width, E), -float("inf"), device="cuda",
                               dtype=torch.float16),
        "huge": torch.randn(width, E, device="cuda").half() * 30000,
    }
    for name, logits in cases.items():
        logits = logits.contiguous()
        _assert_bitwise(_route(r, logits), _route(r, logits, key_args=()),
                        (width, name))


@pytest.mark.parametrize("width", [5, 10, 20])
def test_graph_replay(routers, width):
    r = routers["lane"]
    logits = torch.randn(width, E, device="cuda").half()
    ref_out = None
    for _ in range(2):
        _route(r, logits)
        _route(r, logits, key_args=())
    torch.accelerator.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    with torch.cuda.graph(graphs[0]):
        out = _route(r, logits)
    with torch.cuda.graph(graphs[1]):
        ref_out = _route(r, logits, key_args=())
    for scale in (0.001, 0.1, 1.0, 30.0, 3000.0):
        logits.normal_(0, scale)
        for t in (*out, *ref_out):
            t.fill_(-777)
        for g in graphs:
            g.replay()
        torch.accelerator.synchronize()
        _assert_bitwise(out, ref_out, (width, scale))


def test_bench(routers):
    r = routers["lane"]
    print("\n[mtp-port-moe] 48 router calls in one CUDA graph, median of 60 (ms)")
    for width in (5, 10):
        logits = torch.randn(width, E, device="cuda").half()
        row = {}
        for name, args in (("64-bit sort", ()), ("packed", (16, False)),
                           ("packed+top16", (16, True))):

            def fn(args=args):
                for _ in range(48):
                    _route(r, logits, key_args=args)

            row[name] = C.graph_ms(fn)
        print(f"  W{width}: " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
