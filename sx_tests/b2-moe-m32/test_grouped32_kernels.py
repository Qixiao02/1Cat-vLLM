# SPDX-License-Identifier: Apache-2.0
"""Op-level tests for the v2 grouped NVFP4 decode ops (design_1 [C3]/[C5]).

Hardware: 1 x V100 (SM70), ~1 GB free. No test needs 4 GPUs.
Needs the image's vllm._C (old grouped ops, QPN W2 reduce) plus the v2 ops:
a rebuilt _C, $SX_OPT_MOE_GROUPED32_LIBRARY, or an on-the-fly sidecar build
(nvcc; see _grouped32_common.ensure_v2_ops / build_sidecar.py).

  /opt/venv/bin/python -m pytest -q sx_tests/b2-moe-m32/test_grouped32_kernels.py
  /opt/venv/bin/python -m pytest -q -s ... -k microbench     # timing table

Asserts (all comparisons torch.equal / rtol=atol=0 unless stated):
  * capacity: max_routes()==320; old op still rejects 17 tokens, v2 rejects 33.
  * v2 (valid=None) == old op at every width 1..16, splits 4/8, interleaved and
    not: W13 intermediate and W2+reduce output, for 7 route patterns
    (random, clustered, pool113, shared10, reversed, invalid ids, all-same).
  * row independence: v2 at M in {17,20,24,32} (split 8) equals, row by row,
    the old op at M16 applied to rows [0,16) and [M-16,M), the v2 op at M=1 on
    each row, and the production QPN W2+reduce (tokens<=16 windows) fed the
    same intermediate.
  * planner: rows/experts/sizes/total exactly equal a CPU model (groups by
    expert id, invalid last, ascending route order in each expert) and are
    identical across repeated runs (deterministic), with and without masking.
  * masking: live rows equal the unmasked call, padded output rows are +0,
    padded routes are not planned (total == groups of the live routes), NaN
    padded activations and NaN-poisoned scratch cannot reach live rows;
    valid<=0 or >=M means no masking (identical to valid=None).
  * CUDA graph: capture at M24/M32 with the valid pointer as the tail of a
    query_start_loc-style buffer; replay with changed ids/x/weights/live
    counts and poisoned outputs/metadata == eager results.
Prints (microbench): 48 x (W13+W2) per graph at M24/M32 split 8 vs 4, masked
M24 live 17/20, and old vs v2 at M8/M16 (planner overhead).
"""

from __future__ import annotations

import pytest
import torch

import _grouped32_common as common
from _grouped32_common import (
    H,
    K,
    KINDS,
    LAYERS,
    Workspace,
    check_plan,
    expected_plan,
    graph_ms,
    routes,
    run_legacy,
    run_v2,
)

pytestmark = pytest.mark.skipif(not common.ON_SM70, reason="requires an SM70 GPU")


@pytest.fixture(scope="module")
def weights():
    common.ensure_v2_ops()
    if common.legacy_ops() is None:
        pytest.skip("image vllm._C lacks the original grouped decode ops")
    return common.synthetic_weights()


def _qpn_w2_reduce():
    from vllm import _sm70_ops as ops

    if not ops.has_nvfp4_qpn_w2_reduce_dispatch():
        return None
    return ops.nvfp4_moe_qpn_w2_reduce_sm70_out


# ---------------------------------------------------------------------------
def test_capacity_and_limits(weights):
    w13, w2, max_routes = common.v2_ops()
    assert int(max_routes()) == 320
    x, topk, ids = routes("random", 17, seed=1)
    ws = Workspace(17)
    with pytest.raises(RuntimeError):
        run_legacy(ws, weights, x, topk, ids)  # original op keeps <=16
    x33, topk33, ids33 = routes("random", 33, seed=2)
    ws33 = Workspace(33, capacity_routes=330)
    with pytest.raises(RuntimeError):
        run_v2(ws33, weights, x33, topk33, ids33)
    with pytest.raises(RuntimeError):  # valid_tokens must live on the device
        run_v2(ws, weights, x, topk, ids, valid=torch.tensor([3], dtype=torch.int32))
    with pytest.raises(RuntimeError):
        run_v2(ws, weights, x, topk, ids, split=3)


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("split", [4, 8])
@pytest.mark.parametrize("kind", KINDS)
def test_v2_equals_legacy_up_to_16(weights, kind, split, interleaved):
    for m in range(1, 17):
        x, topk, ids = routes(kind, m, seed=100 * m + split)
        old, new = Workspace(m), Workspace(m)
        run_legacy(old, weights, x, topk, ids, split, interleaved)
        new.poison()
        run_v2(new, weights, x, topk, ids, split, interleaved)
        torch.cuda.synchronize()
        assert torch.isfinite(old.out).all(), (kind, m)
        assert torch.equal(new.inter, old.inter), (kind, m, split)
        assert torch.equal(new.out, old.out), (kind, m, split)
        # Same number of groups (only numbering/slot order may differ).
        assert int(new.total.item()) == int(old.total.item()), (kind, m)


def _windows(m: int) -> list[tuple[int, int]]:
    return [(0, 16), (m - 16, m)] if m > 16 else [(0, m)]


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", [17, 20, 24, 32])
def test_rows_equal_m16_and_m1(weights, m, kind, interleaved):
    x, topk, ids = routes(kind, m, seed=7 * m + len(kind))
    big = Workspace(m)
    big.poison()
    run_v2(big, weights, x, topk, ids, 8, interleaved)
    torch.cuda.synchronize()
    assert torch.isfinite(big.out).all()
    inter_rows = big.inter.view(m, K, 160)
    # (a) the admitted M16 grouped path (original op, split 8) on the same rows
    for lo, hi in _windows(m):
        ref = Workspace(hi - lo)
        run_legacy(ref, weights, x[lo:hi].contiguous(), topk[lo:hi].contiguous(),
                   ids[lo:hi].contiguous(), 8, interleaved)
        torch.cuda.synchronize()
        assert torch.equal(inter_rows[lo:hi], ref.inter.view(hi - lo, K, 160)), (lo, hi)
        assert torch.equal(big.out[lo:hi], ref.out), (lo, hi)
    # (b) the same kernels at M=1 (split 8), row by row
    for r in range(m):
        one = Workspace(1)
        run_v2(one, weights, x[r : r + 1].contiguous(), topk[r : r + 1].contiguous(),
               ids[r : r + 1].contiguous(), 8, interleaved)
        torch.cuda.synchronize()
        assert torch.equal(inter_rows[r], one.inter), r
        assert torch.equal(big.out[r], one.out[0]), r
    # (c) independent W2+reduce reference (production direct QPN kernel; it is
    # not specified for out-of-range ids, so the "invalid" pattern is skipped)
    reduce_ref = _qpn_w2_reduce()
    if reduce_ref is not None and kind != "invalid":
        _, _, w2w, s2 = weights
        for lo, hi in _windows(m):
            expected = torch.empty(hi - lo, H, device="cuda", dtype=torch.float16)
            reduce_ref(expected, big.inter[lo * K : hi * K].contiguous(), w2w, s2,
                       ids[lo:hi].contiguous().view(-1), topk[lo:hi].contiguous())
            torch.cuda.synchronize()
            assert torch.equal(big.out[lo:hi], expected), (lo, hi)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("m", [1, 8, 16, 17, 24, 32])
def test_planner_metadata_exact_and_deterministic(weights, m, kind):
    x, topk, ids = routes(kind, m, seed=31 * m + 5)
    live_values = [None, max(1, m // 2), m - 1] if m > 1 else [None]
    for live in live_values:
        valid = None
        if live is not None:
            valid = torch.tensor([live], device="cuda", dtype=torch.int32)
        snapshots = []
        for _ in range(2):
            ws = Workspace(m)
            ws.poison()
            run_v2(ws, weights, x, topk, ids, 8, True, valid)
            torch.cuda.synchronize()
            limit = m * K if live is None else live * K
            groups = check_plan(ids.view(-1).cpu(), limit, ws.rows, ws.experts,
                                ws.sizes, ws.total)
            snapshots.append(
                (ws.rows.view(-1, 8)[:groups].clone(),
                 ws.experts[:groups].clone(), ws.sizes[:groups].clone())
            )
        a, b = snapshots
        for ta, tb in zip(a, b):
            # rows beyond a group's size are unspecified; compare with sizes
            if ta.dim() == 2:
                for g in range(ta.shape[0]):
                    n = int(a[2][g])
                    assert torch.equal(ta[g, :n], tb[g, :n])
            else:
                assert torch.equal(ta, tb)


@pytest.mark.parametrize("kind", ["random", "pool113", "invalid", "same"])
@pytest.mark.parametrize(
    "m,lives",
    [
        (8, [1, 5, 7]),
        (16, [1, 9, 15]),
        (24, [1, 17, 20, 23]),
        (32, [1, 17, 25, 31]),
    ],
)
def test_padded_row_masking(weights, m, lives, kind):
    x, topk, ids = routes(kind, m, seed=900 + m)
    ref = Workspace(m)
    run_v2(ref, weights, x, topk, ids)
    torch.cuda.synchronize()
    ref_inter = ref.inter.view(m, K, 160).clone()
    for live in lives:
        valid = torch.tensor([live], device="cuda", dtype=torch.int32)
        xp = x.clone()
        xp[live:] = float("nan")  # stale padded activations must not matter
        ws = Workspace(m)
        ws.poison()
        run_v2(ws, weights, xp, topk, ids, valid=valid)
        torch.cuda.synchronize()
        assert torch.equal(ws.out[:live], ref.out[:live]), (m, live)
        assert torch.equal(ws.inter.view(m, K, 160)[:live], ref_inter[:live])
        pad = ws.out[live:]
        assert torch.equal(pad, torch.zeros_like(pad)), (m, live)
        assert not torch.signbit(pad).any()  # +0, not -0
        groups = len(expected_plan(ids.view(-1).tolist(), live * K))
        assert int(ws.total.item()) == groups
        assert groups <= int(ref.total.item())
    # Out-of-range counts disable masking (fail-open to the full computation).
    for bogus in (0, -3, m, m + 1, 1000):
        valid = torch.tensor([bogus], device="cuda", dtype=torch.int32)
        ws = Workspace(m)
        ws.poison()
        run_v2(ws, weights, x, topk, ids, valid=valid)
        torch.cuda.synchronize()
        assert torch.equal(ws.out, ref.out), bogus
        assert int(ws.total.item()) == int(ref.total.item()), bogus


@pytest.mark.parametrize("split", [8, 4])
@pytest.mark.parametrize("m", [24, 32])
def test_graph_capture_replay(weights, m, split):
    """Static buffers + tail-of-query_start_loc live pointer, as in FULL graphs."""
    x, topk, ids = routes("random", m, seed=4000 + m)
    sx, stopk, sids = x.clone(), topk.clone(), ids.clone()
    qsl = torch.arange(33 + 1, device="cuda", dtype=torch.int32)  # max_num_reqs 33
    tail = qsl[: m + 1][m:]  # runner view [:num_reqs_padded + 1], last element
    ws = Workspace(m)
    run_v2(ws, weights, sx, stopk, sids, split, True, tail)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_v2(ws, weights, sx, stopk, sids, split, True, tail)
    trials = [("pool113", m), ("random", 17), ("invalid", m - 1), ("shared10", 20),
              ("same", 1), ("clustered", m), ("pool113", 18)]
    for trial, (kind, live) in enumerate(trials):
        x, topk, ids = routes(kind, m, seed=4100 + 13 * trial + m)
        sx.copy_(x)
        stopk.copy_(topk)
        sids.copy_(ids)
        # runner semantics: cumsum for live requests, tail = live token count
        host = torch.arange(34, dtype=torch.int32).clamp_(max=live)
        qsl.copy_(host)
        ws.poison()
        graph.replay()
        torch.cuda.synchronize()
        got_out, got_total = ws.out.clone(), int(ws.total.item())
        check_plan(ids.view(-1).cpu(), live * K, ws.rows, ws.experts, ws.sizes,
                   ws.total)
        ref = Workspace(m)
        run_v2(ref, weights, x, topk, ids, split, True)
        torch.cuda.synchronize()
        assert torch.equal(got_out[:live], ref.out[:live]), (trial, kind, live)
        assert torch.equal(got_out[live:], torch.zeros_like(got_out[live:]))
        assert got_total == len(expected_plan(ids.view(-1).tolist(), live * K))
    del graph


# ---------------------------------------------------------------------------
# microbenchmarks (prints only)
def _round(weights, m, split, kind="pool113", live=None, legacy=False):
    x, topk, ids = routes(kind, m, seed=12000 + m)
    ws = Workspace(m)
    valid = None
    if live is not None:
        valid = torch.tensor([live], device="cuda", dtype=torch.int32)

    def fn():
        for _ in range(LAYERS):
            if legacy:
                run_legacy(ws, weights, x, topk, ids, split, True)
            else:
                run_v2(ws, weights, x, topk, ids, split, True, valid)

    return graph_ms(fn)


def test_microbench(weights):
    print("\n[grouped32] 48 x (W13+W2+reduce) in one CUDA graph, median ms "
          "(pool113 routes: ~C24 unique-expert density)")
    for m in (24, 32):
        s8 = _round(weights, m, 8)
        s4 = _round(weights, m, 4)
        print(f"  M{m}: v2 split8 {s8:.3f}  split4 {s4:.3f}  "
              f"({(s8 - s4) / s8 * 100:+.1f}% split4 saving)")
    full = _round(weights, 24, 8)
    for live in (17, 20, 23):
        masked = _round(weights, 24, 8, live=live)
        print(f"  M24 live {live}: masked {masked:.3f} vs unmasked {full:.3f} "
              f"({full - masked:+.3f} ms/48 layers)")
    for m, split in ((8, 4), (16, 8)):
        old = _round(weights, m, split, legacy=True)
        new = _round(weights, m, split)
        print(f"  M{m} split{split}: old planner {old:.3f}  v2 planner {new:.3f} "
              f"({new - old:+.3f} ms/48 layers)")
    for kind in ("random", "shared10"):
        print(f"  M32 {kind}: v2 split8 {_round(weights, 32, 8, kind=kind):.3f}")
