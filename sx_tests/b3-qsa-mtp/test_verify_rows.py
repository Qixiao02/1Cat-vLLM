# SPDX-License-Identifier: Apache-2.0
"""design_1 [MTP-9]: QSA decode-row kernels for MTP verify widths up to 63.

new = live ops with the MTP-lane options (SX_OPT_QSA_MTP_DECODE_ROWS, every
cap 63): two-warp split-K partial, resolved physical rows and the decode
top-k rows kernel admitted for 33..63 rows; dev2 = byte-identical copy of the
sx-1.8.0-dev2 module (four-warp partial, page-table loads, generic top-k
above 32 rows).

Verify batches: B in {1,2,4,8,12,16,24} requests x q = k + 1 in {2,3,4,5}
causal rows per request (rows = B * q, 2..120; >= 64 rows is the unchanged
XQA page4 route), contexts 2K/8K, TP4-local QSA geometry (6 x 256 query
heads, 1 KV head, 2051-token selection, 784-token pages; indexer 4 x 128,
196-row compressed pages). Asserts:
  * indexer selections (score -> top-k -> expand) new(MTP lane) == dev2,
    torch.equal on the int32 token ids, with the rows top-k op engaged for
    2..63 rows and the generic kernel above;
  * sparse attention output new(MTP lane) == dev2 bit for bit (FP16 as
    int16), with/without the output gate, with engagement checks (two warps
    and RESOLVED_INDICES for rows <= 63, four warps / page-table loads above
    32 rows in dev2);
  * the no-MTP lane (sx_mtp_lane=None) == dev2 bit for bit with dev2's exact
    launch configuration (the production lane is untouched);
  * causality: every selected token of a verify row is <= its own position
    (not the newest verify position of the request) and the attention output
    matches an FP32 torch reference over those tokens;
  * CUDA-graph capture of the MTP-lane path, replay with changed metadata
    and poisoned output == dev2 eager.
Microbenchmark (CUDA graphs, 12 QSA layers): attention and indexer at verify
widths 33..63, dev2 vs MTP lane.

GPU: ONE V100 (SM70).
  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-qsa-mtp/test_verify_rows.py
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.is_sm70(), reason="requires SM70 GPU")


def _grid_ids():
    return [f"B{b}xq{q}" for b, q in C.verify_grid()]


def _select_with_spies(ops, case, **kwargs):
    """Run the indexer selection and report which top-k launcher ran."""
    rows_spy = None
    rows_op = ops._sm70_qsa_lexicographic_topk_rows_op()
    generic_spy = C.CallSpy(ops._sm70_qsa_lexicographic_topk_op())
    patches = {"_sm70_qsa_lexicographic_topk_op": lambda: generic_spy}
    if rows_op is not None:
        rows_spy = C.CallSpy(rows_op)
        patches["_sm70_qsa_lexicographic_topk_rows_op"] = lambda: rows_spy
    with C.patched(ops, **patches):
        out = C.run_select(ops, case, **kwargs)
    return out, (rows_spy.calls if rows_spy else 0), generic_spy.calls


@pytest.mark.parametrize("context", C.CONTEXTS)
@pytest.mark.parametrize("num_reqs,q_len", C.verify_grid(), ids=_grid_ids())
def test_indexer_verify_rows_bitwise(num_reqs, q_len, context):
    new, dev2 = C.new_ops(), C.dev2_ops()
    index_case, _ = C.verify_case(num_reqs, q_len, context, seed=1)
    rows = num_reqs * q_len
    expected = C.run_select(dev2, index_case)
    actual, rows_calls, generic_calls = _select_with_spies(
        new, index_case, sx_mtp_lane=C.mtp_lane()
    )
    baseline, base_rows_calls, base_generic_calls = _select_with_spies(new, index_case)
    torch.cuda.synchronize()
    assert torch.equal(expected, actual), (rows, C.first_mismatch(expected, actual))
    assert torch.equal(expected, baseline), (rows, "no-MTP lane changed")
    has_rows_op = new._sm70_qsa_lexicographic_topk_rows_op() is not None
    if not has_rows_op:
        pytest.skip("rebuilt _C_stable_libtorch (decode rows top-k) not mounted")
    # Engagement: the MTP lane takes the rows kernel for 2..63 rows.
    expect_rows = 2 <= rows <= 63
    assert rows_calls == int(expect_rows), (rows, rows_calls, generic_calls)
    assert generic_calls == int(not expect_rows and rows > 1) + int(rows == 1), (
        rows,
        rows_calls,
        generic_calls,
    )
    # And the no-MTP lane keeps the dev2 32-row limit.
    assert base_rows_calls == int(2 <= rows <= 32), (rows, base_rows_calls)


def _page4_available() -> bool:
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        return False
    return hasattr(flash_attn_v100_cuda, "decode_paged_xqa_fwd")


def _attention_with_spies(ops, case, out, **kwargs):
    resolver = C.LaunchSpy(ops._qsa_resolve_physical_indices_kernel)
    partial = C.LaunchSpy(ops._qsa_sparse_paged_gqa_splitk_kernel)
    with C.patched(
        ops,
        _qsa_resolve_physical_indices_kernel=resolver,
        _qsa_sparse_paged_gqa_splitk_kernel=partial,
    ):
        C.run_attention(ops, case, out, **kwargs)
    return resolver, partial


@pytest.mark.parametrize("context", C.CONTEXTS)
@pytest.mark.parametrize("num_reqs,q_len", C.verify_grid(), ids=_grid_ids())
def test_attention_verify_rows_bitwise(num_reqs, q_len, context):
    new, dev2 = C.new_ops(), C.dev2_ops()
    _, case = C.verify_case(num_reqs, q_len, context, seed=2)
    rows = num_reqs * q_len
    for gated in (True, False):
        expected = torch.full_like(case.q, float("nan"))
        actual = torch.full_like(case.q, float("nan"))
        baseline = torch.full_like(case.q, float("nan"))
        dev2_resolver, dev2_partial = _attention_with_spies(
            dev2, case, expected, gated=gated
        )
        resolver, partial = _attention_with_spies(
            new, case, actual, gated=gated, sx_mtp_lane=C.mtp_lane()
        )
        base_resolver, base_partial = _attention_with_spies(
            new, case, baseline, gated=gated
        )
        torch.cuda.synchronize()
        assert C.bitwise_equal(expected, actual), (
            rows,
            gated,
            C.first_mismatch(expected, actual),
        )
        assert C.bitwise_equal(expected, baseline), (rows, gated, "no-MTP lane")
        if rows >= 64:
            # XQA page4 route: no split-K launch in any variant (without the
            # Flash-V100 XQA op every variant falls back to the same Triton
            # four-warp launch; the bitwise asserts above still apply).
            if _page4_available():
                assert not partial.launches and not dev2_partial.launches
            else:
                assert partial.launches[0]["num_warps"] == 4
            continue
        # Engagement of the MTP lane: resolved rows and two warps up to 63.
        assert len(resolver.launches) == 1, rows
        assert len(partial.launches) == 1
        assert partial.launches[0]["RESOLVED_INDICES"] is True
        assert partial.launches[0]["num_warps"] == 2, (rows, partial.launches[0])
        # dev2 and the no-MTP lane: identical launches (<= 32 rows decode
        # kernels, four warps / page-table loads above).
        for spy_resolver, spy_partial in (
            (dev2_resolver, dev2_partial),
            (base_resolver, base_partial),
        ):
            assert len(spy_resolver.launches) == int(rows <= 32), rows
            assert spy_partial.launches[0]["num_warps"] == (2 if rows <= 32 else 4)
            assert spy_partial.launches[0]["RESOLVED_INDICES"] == (rows <= 32)
        # Same split profile (the two-warp claim needs identical constexprs).
        for name in ("NUM_SPLITS", "BLOCK_N", "NUM_TILES", "BLOCK_M"):
            assert partial.launches[0][name] == dev2_partial.launches[0][name], name


@pytest.mark.parametrize("rows", (33, 40, 47, 48, 49, 55, 63))
def test_attention_odd_widths_bitwise(rows):
    """Every M in the new band (incl. the 16-divisible Triton specialisation
    at 48) with non-uniform request sizes (mixed verify lengths)."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    generator = torch.Generator().manual_seed(rows)
    request_rows = []
    while sum(request_rows) < rows:
        request_rows.append(min(int(torch.randint(1, 6, (1,), generator=generator)),
                                rows - sum(request_rows)))
    seq_lens = C.jittered_contexts(generator, len(request_rows), 8192)
    layout = C.verify_layout(request_rows, seq_lens)
    index_case = C.build_indexer_case(layout, seed=rows)
    indices = C.run_select(dev2, index_case)
    assert torch.equal(indices, C.run_select(new, index_case, sx_mtp_lane=C.mtp_lane()))
    case = C.build_attention_case(layout, indices, seed=rows)
    expected = torch.full_like(case.q, float("nan"))
    actual = torch.full_like(case.q, float("nan"))
    C.run_attention(dev2, case, expected)
    C.run_attention(new, case, actual, sx_mtp_lane=C.mtp_lane())
    torch.cuda.synchronize()
    assert C.bitwise_equal(expected, actual), C.first_mismatch(expected, actual)


def test_attention_padded_and_invalid_rows_bitwise():
    """Graph padding (position -1 rows) and invalid pages/requests keep the
    dev2 masking in the widened decode kernels."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    generator = torch.Generator().manual_seed(3)
    seq_lens = C.jittered_contexts(generator, 9, 8192)
    layout = C.verify_layout([5] * 9, seq_lens, padded_rows=5)  # 45 + 5 rows
    index_case = C.build_indexer_case(layout, seed=3)
    indices = C.run_select(dev2, index_case)
    assert torch.equal(indices, C.run_select(new, index_case, sx_mtp_lane=C.mtp_lane()))
    case = C.build_attention_case(layout, indices, seed=3)
    case.indices[0, ::7] = -1
    case.indices[6, 64:192] = case.indices[6, 0:128].clone()  # duplicates
    case.indices[11, ::11] = C.TABLE_WIDTH * C.SCHED_BLOCK + 5  # beyond table
    case.table[3, 0] = -1  # missing page
    case.token_to_req[20] = -1  # invalid request row
    case.token_to_req[21] = 1000  # out-of-range request row
    expected = torch.full_like(case.q, float("nan"))
    actual = torch.full_like(case.q, float("nan"))
    C.run_attention(dev2, case, expected)
    C.run_attention(new, case, actual, sx_mtp_lane=C.mtp_lane())
    torch.cuda.synchronize()
    assert C.bitwise_equal(expected, actual), C.first_mismatch(expected, actual)


@pytest.mark.parametrize("num_reqs,q_len", [(1, 5), (8, 5), (12, 5), (16, 3)])
def test_verify_rows_are_causal(num_reqs, q_len):
    """Each verify row attends only to tokens at or before its own position,
    and the widened kernels match an FP32 reference over that selection."""
    new = C.new_ops()
    index_case, case = C.verify_case(num_reqs, q_len, 8192, seed=4)
    lane = C.mtp_lane()
    indices = C.run_select(new, index_case, sx_mtp_lane=lane)
    positions = index_case.positions
    live = indices >= 0
    assert bool((indices.long() <= positions[:, None]).logical_or(~live).all())
    for request in range(num_reqs):
        start = request * q_len
        for i in range(q_len):
            row = start + i
            position = int(positions[row])
            row_tokens = indices[row][indices[row] >= 0]
            # The causal tail of the open group is exactly up to position.
            tail_start = (position + 1) // C.COMPRESS_RATIO * C.COMPRESS_RATIO
            tail = row_tokens[row_tokens >= tail_start]
            assert tail.tolist() == list(range(tail_start, position + 1))[
                : C.COMPRESS_RATIO - 1
            ], (request, i)
    case.indices.copy_(indices)
    out = torch.empty_like(case.q)
    C.run_attention(new, case, out, sx_mtp_lane=lane)
    reference = C.reference_attention(case)
    torch.testing.assert_close(out.float(), reference, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("num_reqs,q_len", [(8, 5), (12, 4), (12, 5), (16, 3)])
def test_verify_rows_graph_replay(num_reqs, q_len):
    """Capture the MTP-lane path once (indexer + attention), then replay with
    new metadata/inputs and a poisoned output; compare with dev2 eager."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    lane = C.mtp_lane()
    index_case, case = C.verify_case(num_reqs, q_len, 8192, seed=5)
    selected = torch.empty_like(case.indices)
    out = torch.empty_like(case.q)

    def step():
        C.run_select(new, index_case, out=selected, sx_mtp_lane=lane)
        case.indices.copy_(selected)
        C.run_attention(new, case, out, sx_mtp_lane=lane)

    graph = C.capture(step)
    for scenario in range(4):
        fresh_index, fresh = C.verify_case(
            num_reqs, q_len, 2048 if scenario % 2 else 8192, seed=50 + scenario
        )
        # Graph inputs are fixed buffers: copy the new data in place. Pages
        # beyond this cache are masked by the kernels as in dev2.
        index_case.q.copy_(fresh_index.q)
        index_case.table.copy_(fresh_index.table)
        index_case.positions.copy_(fresh_index.positions)
        index_case.seq_lens.copy_(fresh_index.seq_lens)
        rows_c = min(index_case.cache.shape[0], fresh_index.cache.shape[0])
        index_case.cache[:rows_c].copy_(fresh_index.cache[:rows_c])
        case.q.copy_(fresh.q)
        case.gate.copy_(fresh.gate)
        case.table.copy_(fresh.table)
        case.positions.copy_(fresh.positions)
        case.seq_lens.copy_(fresh.seq_lens)
        if scenario == 3:
            case.table.copy_(case.table.roll(1, dims=-1))
        del fresh, fresh_index
        out.fill_(float("nan"))
        selected.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        expected_indices = C.run_select(dev2, index_case)
        assert torch.equal(expected_indices, selected), scenario
        expected = torch.full_like(out, float("nan"))
        eager_case = C.AttentionCase(**{**case.__dict__, "indices": expected_indices})
        C.run_attention(dev2, eager_case, expected)
        torch.cuda.synchronize()
        assert C.bitwise_equal(expected, out), (scenario, C.first_mismatch(expected, out))


# ---------------------------------------------------------------------------
# Microbenchmark
# ---------------------------------------------------------------------------
def _bench_width(num_reqs: int, q_len: int, context: int, layers: int = 12):
    new, dev2 = C.new_ops(), C.dev2_ops()
    lane = C.mtp_lane()
    base_index, base_case = C.verify_case(num_reqs, q_len, context, seed=100)
    index_cases, cases, outs, sels = [], [], [], []
    for layer in range(layers):
        index_cases.append(
            C.IndexerCase(**{**base_index.__dict__, "q": torch.randn_like(base_index.q)})
        )
        cases.append(
            C.AttentionCase(
                **{
                    **base_case.__dict__,
                    "q": torch.randn_like(base_case.q),
                    "gate": torch.randn_like(base_case.gate),
                    "indices": C.run_select(dev2, index_cases[-1]),
                }
            )
        )
        outs.append(torch.empty_like(base_case.q))
        sels.append(torch.empty_like(base_case.indices))

    def attention(ops, **kwargs):
        def run():
            for case, out in zip(cases, outs):
                C.run_attention(ops, case, out, **kwargs)

        return run

    def indexer(ops, **kwargs):
        def run():
            for case, sel in zip(index_cases, sels):
                C.run_select(ops, case, out=sel, **kwargs)

        return run

    graphs = {
        "attn_dev2": C.capture(attention(dev2)),
        "attn_mtp": C.capture(attention(new, sx_mtp_lane=lane)),
        "index_dev2": C.capture(indexer(dev2)),
        "index_mtp": C.capture(indexer(new, sx_mtp_lane=lane)),
    }
    timings = C.paired_graph_ms(graphs)
    timings["attn_delta_ms"] = timings["attn_mtp"] - timings["attn_dev2"]
    timings["index_delta_ms"] = timings["index_mtp"] - timings["index_dev2"]
    C.report(
        f"x{layers} layers, B={num_reqs} q={q_len} rows={num_reqs * q_len} "
        f"ctx={context}",
        timings,
    )
    return timings


@pytest.mark.skipif(C.no_bench(), reason="SX_TEST_NO_BENCH")
@pytest.mark.parametrize("context", (2048, 8192, 32768))
def test_verify_rows_microbench(context):
    # Verify widths 33..63 (k=4: B7..12; k=3: B9..15; k=2: B11..21).
    for num_reqs, q_len in ((7, 5), (8, 5), (9, 4), (10, 5), (12, 5), (16, 3), (21, 3)):
        _bench_width(num_reqs, q_len, context)
        torch.cuda.empty_cache()


def main():
    if not C.is_sm70():
        raise SystemExit("requires a V100 / SM70 GPU")
    for num_reqs, q_len in ((8, 5), (12, 5), (16, 3)):
        test_attention_verify_rows_bitwise(num_reqs, q_len, 8192)
        test_indexer_verify_rows_bitwise(num_reqs, q_len, 8192)
    print("[sx-qsa-mtp] verify rows bitwise: OK", flush=True)
    for context in (8192, 32768):
        test_verify_rows_microbench(context)


if __name__ == "__main__":
    main()
