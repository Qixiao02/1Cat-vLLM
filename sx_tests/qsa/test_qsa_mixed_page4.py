# SPDX-License-Identifier: Apache-2.0
"""Mixed prefill+decode QSA sparse attention (SX_OPT_QSA_MIXED_GROUPS).

The SM70 grouped page4 kernel runs ONE CTA per 8 consecutive query rows and
that CTA walks the union of the 8 rows' selected pages. In a mixed step the
decode rows (one row per request, disjoint selections) formed groups with up
to 8x the pages of a prefill group, which made
flash_attention_grouped_verify_e5m2_partial 3.6 ms/layer instead of 1.5.
The new routing never groups rows of different requests: decode rows and
per-request remainders go to the row-wise XQA page4 kernel.

Asserted:
  * the request-aligned routing actually runs for every mixed scenario and
    does not run for single-request batches (engagement check);
  * every prefill request's rows are BITWISE equal to the same request run
    alone through the OLD code (the C1 production path);
  * decode rows match the old mixed result within FP16 attention tolerance
    (different kernel; NOT bitwise by design);
  * single-request batches and the switch-off path are bitwise equal to old.
Printed: old vs new per-layer time (CUDA events, median of 50).

Single GPU (V100) with the Flash-V100 grouped page4 ABI (the deployed image).
From the source-tree root:
  /opt/venv/bin/python -m pytest -q -s sx_tests/qsa/test_qsa_mixed_page4.py
  /opt/venv/bin/python sx_tests/qsa/test_qsa_mixed_page4.py   # checks + bench
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from _qsa_test_utils import (
    HEAD_DIM,
    MAX_MODEL_LEN,
    Q_HEADS,
    SELECTION,
    bitwise_equal,
    is_sm70,
    make_block_table,
    median_ms,
    new_ops,
    old_ops,
    patched,
    report,
    require_sm70,
    sample_row_selection,
)


def grouped_page4_available() -> bool:
    if not is_sm70():
        return False
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        return False
    new = new_ops()
    return bool(
        hasattr(flash_attn_v100_cuda, "decode_paged_xqa_fwd")
        and new._qsa_grouped_page4_supported(flash_attn_v100_cuda, "auto")
        and new._SM70_QSA_GROUPED_PAGE4
        and new._SM70_QSA_XQA_PAGE4
    )


@pytest.fixture(autouse=True)
def _requirements():
    require_sm70()
    if not grouped_page4_available():
        pytest.skip("Flash-V100 grouped page4 ABI is unavailable")


@dataclass
class MixedCase:
    q: torch.Tensor
    gate: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    table: torch.Tensor
    token_to_req: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    prefill_slices: list[tuple[int, int, int]]  # (request, start, end)
    decodes: int


def build_mixed_case(
    decodes: int,
    prefills: list[tuple[int, int]],  # (rows, context)
    page_size: int = 784,
    decode_context: tuple[int, int] = (1536, 3072),
    seed: int = 0,
) -> MixedCase:
    generator = torch.Generator().manual_seed(seed)
    decode_contexts = [
        int(torch.randint(decode_context[0], decode_context[1] + 1, (1,),
                          generator=generator))
        for _ in range(decodes)
    ]
    prefills = sorted(prefills)  # the V2 runner orders by scheduled tokens
    contexts = decode_contexts + [context for _, context in prefills]
    request_rows = [1] * decodes + [rows for rows, _ in prefills]
    width = math.ceil(MAX_MODEL_LEN / page_size)
    table, num_blocks = make_block_table(generator, contexts, page_size, width)
    total_rows = sum(request_rows)
    indices = torch.full((total_rows, SELECTION), -1, dtype=torch.int32)
    token_to_req = torch.empty(total_rows, dtype=torch.int32)
    positions = torch.empty(total_rows, dtype=torch.int64)
    starts = [0]
    prefill_slices = []
    for request, (rows, context) in enumerate(zip(request_rows, contexts)):
        start = starts[-1]
        end = start + rows
        starts.append(end)
        token_to_req[start:end] = request
        positions[start:end] = torch.arange(context - rows, context)
        base = None
        if rows > 1:
            base = torch.randperm(max(context // 4, 1), generator=generator)[:512]
            prefill_slices.append((request, start, end))
        for row in range(start, end):
            indices[row] = sample_row_selection(
                generator,
                int(positions[row]) + 1,
                base_blocks=base,
                resample_fraction=0.05 if base is not None else 0.0,
            )
    device = "cuda"
    kv = torch.randn(num_blocks, 2, page_size, 1, HEAD_DIM, dtype=torch.float16,
                     device=device)
    k, v = kv.unbind(1)
    return MixedCase(
        q=torch.randn(total_rows, Q_HEADS, HEAD_DIM, dtype=torch.float16,
                      device=device),
        gate=torch.randn(total_rows, Q_HEADS * HEAD_DIM, dtype=torch.float16,
                         device=device),
        k=k,
        v=v,
        kv=kv,
        indices=indices.to(device),
        table=table.to(device),
        token_to_req=token_to_req.to(device),
        positions=positions.to(device),
        seq_lens=torch.tensor(contexts, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        prefill_slices=prefill_slices,
        decodes=decodes,
    )


def attention(ops, case: MixedCase, host: bool, rows: slice | None = None,
              table=None, t2r=None, seq=None) -> torch.Tensor:
    rows = slice(None) if rows is None else rows
    out = torch.full_like(case.q[rows], float("nan"))
    kwargs = {"query_start_loc_cpu": case.query_start_loc_cpu} if host else {}
    ops.qsa_sparse_paged_attention(
        case.q[rows],
        case.k,
        case.v,
        case.indices[rows],
        case.table if table is None else table,
        case.token_to_req[rows] if t2r is None else t2r,
        out=out,
        output_gate=case.gate[rows],
        query_positions=case.positions[rows],
        sequence_lengths=case.seq_lens if seq is None else seq,
        kv_cache_dtype="auto",
        **kwargs,
    )
    return out


def alone(case: MixedCase, request: int, start: int, end: int) -> torch.Tensor:
    """The request by itself (C1 production path) through the OLD code."""
    return attention(
        old_ops(),
        case,
        host=False,
        rows=slice(start, end),
        table=case.table[request : request + 1],
        t2r=torch.zeros(end - start, dtype=torch.int32, device="cuda"),
        seq=case.seq_lens[request : request + 1],
    )


MIXED = [
    # (decodes, [(prefill rows, prefill context)], page size)
    (16, [(784, 8192)], 784),       # profile mixed_m16 step
    (13, [(784, 8192)], 784),       # decode count not a multiple of 8
    (23, [(784, 32768)], 784),
    (8, [(450, 450)], 784),         # real traffic: short prompt, one chunk
    (7, [(300, 300), (784, 4704)], 784),  # two prefills, remainder rows
    (3, [(61, 61), (130, 2000)], 784),    # small requests
    (16, [(784, 8192)], 400),
    (1, [(784, 8192)], 784),
    (0, [(450, 450), (784, 8192)], 784),  # prefills only, old boundary group
]


def attention_with_route_count(ops, case: MixedCase) -> tuple[torch.Tensor, int]:
    """Run the new ops with host metadata; count request-aligned routings."""
    calls = []
    original = ops._qsa_sparse_paged_attention_sm70_page4_segments

    def spy(segments, *args, **kwargs):
        calls.append(list(segments))
        return original(segments, *args, **kwargs)

    with patched(ops, _qsa_sparse_paged_attention_sm70_page4_segments=spy):
        out = attention(ops, case, host=True)
    return out, len(calls)


@pytest.mark.parametrize("scenario", range(len(MIXED)))
def test_mixed_prefill_rows_bitwise_equal_alone(scenario):
    decodes, prefills, page_size = MIXED[scenario]
    case = build_mixed_case(decodes, prefills, page_size, seed=scenario)
    old = old_ops()
    new = new_ops()
    new_out, routed = attention_with_route_count(new, case)
    old_out = attention(old, case, host=False)
    torch.cuda.synchronize()
    # The request-aligned routing must run exactly when some old 8-row group
    # mixes >= _SX_QSA_MIXED_MIN_GROUP_REQUESTS requests (validation tuning:
    # prefill-only multi-request steps keep the faster single launch).
    expect_routed = int(
        new._qsa_page4_old_group_max_requests(case.query_start_loc_cpu, case.q.shape[0])
        >= new._SX_QSA_MIXED_MIN_GROUP_REQUESTS
    )
    assert routed == expect_routed, (routed, expect_routed)
    assert not torch.isnan(new_out).any()
    if not routed:
        assert bitwise_equal(old_out, new_out)
        with patched(new, _SX_QSA_MIXED_MIN_GROUP_REQUESTS=0):
            forced, forced_routed = attention_with_route_count(new, case)
        torch.cuda.synchronize()
        assert forced_routed == 1
        new_out = forced  # validate the routed path's per-request exactness too
    min_rows = new._SM70_QSA_XQA_PAGE4_MIN_ROWS
    for request, start, end in case.prefill_slices:
        if end - start < min_rows:
            # Alone, such a small request takes the Triton split-K route, so
            # only the tolerance check below applies to it.
            continue
        expected = alone(case, request, start, end)
        torch.cuda.synchronize()
        # Split the check so a failure names its cause: grouped rows come from
        # the same 8-row groups as alone; the per-request remainder rows come
        # from a different row-wise XQA batch (per-row batch invariance).
        grouped_rows = (end - start) // 8 * 8
        assert bitwise_equal(
            expected[:grouped_rows], new_out[start : start + grouped_rows]
        ), ("grouped rows differ from the request alone", scenario, request)
        assert bitwise_equal(
            expected[grouped_rows:], new_out[start + grouped_rows : end]
        ), ("row-wise XQA remainder rows differ from alone", scenario, request)
    difference = (new_out.float() - old_out.float()).abs()
    scale = old_out.float().abs().max().item()
    max_difference = difference.max().item()
    decode_difference = difference[: case.decodes].max().item() if case.decodes else 0.0
    report(f"mixed page4 scenario {scenario} decodes={decodes} prefills={prefills} "
           f"page={page_size}", {
               "max_abs_diff_vs_old": max_difference,
               "decode_rows_max_abs_diff_vs_old": decode_difference,
               "output_abs_max": scale,
           })
    # Different kernels for the decode rows: FP16 attention-level agreement.
    assert max_difference <= 1e-2 * max(scale, 1.0), max_difference
    with patched(new, _SX_OPT_QSA_MIXED_GROUPS=False):
        switched_off = attention(new, case, host=True)
    torch.cuda.synchronize()
    assert bitwise_equal(old_out, switched_off)


@pytest.mark.parametrize("rows", (784, 450, 3136))
def test_single_request_unchanged(rows):
    case = build_mixed_case(0, [(rows, 8192)], seed=rows)
    new_out, routed = attention_with_route_count(new_ops(), case)
    old_out = attention(old_ops(), case, host=False)
    torch.cuda.synchronize()
    assert routed == 0, routed  # identical plan: the literal old path runs
    assert bitwise_equal(old_out, new_out)


def bench_mixed_page4():
    old = old_ops()
    new = new_ops()
    for decodes, prefills, page_size in (
        (16, [(784, 8192)], 784),
        (23, [(784, 32768)], 784),
        (8, [(450, 450)], 784),
        (16, [(784, 8192)], 400),
        # Several prefills: the new routing launches one grouped kernel per
        # request (serialized partial waves) instead of one launch. Watch
        # these for a regression against the old single launch.
        (0, [(450, 450), (784, 8192)], 784),
        (8, [(300, 300), (450, 450), (784, 8192)], 784),
        (0, [(1570, 8192), (1570, 32768)], 784),
    ):
        case = build_mixed_case(decodes, prefills, page_size, seed=99)
        result = {
            "old_ms": median_ms(lambda: attention(old, case, host=False)),
            "new_ms": median_ms(lambda: attention(new, case, host=True)),
        }
        result["saving_ms_per_layer"] = result["old_ms"] - result["new_ms"]
        report(f"mixed page4 attention per layer, {decodes} decodes @1.5-3K + "
               f"{prefills} page={page_size}", result)


def test_mixed_page4_microbench():
    bench_mixed_page4()


def main():
    if not grouped_page4_available():
        raise SystemExit("requires V100 + Flash-V100 grouped page4 ABI")
    for scenario in range(len(MIXED)):
        test_mixed_prefill_rows_bitwise_equal_alone(scenario)
    print("[sx-qsa] mixed page4 routing checks: OK", flush=True)
    bench_mixed_page4()


if __name__ == "__main__":
    main()
