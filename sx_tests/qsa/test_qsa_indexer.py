# SPDX-License-Identifier: Apache-2.0
"""QSA indexer (weight-free top-k selection) checks, new vs old.

1. SX_OPT_QSA_HOST_BOUND: the single-request cuBLAS path takes its score width
   from host seq_lens instead of int(all_visible.max().item()). Output must be
   bitwise identical and the call must not block the CPU on the GPU.
2. SX_OPT_QSA_SCORER_STRIDE (design_1 [C7]): grid-stride paged scorer for
   2 <= M <= 32. Scores of every live column and visible_blocks bitwise equal;
   downstream selected indices bitwise equal, eager and under graph replay.
3. SX_OPT_QSA_MIXED_CUBLAS (design_3 [P2]): in a mixed batch, a large prefill
   request is scored with the single-request cuBLAS path. Its rows must be
   bitwise equal to the same request run alone through the OLD code (C1
   production path); the other rows must be bitwise equal to the old mixed
   result. Differences versus the old mixed Triton scores are reported, not
   asserted (they are the expected Triton-FMA vs cuBLAS-HMMA rounding).
Every new path is also checked for engagement (strided kernel launched,
exactly the qualifying requests routed to cuBLAS), so a silently rejected
gate cannot make the bitwise comparisons pass vacuously.

Single GPU (V100). From the source-tree root (the installed, overlaid vllm is
imported; see _qsa_test_utils._prefer_installed_vllm):
  /opt/venv/bin/python -m pytest -q -s sx_tests/qsa/test_qsa_indexer.py
  /opt/venv/bin/python sx_tests/qsa/test_qsa_indexer.py   # quick + bench
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch

from _qsa_test_utils import (
    COMPRESS_RATIO,
    COMPRESSED_PAGE,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_TABLE_WIDTH,
    SCHED_BLOCK,
    TOKEN_TOPK,
    bitwise_equal,
    capture,
    is_sm70,
    make_block_table,
    median_ms,
    new_ops,
    old_ops,
    paired_graph_ms,
    patched,
    report,
    require_sm70,
)

MAX_CONTEXT = INDEX_TABLE_WIDTH * SCHED_BLOCK  # 131712 tokens of capacity


@dataclass
class IndexCase:
    q: torch.Tensor
    cache: torch.Tensor
    table: torch.Tensor
    token_to_req: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor


def build_index_case(
    request_rows: list[int],
    request_contexts: list[int],
    seed: int = 0,
    padded_rows: int = 0,
) -> IndexCase:
    """Flat token table: request r contributes request_rows[r] rows ending at
    position request_contexts[r] - 1 (the QSA metadata builder's contract)."""
    assert len(request_rows) == len(request_contexts)
    generator = torch.Generator().manual_seed(seed)
    table, num_pages = make_block_table(
        generator, request_contexts, SCHED_BLOCK, INDEX_TABLE_WIDTH
    )
    token_to_req, positions, starts = [], [], [0]
    for request, (rows, context) in enumerate(zip(request_rows, request_contexts)):
        assert 1 <= rows <= context <= MAX_CONTEXT
        token_to_req += [request] * rows
        positions += list(range(context - rows, context))
        starts.append(starts[-1] + rows)
    token_to_req += [0] * padded_rows
    positions += [-1] * padded_rows
    total_rows = len(positions)
    device = "cuda"
    cache = torch.randn(
        num_pages, COMPRESSED_PAGE, 1, INDEX_DIM, dtype=torch.float16, device=device
    )
    q = torch.randn(
        total_rows, INDEX_HEADS, INDEX_DIM, dtype=torch.float16, device=device
    )
    seq_lens_cpu = torch.tensor(request_contexts, dtype=torch.int32)
    return IndexCase(
        q=q,
        cache=cache,
        table=table.to(device),
        token_to_req=torch.tensor(token_to_req, dtype=torch.int32, device=device),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        seq_lens=seq_lens_cpu.to(device),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        seq_lens_cpu=seq_lens_cpu,
    )


def select(ops, case: IndexCase, host: bool, rows=None, table=None, t2r=None,
           seq=None, out=None):
    rows = slice(None) if rows is None else rows
    kwargs = {}
    if host:
        kwargs = dict(
            query_start_loc_cpu=case.query_start_loc_cpu,
            seq_lens_cpu=case.seq_lens_cpu,
        )
    return ops.qsa_select_paged_tokens(
        case.q[rows],
        case.cache,
        case.table if table is None else table,
        case.token_to_req[rows] if t2r is None else t2r,
        case.positions[rows],
        case.seq_lens if seq is None else seq,
        TOKEN_TOPK,
        COMPRESS_RATIO,
        out,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _sm70():
    require_sm70()


# ---------------------------------------------------------------------------
# 1. Host-derived score width (no .item())
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("padded_rows", (0, 8))
@pytest.mark.parametrize("context", (2048, 8192, 32768))
@pytest.mark.parametrize("rows", (784, 3136))
def test_single_request_host_bound_bitwise_and_sync_free(rows, context, padded_rows):
    if rows > context:
        pytest.skip("chunk longer than the context")
    case = build_index_case([rows], [context], seed=rows + context,
                            padded_rows=padded_rows)
    old = old_ops()
    new = new_ops()
    expected = select(old, case, host=False)
    select(new, case, host=True)  # JIT warm-up outside the sync check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        actual = select(new, case, host=True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert bitwise_equal(expected, actual)
    # Without host metadata the new code must equal the old one as well.
    assert bitwise_equal(expected, select(new, case, host=False))


def test_host_bound_debug_check_detects_wrong_metadata():
    new = new_ops()
    case = build_index_case([784], [8192], seed=7)
    with patched(new, _SX_OPT_QSA_HOST_BOUND_CHECK=True):
        select(new, case, host=True)  # exact metadata passes
        case.seq_lens_cpu = case.seq_lens_cpu + COMPRESS_RATIO
        with pytest.raises(RuntimeError):
            select(new, case, host=True)
            torch.cuda.synchronize()


def cpu_blocking_ms(fn, spin_cycles: int = 200_000_000) -> float:
    """CPU wall time of fn() issued behind a long GPU spin kernel."""
    torch.cuda.synchronize()
    torch.cuda._sleep(spin_cycles)
    start = time.perf_counter()
    fn()
    elapsed = (time.perf_counter() - start) * 1e3
    torch.cuda.synchronize()
    return elapsed


def bench_host_bound(rows=784, context=8192, layers=12):
    old = old_ops()
    new = new_ops()
    case = build_index_case([rows], [context], seed=11)
    spin_ms = median_ms(lambda: torch.cuda._sleep(200_000_000), iters=5, warmup=1)

    def many(ops, host):
        def run():
            for _ in range(layers):
                select(ops, case, host)

        return run

    many(old, False)()  # JIT warm-up
    many(new, True)()
    torch.cuda.synchronize()
    result = {
        "gpu_spin_ms": spin_ms,
        "old_cpu_block_ms": cpu_blocking_ms(many(old, False)),
        "new_cpu_block_ms": cpu_blocking_ms(many(new, True)),
        "old_gpu_ms_per_call": median_ms(lambda: select(old, case, False)),
        "new_gpu_ms_per_call": median_ms(lambda: select(new, case, True)),
    }
    report(f"indexer single-request host bound, rows={rows}, ctx={context}, "
           f"{layers} calls behind a GPU spin", result)
    return result


def test_host_bound_does_not_block_cpu():
    result = bench_host_bound()
    # 12 old calls wait for the spin kernel; 12 new calls only enqueue work.
    assert result["new_cpu_block_ms"] < 0.5 * result["gpu_spin_ms"], result


# ---------------------------------------------------------------------------
# 2. Strided decode scorer
# ---------------------------------------------------------------------------
EDGE_CONTEXTS = (1, 3, 4, 5, 255, 256, 257, 783, 784, 785, 2048, 8191,
                 32768, 131071, 131072)


def decode_contexts(rows: int, seed: int, long: bool) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    contexts = []
    for row in range(rows):
        if row < len(EDGE_CONTEXTS) and row % 2 == 0:
            contexts.append(EDGE_CONTEXTS[(row + seed) % len(EDGE_CONTEXTS)])
        else:
            high = 131072 if long else 8192
            contexts.append(int(torch.randint(1, high + 1, (1,), generator=generator)))
    return contexts


class CountingKernel:
    """Wrap a Triton kernel and count ``kernel[grid](...)`` launches."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


@pytest.mark.parametrize("long", (False, True))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 8, 16, 17, 24, 32))
def test_strided_scorer_bitwise(rows, long):
    old = old_ops()
    new = new_ops()
    contexts = decode_contexts(rows, seed=rows, long=long)
    case = build_index_case([1] * rows, contexts, seed=rows * 7 + int(long))
    args = (case.q, case.cache, case.table, case.token_to_req, case.positions,
            case.seq_lens, COMPRESS_RATIO)
    old_logits, old_visible = old.qsa_mqa_paged(*args)
    strided = CountingKernel(new._qsa_mqa_paged_strided_kernel)
    # The strided scorer is opt-in after V100 validation; force it on here.
    with patched(new, _qsa_mqa_paged_strided_kernel=strided, _SX_OPT_QSA_SCORER_STRIDE=True):
        new_logits, new_visible = new.qsa_mqa_paged(*args)
    torch.cuda.synchronize()
    # The comparison below is only meaningful if the new kernel really ran.
    assert strided.launches == (1 if 2 <= rows <= 32 else 0), strided.launches
    assert torch.equal(old_visible, new_visible)
    for row, visible in enumerate(old_visible.tolist()):
        assert bitwise_equal(old_logits[row, :visible], new_logits[row, :visible]), row
    # Downstream selection (decode path, no host metadata in FULL graphs).
    with patched(new, _SX_OPT_QSA_SCORER_STRIDE=True):
        assert bitwise_equal(select(old, case, False), select(new, case, False))
    with patched(new, _SX_OPT_QSA_SCORER_STRIDE=False):
        assert bitwise_equal(select(old, case, False), select(new, case, False))


@pytest.mark.parametrize("rows", (8, 16, 24, 32))
def test_strided_selection_graph_replay(rows):
    old = old_ops()
    new = new_ops()
    case = build_index_case([1] * rows, decode_contexts(rows, 5, True), seed=rows)
    out = torch.empty((rows, TOKEN_TOPK + COMPRESS_RATIO - 1), dtype=torch.int32,
                      device="cuda")
    graph = capture(lambda: select(new, case, False, out=out))
    for scenario in range(4):
        fresh = build_index_case(
            [1] * rows, decode_contexts(rows, 20 + scenario, scenario % 2 == 1),
            seed=100 + scenario,
        )
        # Keep the cache buffer; re-point the table into its page range.
        case.q.copy_(fresh.q)
        case.table.copy_(fresh.table.clamp(max=case.cache.shape[0] - 1))
        case.positions.copy_(fresh.positions)
        case.seq_lens.copy_(fresh.seq_lens)
        if scenario == 3:
            case.token_to_req[::3] = -1
        out.fill_(-7)
        graph.replay()
        expected = select(old, case, False)
        torch.cuda.synchronize()
        assert bitwise_equal(expected, out), (rows, scenario)


def bench_scorer(rows_list=(8, 16, 24, 32), long=False, layers=12):
    old = old_ops()
    new = new_ops()
    for rows in rows_list:
        case = build_index_case([1] * rows, decode_contexts(rows, 3, long), seed=9)
        outs = [torch.empty((rows, TOKEN_TOPK + COMPRESS_RATIO - 1),
                            dtype=torch.int32, device="cuda") for _ in range(layers)]

        def run(ops):
            def body():
                for out in outs:
                    select(ops, case, False, out=out)

            return body

        def run_scorer(ops):
            def body():
                for _ in range(layers):
                    ops.qsa_mqa_paged(case.q, case.cache, case.table,
                                      case.token_to_req, case.positions,
                                      case.seq_lens, COMPRESS_RATIO)

            return body

        timings = paired_graph_ms({
            "old_select": capture(run(old)),
            "new_select": capture(run(new)),
            "old_scorer": capture(run_scorer(old)),
            "new_scorer": capture(run_scorer(new)),
        })
        report(f"indexer decode x{layers}, M={rows}, "
               f"contexts={'<=128K' if long else '<=8K'}", timings)


def test_scorer_microbench():
    bench_scorer(long=False)
    bench_scorer(long=True)


# ---------------------------------------------------------------------------
# 3. Mixed batch: per-request cuBLAS scorer (design_3 [P2])
# ---------------------------------------------------------------------------
MIXED_CASES = [
    # (decode rows, [prefill rows], [prefill contexts])
    (23, [784], [8192]),
    (23, [784], [32768]),
    (23, [3136], [32768]),
    (16, [784], [8192 + 784 * 3]),
    (7, [600, 784], [8192, 32768]),
    (23, [784], [2048]),  # does not qualify: must equal the old result
]


def mixed_case(decodes: int, prefill_rows: list[int], prefill_contexts: list[int],
               seed: int) -> tuple[IndexCase, list[slice]]:
    generator = torch.Generator().manual_seed(seed)
    decode_contexts_ = [int(torch.randint(1024, 8193, (1,), generator=generator))
                        for _ in range(decodes)]
    # The V2 runner orders requests by scheduled tokens (decodes first).
    order = sorted(zip(prefill_rows, prefill_contexts))
    rows = [1] * decodes + [r for r, _ in order]
    contexts = decode_contexts_ + [c for _, c in order]
    case = build_index_case(rows, contexts, seed=seed)
    starts = case.query_start_loc_cpu.tolist()
    slices = [slice(starts[decodes + i], starts[decodes + i + 1])
              for i in range(len(order))]
    return case, slices


@pytest.mark.parametrize("scenario", range(len(MIXED_CASES)))
def test_mixed_batch_cublas_rows_match_alone(scenario):
    decodes, prefill_rows, prefill_contexts = MIXED_CASES[scenario]
    old = old_ops()
    new = new_ops()
    case, prefill_slices = mixed_case(decodes, prefill_rows, prefill_contexts,
                                      seed=40 + scenario)
    old_mixed = select(old, case, host=False)
    select(new, case, host=True)  # JIT warm-up outside the sync check
    torch.cuda.synchronize()
    cublas_requests: list[int] = []
    original_mixed = new._qsa_select_mixed_batch

    def spy(segments, *args, **kwargs):
        cublas_requests.extend(s[2] for s in segments if s[2] is not None)
        return original_mixed(segments, *args, **kwargs)

    torch.cuda.set_sync_debug_mode("error")
    try:
        with patched(new, _qsa_select_mixed_batch=spy):
            new_mixed = select(new, case, host=True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    # Engagement: exactly the qualifying requests take the per-request cuBLAS
    # scorer (otherwise the bitwise checks below could pass vacuously).
    expected_cublas = [
        decodes + index
        for index, rows in enumerate(prefill_slices)
        if (rows.stop - rows.start) >= 512
        and (rows.stop - rows.start)
        * max(512, int(case.seq_lens_cpu[decodes + index]) // COMPRESS_RATIO)
        >= 1024**2
    ]
    assert cublas_requests == expected_cublas, (cublas_requests, expected_cublas)
    decode_rows = slice(0, decodes)
    assert bitwise_equal(old_mixed[decode_rows], new_mixed[decode_rows])
    differing = 0
    for index, rows in enumerate(prefill_slices):
        request = decodes + index
        alone = select(
            old, case, host=False, rows=rows,
            table=case.table[request : request + 1],
            t2r=torch.zeros(rows.stop - rows.start, dtype=torch.int32, device="cuda"),
            seq=case.seq_lens[request : request + 1],
        )
        assert bitwise_equal(alone, new_mixed[rows]), (scenario, index)
        differing += int((old_mixed[rows] != new_mixed[rows]).any(dim=1).sum())
    qualifies = any(
        rows >= 512 and rows * max(512, context // COMPRESS_RATIO) >= 1024**2
        for rows, context in zip(prefill_rows, prefill_contexts)
    )
    if not qualifies:
        assert bitwise_equal(old_mixed, new_mixed)
    with patched(new, _SX_OPT_QSA_MIXED_CUBLAS=False):
        assert bitwise_equal(old_mixed, select(new, case, host=True))
    report(f"indexer mixed batch scenario {scenario}", {
        "decodes": decodes, "prefill_rows": str(prefill_rows),
        "prefill_contexts": str(prefill_contexts),
        "prefill_rows_with_changed_selection_vs_old_mixed": differing,
    })


def test_mixed_cublas_debug_check_detects_wrong_metadata():
    """SX_OPT_QSA_HOST_BOUND_CHECK=1 also validates the per-request widths."""
    new = new_ops()
    case, _ = mixed_case(23, [784], [8192], seed=5)
    with patched(new, _SX_OPT_QSA_HOST_BOUND_CHECK=True):
        select(new, case, host=True)  # exact metadata passes
        torch.cuda.synchronize()
        wrong = case.seq_lens_cpu.clone()
        wrong[-1] += 64 * COMPRESS_RATIO  # the prefill: +64 score columns
        case.seq_lens_cpu = wrong
        with pytest.raises(RuntimeError):
            select(new, case, host=True)
            torch.cuda.synchronize()


def bench_mixed():
    old = old_ops()
    new = new_ops()
    for decodes, prefill_rows, prefill_contexts in MIXED_CASES[:3]:
        case, _ = mixed_case(decodes, prefill_rows, prefill_contexts, seed=77)
        result = {
            "old_ms": median_ms(lambda: select(old, case, False)),
            "new_ms": median_ms(lambda: select(new, case, True)),
        }
        report(f"indexer mixed {decodes} decodes + {prefill_rows} rows @ "
               f"{prefill_contexts} (per layer)", result)


def test_mixed_microbench():
    bench_mixed()


def main():
    if not is_sm70():
        raise SystemExit("requires a V100 / SM70 GPU")
    test_single_request_host_bound_bitwise_and_sync_free(784, 8192, 0)
    test_single_request_host_bound_bitwise_and_sync_free(3136, 32768, 8)
    for rows in (2, 8, 17, 24, 32):
        test_strided_scorer_bitwise(rows, True)
    for scenario in range(len(MIXED_CASES)):
        test_mixed_batch_cublas_rows_match_alone(scenario)
    print("[sx-qsa] indexer bitwise checks: OK", flush=True)
    bench_host_bound()
    bench_host_bound(rows=784, context=32768)
    bench_scorer(long=False)
    bench_scorer(long=True)
    bench_mixed()


if __name__ == "__main__":
    main()
