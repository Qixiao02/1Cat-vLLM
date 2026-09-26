# SPDX-License-Identifier: Apache-2.0
"""QSA sparse decode attention: design_1 [C1] + design_4 [MR4].

new = live vllm ops (two-warp partial for M <= 32, physical-address resolver
for 1 <= M <= 32 and any page size); old = git baseline 71c1822 copy.
Every case must be bitwise identical (FP16 output compared as int16 bits),
eagerly and after CUDA-graph replay with changed inputs and poisoned output.

Single GPU (V100). Run:
  /opt/venv/bin/python -m pytest -q -s sx_tests/qsa/test_qsa_decode_attention.py
  /opt/venv/bin/python sx_tests/qsa/test_qsa_decode_attention.py   # quick + bench
"""

from __future__ import annotations

import contextlib
import dataclasses
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
    capture,
    is_sm70,
    make_block_table,
    new_ops,
    old_ops,
    paired_graph_ms,
    patched,
    report,
    require_sm70,
    sample_row_selection,
)

DECODE_ROWS = (1, 2, 3, 4, 8, 16, 17, 20, 24, 31, 32)
CONTEXTS = (2048, 8192, 32768)
PAGE_SIZES = (784, 400)


@dataclass
class DecodeCase:
    q: torch.Tensor
    gate: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kv: torch.Tensor | None
    indices: torch.Tensor
    table: torch.Tensor
    token_to_req: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    page_size: int


def build_decode_case(
    rows: int,
    context: int,
    page_size: int,
    interleaved: bool = True,
    seed: int = 0,
    padded_rows: int = 0,
    corrupt: bool = True,
) -> DecodeCase:
    generator = torch.Generator().manual_seed(seed * 1000003 + rows * 101 + context)
    real_rows = rows - padded_rows
    assert real_rows >= 1
    contexts = [
        max(1, context - int(torch.randint(0, 97, (1,), generator=generator)))
        for _ in range(real_rows)
    ]
    width = math.ceil(MAX_MODEL_LEN / page_size)
    table, num_blocks = make_block_table(generator, contexts, page_size, width)
    indices = torch.full((rows, SELECTION), -1, dtype=torch.int32)
    for row in range(real_rows):
        indices[row] = sample_row_selection(generator, contexts[row])
    token_to_req = torch.zeros(rows, dtype=torch.int32)
    token_to_req[:real_rows] = torch.arange(real_rows, dtype=torch.int32)
    positions = torch.full((rows,), -1, dtype=torch.int64)
    positions[:real_rows] = torch.tensor(contexts, dtype=torch.int64) - 1
    seq_lens = torch.tensor(contexts, dtype=torch.int32)
    if corrupt and real_rows >= 2:
        # Invalid slots, duplicates, out-of-table tokens and a missing page.
        indices[0, ::7] = -1
        indices[1 % real_rows, 64:192] = indices[1 % real_rows, 0:128].clone()
        if real_rows >= 3:
            indices[2, ::11] = width * page_size + 5
        if real_rows >= 4:
            table[3, 0] = -1
            table[3, 1] = num_blocks + 17  # beyond the cache
        if real_rows >= 6:
            token_to_req[5] = -1  # invalid request row
        if real_rows >= 8:
            token_to_req[7] = real_rows + 3  # out-of-range request row
    device = "cuda"
    if interleaved:
        kv = torch.randn(
            num_blocks, 2, page_size, 1, HEAD_DIM, dtype=torch.float16, device=device
        )
        k, v = kv.unbind(1)
    else:
        kv = None
        k = torch.randn(
            num_blocks, page_size, 1, HEAD_DIM, dtype=torch.float16, device=device
        )
        v = torch.randn_like(k)
    q = torch.randn(rows, Q_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    gate = torch.randn(rows, Q_HEADS * HEAD_DIM, dtype=torch.float16, device=device)
    return DecodeCase(
        q=q,
        gate=gate,
        k=k,
        v=v,
        kv=kv,
        indices=indices.to(device),
        table=table.to(device),
        token_to_req=token_to_req.to(device),
        positions=positions.to(device),
        seq_lens=seq_lens.to(device),
        page_size=page_size,
    )


def run_attention(ops, case: DecodeCase, out: torch.Tensor, gated: bool = True):
    return ops.qsa_sparse_paged_attention(
        case.q,
        case.k,
        case.v,
        case.indices,
        case.table,
        case.token_to_req,
        out=out,
        output_gate=case.gate if gated else None,
        query_positions=case.positions,
        sequence_lengths=case.seq_lens,
    )


class LaunchSpy:
    """Wrap a Triton kernel; record the keyword arguments of every launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches: list[dict] = []

    def __getitem__(self, grid):
        launcher = self.kernel[grid]

        def run(*args, **kwargs):
            self.launches.append(kwargs)
            return launcher(*args, **kwargs)

        return run


def compare_once(case: DecodeCase, gated: bool = True, **new_flags) -> None:
    old = old_ops()
    new = new_ops()
    expected = torch.full_like(case.q, float("nan"))
    actual = torch.full_like(case.q, float("nan"))
    run_attention(old, case, expected, gated)
    resolver = LaunchSpy(new._qsa_resolve_physical_indices_kernel)
    partial = LaunchSpy(new._qsa_sparse_paged_gqa_splitk_kernel)
    with patched(new, **new_flags), patched(
        new,
        _qsa_resolve_physical_indices_kernel=resolver,
        _qsa_sparse_paged_gqa_splitk_kernel=partial,
    ):
        run_attention(new, case, actual, gated)
        resolved_on = new._SX_OPT_QSA_RESOLVED_ROWS
        two_warp32_on = new._SX_OPT_QSA_TWO_WARP32
    torch.cuda.synchronize()
    # Engagement: the new launches really ran (a rejected gate would make the
    # bitwise comparison below pass vacuously).
    rows = case.q.shape[0]
    expect_resolved = resolved_on or (rows == 1 and case.page_size == 400)
    assert len(resolver.launches) == int(expect_resolved), resolver.launches
    assert len(partial.launches) == 1
    assert partial.launches[0]["RESOLVED_INDICES"] == expect_resolved
    expect_two_warps = rows <= (32 if two_warp32_on else 16)
    assert (partial.launches[0]["num_warps"] == 2) == expect_two_warps, (
        rows,
        partial.launches[0]["num_warps"],
    )
    assert bitwise_equal(expected, actual), (
        case.q.shape[0],
        case.page_size,
        new_flags,
        (expected.float() - actual.float()).abs().nan_to_num(1e9).max().item(),
    )


@pytest.fixture(autouse=True)
def _sm70():
    require_sm70()


def test_new_path_is_admitted():
    new = new_ops()
    case = build_decode_case(24, 8192, 784)
    assert new._use_sm70_qsa_two_warp_partial(24, 6, 256)
    assert new._use_sm70_qsa_two_warp_partial(32, 6, 256)
    assert not new._use_sm70_qsa_two_warp_partial(33, 6, 256)
    assert new._use_sm70_qsa_resolved_indices(case.q, case.k, case.indices, "auto")
    with patched(new, _SX_OPT_QSA_TWO_WARP32=False, _SX_OPT_QSA_RESOLVED_ROWS=False):
        assert not new._use_sm70_qsa_two_warp_partial(17, 6, 256)
        assert not new._use_sm70_qsa_resolved_indices(
            case.q, case.k, case.indices, "auto"
        )


@pytest.mark.parametrize("page_size", PAGE_SIZES)
@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_decode_attention_bitwise(rows, context, page_size):
    case = build_decode_case(rows, context, page_size, interleaved=True, seed=1)
    compare_once(case)  # both changes on (production default)
    compare_once(case, _SX_OPT_QSA_RESOLVED_ROWS=False)  # C1 alone
    compare_once(case, _SX_OPT_QSA_TWO_WARP32=False)  # MR4 alone
    compare_once(case, gated=False)


@pytest.mark.parametrize("rows", (1, 16, 24, 32))
def test_decode_attention_bitwise_separate_kv_and_padding(rows):
    case = build_decode_case(rows, 8192, 400, interleaved=False, seed=2)
    compare_once(case)
    if rows > 2:
        padded = build_decode_case(rows, 8192, 784, seed=3, padded_rows=rows // 3)
        compare_once(padded)


@pytest.mark.parametrize("rows", (8, 16, 17, 24, 32))
def test_decode_attention_graph_replay(rows):
    """Capture the new path once, then change inputs, poison output, replay."""
    old = old_ops()
    new = new_ops()
    case = build_decode_case(rows, 8192, 784, seed=4)
    out = torch.empty_like(case.q)
    graph = capture(lambda: run_attention(new, case, out))
    for scenario in range(5):
        fresh = build_decode_case(
            rows, 8192 if scenario % 2 else 2048, 784, seed=10 + scenario
        )
        # Graph inputs are fixed buffers: copy the new metadata in place. Pages
        # or requests that fall outside this cache are masked by the kernels
        # exactly as in the old path.
        case.q.copy_(fresh.q)
        case.gate.copy_(fresh.gate)
        case.table.copy_(fresh.table)
        case.indices.copy_(fresh.indices)
        case.token_to_req.copy_(fresh.token_to_req)
        case.positions.copy_(fresh.positions)
        case.seq_lens.copy_(fresh.seq_lens)
        del fresh
        if scenario == 3:
            case.table.copy_(case.table.roll(1, dims=-1))  # page relocation
        if scenario == 4:
            case.token_to_req.fill_(-1)
        out.fill_(float("nan"))
        graph.replay()
        expected = torch.full_like(out, float("nan"))
        run_attention(old, case, expected)
        torch.cuda.synchronize()
        assert bitwise_equal(expected, out), (rows, scenario)


def _bench_decode_rows(rows: int, context: int, layers: int) -> dict[str, float]:
    old = old_ops()
    new = new_ops()
    # One query/selection set per layer (12 QSA layers). The layers share one
    # KV cache to bound memory; a layer's selection (rows x 2051 x 1 KiB of
    # K+V) is far larger than L2, so sharing does not warm the next layer.
    base = build_decode_case(rows, context, 784, seed=100, corrupt=False)
    generator = torch.Generator().manual_seed(1234 + rows)
    cases = []
    for _ in range(layers):
        indices = torch.stack(
            [
                sample_row_selection(generator, int(position) + 1)
                for position in base.positions.tolist()
            ]
        ).to(base.indices.device)
        cases.append(
            dataclasses.replace(
                base,
                q=torch.randn_like(base.q),
                gate=torch.randn_like(base.gate),
                indices=indices,
            )
        )
    outs = [torch.empty_like(case.q) for case in cases]

    def make(ops, **flags):
        def run():
            context_manager = (
                patched(new, **flags) if ops is new else contextlib.nullcontext()
            )
            with context_manager:
                for case, out in zip(cases, outs):
                    run_attention(ops, case, out)

        return run

    graphs = {
        "old": capture(make(old)),
        "new": capture(make(new)),
        "new_c1_only": capture(make(new, _SX_OPT_QSA_RESOLVED_ROWS=False)),
        "new_mr4_only": capture(make(new, _SX_OPT_QSA_TWO_WARP32=False)),
    }
    timings = paired_graph_ms(graphs)
    timings["delta_ms"] = timings["new"] - timings["old"]
    report(f"decode attention x{layers} layers, M={rows}, ctx={context}", timings)
    return timings


def bench_decode(rows_list=(1, 8, 16, 17, 24, 32), context=8192, layers=12):
    results = {}
    for rows in rows_list:
        results[rows] = _bench_decode_rows(rows, context, layers)
        torch.cuda.empty_cache()
    return results


@pytest.mark.parametrize("context", (8192, 32768))
def test_decode_attention_microbench(context):
    bench_decode(context=context)


def main():
    if not is_sm70():
        raise SystemExit("requires a V100 / SM70 GPU")
    for rows in (1, 2, 16, 17, 24, 32):
        for page in PAGE_SIZES:
            compare_once(build_decode_case(rows, 8192, page, seed=1))
    print("[sx-qsa] decode attention bitwise: OK", flush=True)
    bench_decode(context=8192)
    bench_decode(context=32768)


if __name__ == "__main__":
    main()
