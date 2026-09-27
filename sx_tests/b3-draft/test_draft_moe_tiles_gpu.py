# SPDX-License-Identifier: Apache-2.0
"""MTP-K1 / MTP-7: draft MoE tile table vs the previous default config.

GPU: ONE V100 (SM70). Needs the changed fused_moe.py (+ configs JSON)
installed into / bind-mounted over the imported vllm:

  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-draft/test_draft_moe_tiles_gpu.py

Environment knobs:
  SX_TEST_QUICK=1              fewer widths / routings
  SX_TEST_NO_BENCH=1           skip the microbenchmark
  SX_TEST_BENCH_ASSERT=1       fail if the table is >5% slower than the
                               previous default at any width
  SX_TEST_ROUTINGS=N           routings per width for the output check (8)
  SX_TEST_MTP_WEIGHTS_DIR=dir  use the checkpoint's real MTP experts (TP rank
                               SX_TEST_MTP_TP_RANK, default 0) instead of
                               random N(0, 0.02) weights
  SX_OPT_MTP_DRAFT_TILES_TABLE / _FILE   test a candidate table

Widths: every draft width M = B (draft decode) and M = B*(k+1) (draft
prefill) for k in {1,2,3,4}, B in {1,2,4,8,12,16,24}, plus split-graph /
table-boundary widths (7, 9, 13, 15, 18, 30, 33, 45, 128, 129, 136, 160).

Asserts
  * the armed table selects its entry (and 1Cat's tile at M1/M5), the
    previous default selects 1Cat's tile at M1/M5 and the 0.0.3 tile
    elsewhere;
  * draft MoE output with the table is finite and close to the previous
    default (draft-only change, tolerance), and its error against an FP32
    oracle is not worse than the previous default's (x1.5 + eps); the number
    of bitwise-identical outputs is reported (BK=64 keeps the K order, so 1Cat
    saw identical bits at M1/M5); where the table selects the same config as
    before (M1/M5 by default, M > 128) the outputs must be bitwise equal;
  * CUDA-graph capture at every width works and replay with new inputs is
    bitwise equal to eager.
Reports a CUDA-graph microbenchmark per width (8 rotating routings per graph,
cold weights): previous default, 0.0.3 tile, 1Cat tile, SX table.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_draft_common as C  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")

ONECAT = {
    "BLOCK_SIZE_M": 2,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}
_SUMMARY: dict[int, dict[str, object]] = {}


def _routings() -> int:
    default = "2" if C.quick() else "8"
    return max(1, int(os.environ.get("SX_TEST_ROUTINGS", default)))


@pytest.fixture(scope="module")
def experts():
    torch.manual_seed(0)
    w1, w2 = C.experts_for_test()
    yield w1, w2
    del w1, w2
    torch.cuda.empty_cache()


def _expected_table_tile(m: int) -> dict[str, int] | None:
    mod = C.fm()
    with C.table_ctx():
        return mod._sx_mtp_draft_tile(m)


@pytest.mark.parametrize("m", C.widths())
def test_config_selection(m: int) -> None:
    with C.previous_default_ctx():
        prev = C.selected_config(m)
    if m in (1, 5):
        assert prev == ONECAT
    else:
        assert (prev["BLOCK_SIZE_M"], prev["BLOCK_SIZE_N"], prev["BLOCK_SIZE_K"]) == (
            16,
            32,
            64,
        ), prev
    with C.table_ctx():
        table = C.selected_config(m)
    want = _expected_table_tile(m)
    assert table == (want if want is not None else prev), (m, table, want)
    if m in (1, 5) and not os.environ.get("SX_OPT_MTP_DRAFT_TILES_TABLE") and not (
        os.environ.get("SX_OPT_MTP_DRAFT_TILES_FILE")
    ):
        assert table == ONECAT  # C1 k=4 draft keeps 1Cat's tile


@pytest.mark.parametrize("m", C.widths())
@torch.inference_mode()
def test_outputs_vs_previous_default(experts, m: int) -> None:
    w1, w2 = experts
    bitwise = 0
    worst = 0.0
    worst_ref_table = 0.0
    worst_ref_prev = 0.0
    n = _routings()
    with C.previous_default_ctx():
        prev_cfg = C.selected_config(m)
    with C.table_ctx():
        table_cfg = C.selected_config(m)
    # Same config (M1/M5 with the default table, widths the table does not
    # cover) -> the claim is "unchanged", i.e. bitwise.
    same_config = prev_cfg == table_cfg
    for r in range(n):
        # Odd routings: correlated rows within one request's k+1 = 5 rows.
        overlap, group = (0.5, 5) if r % 2 else (0.0, 1)
        weights, ids = C.make_routing(m, 1000 * m + r, overlap, group)
        hidden = C.make_hidden(m, 77 * m + r, scale=1.0 + 0.5 * (r % 3))
        with C.previous_default_ctx():
            prev = C.run_moe(hidden, w1, w2, weights, ids)
        with C.table_ctx():
            table = C.run_moe(hidden, w1, w2, weights, ids)
        torch.cuda.synchronize()
        assert torch.isfinite(table).all(), (m, r)
        ref = C.reference_fp32(hidden, w1, w2, weights, ids)
        scale = max(1.0, float(ref.abs().max()))
        torch.testing.assert_close(
            table.float(), prev.float(), atol=2e-3 * scale, rtol=2e-2
        )
        err_table = C.max_abs(table, ref)
        err_prev = C.max_abs(prev, ref)
        assert err_table <= 1.5 * err_prev + 1e-3 * scale, (m, r, err_table, err_prev)
        equal = C.bit_equal16(table, prev)
        if same_config:
            assert equal, (m, r, C.tile_str(table_cfg))
        bitwise += int(equal)
        worst = max(worst, C.max_abs(table, prev))
        worst_ref_table = max(worst_ref_table, err_table)
        worst_ref_prev = max(worst_ref_prev, err_prev)
    _SUMMARY[m] = {
        "bitwise": f"{bitwise}/{n}" + (" (same tile)" if same_config else ""),
        "max|table-prev|": f"{worst:.3g}",
        "err_table": f"{worst_ref_table:.3g}",
        "err_prev": f"{worst_ref_prev:.3g}",
        "tile": C.tile_str(table_cfg),
    }


@pytest.mark.parametrize("m", C.widths())
@torch.inference_mode()
def test_graph_replay_matches_eager(experts, m: int) -> None:
    w1, w2 = experts
    weights, ids = C.make_routing(m, 5 * m)
    hidden = C.make_hidden(m, 11 * m)
    static = [hidden.clone(), weights.clone(), ids.clone()]
    holder: dict[str, torch.Tensor] = {}

    def fn() -> None:
        holder["out"] = C.run_moe(static[0], w1, w2, static[1], static[2])

    with C.table_ctx():
        graph = C.capture(fn)
        for it in range(3):
            new_w, new_ids = C.make_routing(m, 5 * m + 1 + it, 0.5 * (it % 2), 5)
            new_h = C.make_hidden(m, 11 * m + 1 + it)
            static[0].copy_(new_h)
            static[1].copy_(new_w)
            static[2].copy_(new_ids)
            graph.replay()
            torch.cuda.synchronize()
            eager = C.run_moe(new_h, w1, w2, new_w, new_ids)
            torch.cuda.synchronize()
            assert C.bit_equal16(holder["out"], eager), (m, it)
    del graph
    torch.cuda.synchronize()


def test_summary() -> None:
    if not _SUMMARY:
        pytest.skip("no output results collected")
    rows = [
        [m, *(_SUMMARY[m][k] for k in ("tile", "bitwise", "max|table-prev|", "err_table", "err_prev"))]
        for m in sorted(_SUMMARY)
    ]
    C.print_table(
        "SX draft MoE table vs previous default (draft-only; tolerance asserted)",
        ["M", "table tile", "bitwise", "max|table-prev|", "err_table(fp32)", "err_prev(fp32)"],
        rows,
    )


@pytest.mark.skipif(C.no_bench(), reason="SX_TEST_NO_BENCH=1")
@torch.inference_mode()
def test_microbench_per_width(experts) -> None:
    copies = [experts]
    extra = int(os.environ.get("SX_TEST_WEIGHT_COPIES", "1")) - 1
    for i in range(max(0, extra)):
        copies.append(C.make_experts(4242 + i))
    rows = []
    slower = []
    for m in C.widths():
        bench = C.MoeBench(m, copies, routings=8, seed=m)
        graphs = {
            "prev": bench.graph(C.previous_default_ctx),
            "legacy": bench.graph(C.legacy_ctx),
            "1cat": bench.graph(lambda: C.tile_ctx(ONECAT)),
            "table": bench.graph(C.table_ctx),
        }
        us = C.bench_alternating(graphs, bench.calls)
        del graphs
        torch.cuda.synchronize()
        with C.table_ctx():
            cfg = C.selected_config(m)
        rows.append(
            [
                m,
                f"{C.expected_distinct_experts(m):.0f}",
                f"{us['prev']:.1f}",
                f"{us['legacy']:.1f}",
                f"{us['1cat']:.1f}",
                f"{us['table']:.1f}",
                f"{us['prev'] / us['table']:.2f}x",
                C.tile_str(cfg),
            ]
        )
        if us["table"] > 1.05 * us["prev"]:
            slower.append((m, us["prev"], us["table"]))
    C.print_table(
        "Draft MoE fused_experts per call, CUDA graph, 8 rotating routings (us)",
        ["M", "D(10M)", "prev", "0.0.3", "1Cat", "table", "prev/table", "table tile"],
        rows,
    )
    if slower:
        print(f"\nWidths where the table is >5% slower than the previous default: {slower}")
    if os.environ.get("SX_TEST_BENCH_ASSERT", "0") not in ("", "0"):
        assert not slower, slower


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
