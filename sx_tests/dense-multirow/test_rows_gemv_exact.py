# SPDX-License-Identifier: Apache-2.0
"""dense-multirow MR1 + MR2: exact multi-row FP16 row GEMV and fused GDN input.

GPU: ONE V100 (SM70).  Needs the overlay installed into the imported vllm.

  /opt/venv/bin/python -m pytest -q -s sx_tests/dense-multirow/test_rows_gemv_exact.py
  /opt/venv/bin/python sx_tests/dense-multirow/test_rows_gemv_exact.py
  (SX_TEST_QUICK=1 shrinks the M ladder to 1,2,4,8,16,24; SX_TEST_NO_BENCH=1
  skips the microbenchmark.)

Asserts (bitwise = torch.equal on the int16 view, so NaN/Inf/-0 count):
  * every row of the multi-row kernel == the UNCHANGED M=1 production route
    run on that row alone (_qwen38_sm70_fp16_gemv / _qwen38_sm70_fp16_gdn_input
    with a fresh (1, K) input) for every _ROLE_PLANS role (router 512x2560,
    QSA qkv 3584x2560, QSA o 2560x1536, index 640x2560, GDN out 2560x1536,
    GDN qkvz 4096x2560, GDN ba 24x2560, HC down 336x10240) and the fused
    GDN input (qkvz 4096x2560 + ba 24x2560), M in {1,2,3,4,8,16,17,24,32},
    both reduction forms (SX_OPT_ROWS_FUSED_REDUCE 1/0), default and
    alternate/masked token tiles, activations randn*{0.25,1,3} + special
    rows (+-0, subnormals, +-65504, tiny).  Mismatches are collected per
    reduce form and reported together (a failing SX_OPT_ROWS_FUSED_REDUCE=1
    does not hide whether =0 is clean);
  * stale NaN/+-Inf padded rows (FULL-graph padding) never change the bits
    of the other rows, for every role, tile (incl. masked tails) and form;
  * CUDA-graph replay with changing inputs and NaN-poisoned outputs == eager;
  * op dispatch: FULL decode-graph context -> multi-row kernel; SX_OPT_ROWS=0,
    no decode context, or M above the role table -> old F.linear bit-for-bit.
Reports max|rows - cuBLAS| (the old M>1 path) per role/M, and a CUDA-graph
microbenchmark (32 calls per graph, rotating >=48 MB of distinct weights,
median of 64 replays, A/B alternated) of cuBLAS vs rows at M in
{2,4,8,16,24}, plus a suggested SX_OPT_ROWS_TABLE.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sx_rows_common as C  # noqa: E402

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")


def _tiles_for(m: int) -> list[int]:
    gemv = C.gemv_module()
    tiles = {gemv._sx_split_rows(m, 8)}
    if not C.quick():
        tiles.add(gemv._sx_split_rows(m, 4))
        if m > 8:
            tiles.add(8)  # masked tail when 8 does not divide m
        if m >= 4:
            tiles.add(3)
    return sorted(tiles)


def _form(fused: bool) -> str:
    return f"SX_OPT_ROWS_FUSED_REDUCE={int(fused)}"


def _check_gemv_role(
    name: str, prefix: str, shape: tuple[int, int], mm: C.Mismatches
) -> list[list]:
    gemv = C.gemv_module()
    n, k = shape
    plan = gemv._plan_for(prefix, shape)
    assert plan is not None, name
    weight = C.make_weight(n, k, 0.05, seed=1000 + n + k)
    report = []
    for m in C.m_values():
        worst = 0.0
        for case_name, x in C.input_cases(m, k, seed=17 * m + n):
            # The oracle must be the unchanged M=1 Triton kernel, not the
            # F.linear fallback (e.g. VLLM_BATCH_INVARIANT set in the shell).
            assert gemv._runtime_ok(x[0:1].clone(), weight), (
                f"{name}: the M=1 oracle would not take the M=1 kernel"
            )
            oracle = torch.cat(
                [
                    gemv._qwen38_sm70_fp16_gemv(x[r : r + 1].clone(), weight, prefix)
                    for r in range(m)
                ]
            )
            for tile in _tiles_for(m):
                for fused in (True, False):
                    out = gemv._sx_rows_gemv(x, weight, plan, tile, fused)
                    torch.cuda.synchronize()
                    for r in range(m):
                        mm.check(
                            _form(fused),
                            C.bit_equal(out[r], oracle[r]),
                            f"{name} M={m} tile={tile} case={case_name} row={r}",
                        )
            worst = max(worst, C.max_abs_finite(oracle, F.linear(x, weight)))
        report.append([name, m, f"{worst:.3e}"])
    return report


def _old_gdn_input(x, qkvz_weight, ba_weight):
    """The pre-overlay M>1 path of _qwen38_sm70_fp16_gdn_input."""
    gemv = C.gemv_module()
    qkvz = F.linear(x, qkvz_weight)
    ba = F.linear(x, ba_weight)
    if gemv._can_fuse_gdn_projection_split(qkvz, ba):
        return gemv._split_gdn_projection_outputs(qkvz, ba)
    return (
        qkvz[..., :2560].contiguous(),
        qkvz[..., 2560:].contiguous(),
        ba[..., :12].contiguous(),
        ba[..., 12:].contiguous(),
    )


def _gdn_weights(seed: int = 4242):
    return (
        C.make_weight(4096, 2560, 0.05, seed),
        C.make_weight(24, 2560, 0.05, seed + 1),
    )


def _check_gdn_input(mm: C.Mismatches) -> list[list]:
    gemv = C.gemv_module()
    qkvz_weight, ba_weight = _gdn_weights()
    report = []
    for m in C.m_values():
        worst = 0.0
        for case_name, x in C.input_cases(m, 2560, seed=31 * m):
            assert gemv._runtime_ok(x[0:1].clone(), qkvz_weight)
            assert gemv._runtime_ok(x[0:1].clone(), ba_weight)
            oracle = [
                gemv._qwen38_sm70_fp16_gdn_input(
                    x[r : r + 1].clone(), qkvz_weight, ba_weight
                )
                for r in range(m)
            ]
            for tile in _tiles_for(m):
                for fused in (True, False):
                    outs = gemv._sx_rows_gdn_input(
                        x, qkvz_weight, ba_weight, tile, fused
                    )
                    torch.cuda.synchronize()
                    for r in range(m):
                        for part, out, ref in zip(
                            ("qkv", "z", "b", "a"), outs, oracle[r]
                        ):
                            mm.check(
                                _form(fused),
                                C.bit_equal(out[r], ref[0]),
                                f"gdn_in.{part} M={m} tile={tile} "
                                f"case={case_name} row={r}",
                            )
            old = _old_gdn_input(x, qkvz_weight, ba_weight)
            for part_index in range(4):
                ref = torch.cat([o[part_index] for o in oracle])
                worst = max(worst, C.max_abs_finite(ref, old[part_index]))
        report.append(["gdn_in(fused)", m, f"{worst:.3e}"])
    return report


def test_rows_gemv_bitwise_per_row():
    mm = C.Mismatches()
    report = []
    for name, prefix, shape, _, _ in C.GEMV_ROLES:
        report += _check_gemv_role(name, prefix, shape, mm)
    C.print_table(
        "row GEMV: rows == M=1 bitwise (asserted); max|M1-route - cuBLAS|",
        ["role", "M", "max_abs_vs_cublas"],
        report,
    )
    mm.assert_clean("row GEMV rows vs M=1")


def test_rows_gdn_input_bitwise_per_row():
    mm = C.Mismatches()
    report = _check_gdn_input(mm)
    C.print_table(
        "fused GDN input: rows == M=1 bitwise (asserted); max|M1 - old cuBLAS path|",
        ["role", "M", "max_abs_vs_cublas"],
        report,
    )
    mm.assert_clean("fused GDN input rows vs M=1")


def test_rows_poisoned_padding_rows_isolated():
    """Stale NaN/Inf padded rows never change any other row's bits."""
    gemv = C.gemv_module()
    mm = C.Mismatches()
    for name, prefix, shape, _, _ in C.GEMV_ROLES:
        n, k = shape
        plan = gemv._plan_for(prefix, shape)
        weight = C.make_weight(n, k, 0.05, seed=2000 + n + k)
        for m in (2, 3, 8, 17, 24):
            x = C.make_rows(m, k, 1.0, seed=900 + m, special="none")
            poisoned = set(range(1, m, 2))
            C.poison_rows(x, sorted(poisoned))
            clean = [r for r in range(m) if r not in poisoned]
            oracle = {
                r: gemv._qwen38_sm70_fp16_gemv(x[r : r + 1].clone(), weight, prefix)[0]
                for r in clean
            }
            for tile in sorted({*_tiles_for(m), gemv._sx_split_rows(m, 2)}):
                for fused in (True, False):
                    out = gemv._sx_rows_gemv(x, weight, plan, tile, fused)
                    torch.cuda.synchronize()
                    for r in clean:
                        mm.check(
                            _form(fused),
                            C.bit_equal(out[r], oracle[r]),
                            f"{name} M={m} tile={tile} clean row={r}",
                        )
    qkvz_weight, ba_weight = _gdn_weights(seed=5151)
    for m in (2, 3, 8):
        x = C.make_rows(m, 2560, 1.0, seed=950 + m, special="none")
        C.poison_rows(x, range(1, m, 2))
        clean = list(range(0, m, 2))
        oracle = {
            r: gemv._qwen38_sm70_fp16_gdn_input(
                x[r : r + 1].clone(), qkvz_weight, ba_weight
            )
            for r in clean
        }
        for tile in sorted({*_tiles_for(m), gemv._sx_split_rows(m, 2)}):
            for fused in (True, False):
                outs = gemv._sx_rows_gdn_input(x, qkvz_weight, ba_weight, tile, fused)
                torch.cuda.synchronize()
                for r in clean:
                    for part, out, ref in zip(("qkv", "z", "b", "a"), outs, oracle[r]):
                        mm.check(
                            _form(fused),
                            C.bit_equal(out[r], ref[0]),
                            f"gdn_in.{part} M={m} tile={tile} clean row={r}",
                        )
    mm.assert_clean("poisoned padding rows (GEMV + GDN input)")


def test_rows_graph_replay_changing_inputs():
    gemv = C.gemv_module()
    torch.manual_seed(7)
    for name, prefix, shape, _, _ in C.GEMV_ROLES[:5]:
        n, k = shape
        plan = gemv._plan_for(prefix, shape)
        weight = C.make_weight(n, k, 0.05, seed=5 + n)
        for m in (2, 4, 8, 24):
            tile = gemv._sx_split_rows(m, 8)
            x = C.make_rows(m, k, 1.0, seed=m)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                gemv._sx_rows_gemv(x, weight, plan, tile, True)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = gemv._sx_rows_gemv(x, weight, plan, tile, True)
            for step in range(16):
                x.copy_(C.make_rows(m, k, 0.25 + step % 3, seed=100 + step))
                out.fill_(float("nan"))
                graph.replay()
                eager = gemv._sx_rows_gemv(x, weight, plan, tile, True)
                torch.cuda.synchronize()
                assert C.bit_equal(out, eager), f"{name} M={m} replay {step}"
    qkvz_weight, ba_weight = _gdn_weights()
    for m in (2, 4, 8):
        x = C.make_rows(m, 2560, 1.0, seed=m)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gemv._sx_rows_gdn_input(x, qkvz_weight, ba_weight, m, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = gemv._sx_rows_gdn_input(x, qkvz_weight, ba_weight, m, True)
        for step in range(16):
            x.copy_(C.make_rows(m, 2560, 0.25 + step % 3, seed=200 + step))
            for o in outs:
                o.fill_(float("nan"))
            graph.replay()
            eager = gemv._sx_rows_gdn_input(x, qkvz_weight, ba_weight, m, True)
            torch.cuda.synchronize()
            for o, e in zip(outs, eager):
                assert C.bit_equal(o, e), f"gdn_in M={m} replay {step}"


def test_rows_op_dispatch():
    gemv = C.gemv_module()
    prefix = "model.layers.0.linear_attn.out_proj"
    shape = (2560, 1536)
    plan = gemv._plan_for(prefix, shape)
    weight = C.make_weight(*shape, 0.05, seed=3)
    op = torch.ops.vllm.qwen38_sm70_fp16_gemv
    x4 = C.make_rows(4, 1536, 1.0, seed=4)
    x16 = C.make_rows(16, 1536, 1.0, seed=16)
    x1 = C.make_rows(1, 1536, 1.0, seed=1)
    with C.decode_graph_ctx():
        got = op(x4, weight, prefix)
        assert C.bit_equal(got, gemv._sx_rows_gemv(x4, weight, plan, 4))
        # Default table: gdn_out admits M <= 8 only.
        assert torch.equal(op(x16, weight, prefix), F.linear(x16, weight))
        # M=1 keeps the unchanged M=1 kernel.
        m1 = op(x1, weight, prefix)
    with C.decode_graph_ctx(SX_OPT_ROWS="0"):
        assert torch.equal(op(x4, weight, prefix), F.linear(x4, weight))
    with C.decode_graph_ctx(SX_OPT_ROWS_TABLE="gdn_out=16"):
        assert C.bit_equal(
            op(x16, weight, prefix), gemv._sx_rows_gemv(x16, weight, plan, 8)
        )
    with C.decode_graph_ctx(SX_OPT_ROWS_MAX_M="2"):
        assert torch.equal(op(x4, weight, prefix), F.linear(x4, weight))
    # Outside the FULL decode-graph capture (eager / mixed / legacy lane).
    with C.sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="0"):
        assert torch.equal(op(x4, weight, prefix), F.linear(x4, weight))
        assert C.bit_equal(op(x1, weight, prefix), m1)
    with C.sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="1"):
        assert torch.equal(op(x4, weight, prefix), F.linear(x4, weight))
    # Role-less legacy calls never take the rows route.
    with C.decode_graph_ctx():
        assert torch.equal(op(x4, weight, ""), F.linear(x4, weight))

    router = "model.layers.3.mlp.gate"
    router_w = C.make_weight(512, 2560, 0.05, seed=9)
    router_plan = gemv._plan_for(router, (512, 2560))
    x24 = C.make_rows(24, 2560, 1.0, seed=24)
    with C.decode_graph_ctx():
        assert C.bit_equal(
            op(x24, router_w, router),
            gemv._sx_rows_gemv(x24, router_w, router_plan, 8),
        )

    qkvz_weight, ba_weight = _gdn_weights()
    gdn_op = torch.ops.vllm.qwen38_sm70_fp16_gdn_input
    x = C.make_rows(4, 2560, 1.0, seed=44)
    with C.decode_graph_ctx():
        got = gdn_op(x, qkvz_weight, ba_weight)
        want = gemv._sx_rows_gdn_input(x, qkvz_weight, ba_weight, 4)
        assert all(C.bit_equal(a, b) for a, b in zip(got, want))
    with C.decode_graph_ctx(SX_OPT_ROWS="0"):
        got = gdn_op(x, qkvz_weight, ba_weight)
        want = _old_gdn_input(x, qkvz_weight, ba_weight)
        assert all(torch.equal(a, b) for a, b in zip(got, want))


def run_benchmark() -> None:
    gemv = C.gemv_module()
    rows_out = []
    suggestions = {}
    for name, prefix, shape, calls_per_step, bench in C.GEMV_ROLES:
        if not bench:
            continue
        n, k = shape
        plan = gemv._plan_for(prefix, shape)
        copies = C.n_copies(n * k * 2)
        weights = [C.make_weight(n, k, 0.05, seed=50 + i) for i in range(copies)]
        per_m = {}
        for m in C.BENCH_M:
            x = C.make_rows(m, k, 1.0, seed=m, special="none")
            tile = gemv._sx_split_rows(m, 8)
            variants = {
                "cublas": lambda i: F.linear(x, weights[i % copies]),
                "rows_fused": lambda i, t=tile: gemv._sx_rows_gemv(
                    x, weights[i % copies], plan, t, True
                ),
                "rows_plain": lambda i, t=tile: gemv._sx_rows_gemv(
                    x, weights[i % copies], plan, t, False
                ),
            }
            if m >= 8:
                variants["rows_fused_t4"] = lambda i: gemv._sx_rows_gemv(
                    x, weights[i % copies], plan, 4, True
                )
            if m <= 8:
                # Reference: M separate M=1 launches (reads the weights M times).
                variants["m1_loop"] = lambda i: [
                    gemv._qwen38_sm70_fp16_gemv(x[r : r + 1], weights[i % copies], prefix)
                    for r in range(m)
                ]
            t = C.bench_graphs(variants, calls=32)
            best_rows = min(v for key, v in t.items() if key.startswith("rows"))
            per_m[m] = (t["cublas"], best_rows)
            rows_out.append(
                [
                    name,
                    m,
                    f"{t['cublas']:.2f}",
                    f"{t['rows_fused']:.2f}",
                    f"{t['rows_plain']:.2f}",
                    f"{t.get('rows_fused_t4', float('nan')):.2f}",
                    f"{t.get('m1_loop', float('nan')):.2f}",
                    f"{t['cublas'] / best_rows:.2f}x",
                    f"{(t['cublas'] - best_rows) * calls_per_step / 1000:.3f}",
                ]
            )
        suggestions[name] = C.admitted_max_m(per_m)
    fused_gdn = _bench_gdn_input()
    rows_out += fused_gdn[0]
    suggestions["gdn_in"] = fused_gdn[1]
    C.print_table(
        "CUDA-graph microbenchmark, us/call (32 calls/graph, rotating weights)",
        [
            "role",
            "M",
            "cublas",
            "rows_fused",
            "rows_plain",
            "rows_t4",
            "m1_loop",
            "speedup",
            "ms/step_saved",
        ],
        rows_out,
    )
    table = ",".join(f"{key}={value}" for key, value in suggestions.items())
    print(
        "\nSuggested (rows <= 0.97 x cuBLAS at every M up to max): "
        f"SX_OPT_ROWS_TABLE={table}  (hc from test_rows_hc_exact.py)"
    )


def _bench_gdn_input():
    gemv = C.gemv_module()
    copies = C.n_copies((4096 + 24) * 2560 * 2)
    weights = [_gdn_weights(seed=77 + 2 * i) for i in range(copies)]
    out = []
    per_m = {}
    for m in C.BENCH_M:
        x = C.make_rows(m, 2560, 1.0, seed=m, special="none")
        tile = gemv._sx_split_rows(m, 8)
        variants = {
            "cublas": lambda i: _old_gdn_input(x, *weights[i % copies]),
            "rows_fused": lambda i, t=tile: gemv._sx_rows_gdn_input(
                x, *weights[i % copies], t, True
            ),
            "rows_plain": lambda i, t=tile: gemv._sx_rows_gdn_input(
                x, *weights[i % copies], t, False
            ),
        }
        if m >= 8:
            variants["rows_fused_t4"] = lambda i: gemv._sx_rows_gdn_input(
                x, *weights[i % copies], 4, True
            )
        t = C.bench_graphs(variants, calls=32)
        best_rows = min(v for key, v in t.items() if key.startswith("rows"))
        per_m[m] = (t["cublas"], best_rows)
        out.append(
            [
                "gdn_in(fused)",
                m,
                f"{t['cublas']:.2f}",
                f"{t['rows_fused']:.2f}",
                f"{t['rows_plain']:.2f}",
                f"{t.get('rows_fused_t4', float('nan')):.2f}",
                "-",
                f"{t['cublas'] / best_rows:.2f}x",
                f"{(t['cublas'] - best_rows) * 36 / 1000:.3f}",
            ]
        )
    return out, C.admitted_max_m(per_m)


@pytest.mark.skipif(
    os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0"), reason="bench off"
)
def test_zz_microbenchmark(capsys):
    with capsys.disabled():
        run_benchmark()


def main() -> None:
    if not C.sm70_available():
        raise SystemExit("requires an SM70 GPU")
    test_rows_gemv_bitwise_per_row()
    test_rows_gdn_input_bitwise_per_row()
    test_rows_poisoned_padding_rows_isolated()
    test_rows_graph_replay_changing_inputs()
    test_rows_op_dispatch()
    print("\nPASS: all multi-row rows are bitwise equal to the M=1 route.")
    if os.environ.get("SX_TEST_NO_BENCH", "0") in ("", "0"):
        run_benchmark()


if __name__ == "__main__":
    main()
