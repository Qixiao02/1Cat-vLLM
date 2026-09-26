# SPDX-License-Identifier: Apache-2.0
"""dense-multirow MR3 + MR9a: exact multi-row fused FP16 HC mix and combine-norm.

GPU: ONE V100 (SM70).  Needs the overlay installed into the imported vllm.
(The 4-GPU comparison against the production TP4-sharded M=1 route is
test_rows_hc_tp4.py.)

  /opt/venv/bin/python -m pytest -q -s sx_tests/dense-multirow/test_rows_hc_exact.py
  /opt/venv/bin/python sx_tests/dense-multirow/test_rows_hc_exact.py
  (SX_TEST_QUICK=1: shorter M ladder; SX_TEST_NO_BENCH=1: no microbenchmark.)

Asserts (bitwise = int16-view torch.equal):
  * HC down 336x10240 rows kernel: lora (M, 320) and injection (M, 4) rows ==
    the M=1 _qwen38_hc_down_silu_inject_kernel on that row, for weight-row
    tiles NW in {1,2,4}, token tiles 1..4 (incl. masked tails), both
    reduction forms;
  * HC up 10240x320 row-4 rows kernel: block rows == the M=1
    _qwen38_hc_up_gate_mix_row4_kernel on the same lora row, token tiles 1..8;
  * full chain _sx_hc_rows_forward == the unchanged M=1 replicated route
    (_qwen38_sm70_fp16_fused_hc on a (1, 10240) input) per row, block and
    injection, M in {1,2,3,4,8,16,17,24,32}, activations randn*{0.25,1,3}
    + special rows (mismatches collected per reduce form, reported together);
  * stale NaN/+-Inf padded rows never change the other rows (tiles incl.
    masked tails, both reduce forms);
  * op dispatch (decode-graph context -> rows; SX_OPT_ROWS=0 / M above the
    table / no decode context -> the old cuBLAS chain bit-for-bit);
  * graph replay with changing inputs and poisoned outputs;
  * MR9a: inside the admitted HC width, hc_combine_norm rows at N rows ==
    the N=1 call per row (out and y); informational count of rows where the
    old 512 tile differs from the M=1 1024 tile.
Reports max|rows - old cuBLAS chain| and a CUDA-graph microbenchmark (32 HC
calls per graph over >=48 MB of rotating weight pairs, median of 64 replays)
of the cuBLAS chain vs rows plans at M in {2,4,8,16,24}, and 512 vs 1024
combine-norm tiles.
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

K_HC = 10240
RANK = 320
HC_DIM = 2560


def _hc_weights(seed: int):
    down = C.make_weight(336, K_HC, 0.02, seed)
    down[324:].zero_()  # production pads rows 324..335 with zeros
    up = C.make_weight(K_HC, RANK, 0.05, seed + 1)
    return down.contiguous(), up.contiguous()


def _old_chain(x, down, up):
    """The pre-overlay M>1 path of _qwen38_sm70_fp16_fused_hc."""
    dai = F.linear(x, down)
    lora = torch.ops.vllm.qwen4_exp_hc_silu(dai[..., :RANK], 4)
    injection = dai[..., RANK : RANK + 4]
    gate = F.linear(lora, up)
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, 4), injection


def _m1_down(x_row, down):
    hc = C.hc_module()
    lora = x_row.new_empty((1, RANK))
    injection = x_row.new_empty((1, 4))
    hc._qwen38_hc_down_silu_inject_kernel[(RANK + 4,)](
        x_row, down, lora, injection,
        K=K_HC, BLOCK_K=256, RANK_VALUE=RANK, HC_COUNT=4, num_warps=4,
    )  # fmt: skip
    return lora, injection


def _m1_up(lora_row, up, x_row):
    hc = C.hc_module()
    block = x_row.new_empty((1, HC_DIM))
    hc._qwen38_hc_up_gate_mix_row4_kernel[(HC_DIM // 4,)](
        lora_row, up, x_row, block,
        K=RANK, HC_DIMENSION=HC_DIM, HC_COUNT=4, BLOCK_N=4, BLOCK_K=512,
        num_warps=8,
    )  # fmt: skip
    return block


def _rows_down(x, down, rows, nw, fused):
    hc = C.hc_module()
    m = x.shape[0]
    lora = x.new_empty((m, RANK))
    injection = x.new_empty((m, 4))
    hc._qwen38_hc_down_silu_inject_rows_kernel[(-(-m // rows), (RANK + 4) // nw)](
        x, down, lora, injection, m,
        K=K_HC, BLOCK_K=256, RANK_VALUE=RANK, HC_COUNT=4, NW=nw, ROWS=rows,
        MASK_ROWS=m % rows != 0, FUSED_REDUCE=fused, num_warps=4,
    )  # fmt: skip
    return lora, injection


def _rows_up(lora, up, x, rows, fused):
    hc = C.hc_module()
    m = x.shape[0]
    block = x.new_empty((m, HC_DIM))
    hc._qwen38_hc_up_gate_mix_row4_rows_kernel[(-(-m // rows), HC_DIM // 4)](
        lora, up, x, block, m,
        K=RANK, HC_DIMENSION=HC_DIM, HC_COUNT=4, BLOCK_N=4, BLOCK_K=512,
        ROWS=rows, MASK_ROWS=m % rows != 0, FUSED_REDUCE=fused, num_warps=8,
    )  # fmt: skip
    return block


def _plans_for(m: int):
    hc = C.hc_module()
    gemv = C.gemv_module()
    split = gemv._sx_split_rows
    plans = {
        hc._SxHcRowsPlan(split(m, 4), 4, split(m, 8), True),
        hc._SxHcRowsPlan(split(m, 4), 4, split(m, 8), False),
    }
    if not C.quick():
        plans |= {
            hc._SxHcRowsPlan(split(m, 4), 1, split(m, 4), True),
            hc._SxHcRowsPlan(split(m, 2), 2, split(m, 2), True),
            hc._SxHcRowsPlan(min(m, 3), 4, min(m, 3), False),
        }
    return sorted(plans)


def _form(fused: bool) -> str:
    return f"SX_OPT_ROWS_FUSED_REDUCE={int(fused)}"


def test_hc_down_up_kernels_bitwise():
    down, up = _hc_weights(11)
    mm = C.Mismatches()
    for m in C.m_values():
        for case_name, x in C.input_cases(m, K_HC, seed=13 * m):
            m1 = [_m1_down(x[r : r + 1].clone(), down) for r in range(m)]
            lora_ref = torch.cat([lr for lr, _ in m1])
            block_ref = [
                _m1_up(lora_ref[r : r + 1].clone(), up, x[r : r + 1].clone())
                for r in range(m)
            ]
            for plan in _plans_for(m):
                lora, injection = _rows_down(
                    x, down, plan.down_rows, plan.down_nw, plan.fused_reduce
                )
                block = _rows_up(lora_ref, up, x, plan.up_rows, plan.fused_reduce)
                torch.cuda.synchronize()
                key = _form(plan.fused_reduce)
                for r in range(m):
                    ctx = f"M={m} {plan} case={case_name} row={r}"
                    mm.check(key, C.bit_equal(lora[r], m1[r][0][0]), "down lora " + ctx)
                    mm.check(
                        key, C.bit_equal(injection[r], m1[r][1][0]), "down inj " + ctx
                    )
                    mm.check(
                        key, C.bit_equal(block[r], block_ref[r][0]), "up block " + ctx
                    )
    mm.assert_clean("HC down/up rows kernels vs M=1 kernels")


def test_hc_chain_bitwise_vs_m1_route():
    hc = C.hc_module()
    down, up = _hc_weights(21)
    mm = C.Mismatches()
    report = []
    for m in C.m_values():
        worst_block = worst_inj = 0.0
        for case_name, x in C.input_cases(m, K_HC, seed=7 * m + 3):
            # The oracle must be the M=1 replicated Triton route, not the
            # cuBLAS fallback chain.
            assert hc._runtime_ok(x[0:1].clone(), down, up)
            ref = [
                hc._qwen38_sm70_fp16_fused_hc(x[r : r + 1].clone(), down, up)
                for r in range(m)
            ]
            for plan in _plans_for(m):
                block, injection = hc._sx_hc_rows_forward(x, down, up, plan)
                torch.cuda.synchronize()
                assert injection.is_contiguous() and injection.shape == (m, 4)
                key = _form(plan.fused_reduce)
                for r in range(m):
                    ctx = f"M={m} {plan} case={case_name} row={r}"
                    mm.check(key, C.bit_equal(block[r], ref[r][0][0]), "block " + ctx)
                    mm.check(
                        key, C.bit_equal(injection[r], ref[r][1][0]), "inj " + ctx
                    )
            old_block, old_inj = _old_chain(x, down, up)
            worst_block = max(
                worst_block,
                C.max_abs_finite(torch.cat([b for b, _ in ref]), old_block),
            )
            worst_inj = max(
                worst_inj, C.max_abs_finite(torch.cat([i for _, i in ref]), old_inj)
            )
        report.append(["hc", m, f"{worst_block:.3e}", f"{worst_inj:.3e}"])
    C.print_table(
        "HC chain: rows == M=1 route bitwise (asserted); max|M1 - old cuBLAS chain|",
        ["role", "M", "block_max_abs", "inj_max_abs"],
        report,
    )
    mm.assert_clean("HC rows chain vs M=1 replicated route")


def test_hc_poisoned_padding_rows_isolated():
    """Stale NaN/Inf padded rows never change any other row's HC bits."""
    hc = C.hc_module()
    gemv = C.gemv_module()
    down, up = _hc_weights(61)
    mm = C.Mismatches()
    for m in (2, 3, 5, 8, 17, 24):
        x = C.make_rows(m, K_HC, 1.0, seed=600 + m, special="none")
        C.poison_rows(x, range(1, m, 2))
        clean = list(range(0, m, 2))
        ref = {
            r: hc._qwen38_sm70_fp16_fused_hc(x[r : r + 1].clone(), down, up)
            for r in clean
        }
        split = gemv._sx_split_rows
        plans = set(_plans_for(m)) | {
            hc._SxHcRowsPlan(split(m, 3), 4, split(m, 5), fused)
            for fused in (True, False)
        }
        for plan in sorted(plans):
            block, injection = hc._sx_hc_rows_forward(x, down, up, plan)
            torch.cuda.synchronize()
            key = _form(plan.fused_reduce)
            for r in clean:
                ctx = f"M={m} {plan} clean row={r}"
                mm.check(key, C.bit_equal(block[r], ref[r][0][0]), "block " + ctx)
                mm.check(key, C.bit_equal(injection[r], ref[r][1][0]), "inj " + ctx)
    mm.assert_clean("HC rows chain with poisoned padding rows")


def test_hc_graph_replay_changing_inputs():
    hc = C.hc_module()
    down, up = _hc_weights(31)
    for m in (2, 4, 8):
        plan = hc._sx_hc_rows_plan_for_m(m) or hc._SxHcRowsPlan(
            min(m, 4), 4, min(m, 8), True
        )
        x = C.make_rows(m, K_HC, 1.0, seed=m)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            hc._sx_hc_rows_forward(x, down, up, plan)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = hc._sx_hc_rows_forward(x, down, up, plan)
        for step in range(16):
            x.copy_(C.make_rows(m, K_HC, 0.25 + step % 3, seed=300 + step))
            for o in outs:
                o.fill_(float("nan"))
            graph.replay()
            eager = hc._sx_hc_rows_forward(x, down, up, plan)
            torch.cuda.synchronize()
            for o, e in zip(outs, eager):
                assert C.bit_equal(o, e), f"hc M={m} replay {step}"


def test_hc_op_dispatch():
    hc = C.hc_module()
    down, up = _hc_weights(41)
    op = torch.ops.vllm.qwen38_sm70_fp16_fused_hc
    x4 = C.make_rows(4, K_HC, 1.0, seed=4)
    x16 = C.make_rows(16, K_HC, 1.0, seed=16)
    with C.decode_graph_ctx():
        got = op(x4, down, up)
        plan = hc._sx_hc_rows_plan_for_m(4)
        assert plan is not None
        want = hc._sx_hc_rows_forward(x4, down, up, plan)
        assert all(C.bit_equal(a, b) for a, b in zip(got, want))
        # Default table admits HC up to M=8 only.
        got = op(x16, down, up)
        want = _old_chain(x16, down, up)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    with C.decode_graph_ctx(SX_OPT_ROWS="0"):
        got = op(x4, down, up)
        want = _old_chain(x4, down, up)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    with C.sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="1"):
        got = op(x4, down, up)
        want = _old_chain(x4, down, up)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    with C.decode_graph_ctx(SX_OPT_ROWS_TABLE="hc=16"):
        got = op(x16, down, up)
        plan = hc._sx_hc_rows_plan_for_m(16)
        want = hc._sx_hc_rows_forward(x16, down, up, plan)
        assert all(C.bit_equal(a, b) for a, b in zip(got, want))


def _combine_norm_inputs(n: int, seed: int, shared_weight: bool):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = (torch.randn((n, K_HC), generator=gen, device="cuda") * 2).half()
    block = torch.randn((n, HC_DIM), generator=gen, device="cuda").half()
    # Production M=1 injection is a strided view (stride 336) of the gathered
    # down output; the rows route returns a contiguous (N, 4) tensor.
    gathered = torch.randn((n, 336), generator=gen, device="cuda").half()
    weight = (
        torch.randn((HC_DIM if shared_weight else K_HC,), generator=gen, device="cuda")
        * 0.1
    ).half()
    return residual, block, gathered, weight


def test_combine_norm_tile_batch_invariance():
    hc = C.hc_module()
    from vllm.models.qwen4_exp.nvidia.ops import hc as hcops

    eps = 1e-6
    saved = hc._SX_HC_FUSED_MODULES
    hc._SX_HC_FUSED_MODULES = max(1, saved)
    report = []
    try:
        for shared in (True, False):
            for n in (2, 4, 8, 16, 24):
                residual, block, gathered, weight = _combine_norm_inputs(
                    n, 1000 + n, shared
                )
                inj_view = gathered[:, 320:324]
                inj = inj_view.contiguous()
                ref = [
                    hcops._hc_combine_norm(
                        residual[r : r + 1].clone(),
                        block[r : r + 1].clone(),
                        inj_view[r : r + 1],
                        weight,
                        eps,
                        4,
                    )
                    for r in range(n)
                ]
                old_out, old_y = hcops._hc_combine_norm(
                    residual, block, inj, weight, eps, 4
                )
                old_diff_rows = sum(
                    int(not C.bit_equal(old_y[r], ref[r][1][0])) for r in range(n)
                )
                for r in range(n):
                    assert C.bit_equal(old_out[r], ref[r][0][0])  # elementwise
                with C.decode_graph_ctx():
                    admitted = hc._sx_hc_rows_plan_for_m(n) is not None
                    # The kernel must actually switch where HC is admitted.
                    assert hcops._sx_rows_norm_uses_m1_tile(
                        n, HC_DIM, 4, torch.float16
                    ) is admitted, f"combine_norm tile admission N={n}"
                    out, y = hcops._hc_combine_norm(
                        residual, block, inj, weight, eps, 4
                    )
                torch.cuda.synchronize()
                for r in range(n):
                    assert C.bit_equal(out[r], ref[r][0][0])
                    if admitted:
                        assert C.bit_equal(y[r], ref[r][1][0]), (
                            f"combine_norm N={n} shared={shared} row={r}"
                        )
                if not admitted:
                    assert C.bit_equal(y, old_y)
                report.append(
                    [n, shared, admitted, old_diff_rows, n, f"{C.max_abs_finite(old_y, torch.cat([y_ for _, y_ in ref])):.3e}"]
                )
    finally:
        hc._SX_HC_FUSED_MODULES = saved
    C.print_table(
        "hc_combine_norm (MR9a): old 512-tile rows differing from N=1 (info); "
        "admitted widths asserted bitwise",
        ["N", "shared_w", "admitted", "old_rows_diff", "rows", "old_max_abs"],
        report,
    )


def _combine_norm_launch(residual, block, inj, weight, eps, block_size, prefetch):
    from vllm.models.qwen4_exp.nvidia.ops import hc as hcops
    from vllm.platforms import current_platform

    n = residual.shape[0]
    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    hcops._hc_combine_norm_kernel[(n, 4)](
        block, residual, inj, weight, out, y,
        block.stride(0), residual.stride(0), inj.stride(0), out.stride(0),
        y.stride(0), HC_DIM, 4,
        W_SHARED=weight.numel() == HC_DIM, EPS=eps, BLOCK_SIZE=block_size,
        launch_pdl=current_platform.is_arch_support_pdl(),
        PREFETCH_WEIGHT=prefetch, num_warps=4,
    )  # fmt: skip
    return out, y


def run_benchmark() -> None:
    hc = C.hc_module()
    copies = C.n_copies((336 * K_HC + K_HC * RANK) * 2)
    pairs = [_hc_weights(500 + 2 * i) for i in range(copies)]
    rows_out = []
    per_m = {}
    best_cfg = {}
    for m in C.BENCH_M:
        x = C.make_rows(m, K_HC, 1.0, seed=m, special="none")
        split = C.gemv_module()._sx_split_rows
        plans = {
            "rows_default": hc._SxHcRowsPlan(split(m, 4), 4, split(m, 8), True),
            "rows_plain": hc._SxHcRowsPlan(split(m, 4), 4, split(m, 8), False),
            "nw2": hc._SxHcRowsPlan(split(m, 4), 2, split(m, 8), True),
            "nw1": hc._SxHcRowsPlan(split(m, 4), 1, split(m, 8), True),
        }
        if m >= 4:
            plans["down_t2"] = hc._SxHcRowsPlan(split(m, 2), 4, split(m, 8), True)
        if m >= 8:
            plans["up_t4"] = hc._SxHcRowsPlan(split(m, 4), 4, split(m, 4), True)
        variants = {"cublas": lambda i: _old_chain(x, *pairs[i % copies])}
        for key, plan in plans.items():
            variants[key] = lambda i, p=plan: hc._sx_hc_rows_forward(
                x, *pairs[i % copies], p
            )
        if m <= 8:
            variants["m1_loop"] = lambda i: [
                hc._qwen38_sm70_fp16_fused_hc(x[r : r + 1], *pairs[i % copies])
                for r in range(m)
            ]
        t = C.bench_graphs(variants, calls=32)
        best_key = min((k for k in t if k in plans), key=lambda k: t[k])
        best_cfg[m] = (best_key, plans[best_key])
        per_m[m] = (t["cublas"], t[best_key])
        rows_out.append(
            [m, f"{t['cublas']:.2f}"]
            + [f"{t.get(k, float('nan')):.2f}" for k in (
                "rows_default", "rows_plain", "nw2", "nw1", "down_t2", "up_t4",
                "m1_loop",
            )]  # fmt: skip
            + [best_key, f"{(t['cublas'] - t[best_key]) * 96 / 1000:.3f}"]
        )
    C.print_table(
        "HC mix CUDA-graph microbenchmark, us/call (cuBLAS chain vs rows plans)",
        ["M", "cublas", "default", "plain", "nw2", "nw1", "down_t2", "up_t4",
         "m1_loop", "best", "ms/step_saved(96)"],
        rows_out,
    )  # fmt: skip
    print(f"\nSuggested SX_OPT_ROWS_TABLE hc={C.admitted_max_m(per_m)}")
    for m, (key, plan) in best_cfg.items():
        print(
            f"  M={m}: best={key} -> SX_OPT_ROWS_HC_DOWN_TILE>={plan.down_rows} "
            f"SX_OPT_ROWS_HC_DOWN_NW={plan.down_nw} "
            f"SX_OPT_ROWS_HC_UP_TILE>={plan.up_rows} "
            f"SX_OPT_ROWS_FUSED_REDUCE={int(plan.fused_reduce)}"
        )

    norm_rows = []
    for n in C.BENCH_M:
        residual, block, gathered, weight = _combine_norm_inputs(n, 7 + n, True)
        inj = gathered[:, 320:324].contiguous()
        t = C.bench_graphs(
            {
                "tile512": lambda i: _combine_norm_launch(
                    residual, block, inj, weight, 1e-6, 512, False
                ),
                "tile1024_prefetch": lambda i: _combine_norm_launch(
                    residual, block, inj, weight, 1e-6, 1024, True
                ),
            },
            calls=32,
        )
        norm_rows.append(
            [n, f"{t['tile512']:.2f}", f"{t['tile1024_prefetch']:.2f}",
             f"{(t['tile1024_prefetch'] - t['tile512']) * 96 / 1000:+.3f}"]
        )  # fmt: skip
    C.print_table(
        "hc_combine_norm us/call: old N>1 tile vs M=1 tile (MR9a cost, 96 calls)",
        ["N", "tile512", "tile1024+prefetch", "ms/step_delta"],
        norm_rows,
    )


@pytest.mark.skipif(
    os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0"), reason="bench off"
)
def test_zz_microbenchmark(capsys):
    with capsys.disabled():
        run_benchmark()


def main() -> None:
    if not C.sm70_available():
        raise SystemExit("requires an SM70 GPU")
    test_hc_down_up_kernels_bitwise()
    test_hc_chain_bitwise_vs_m1_route()
    test_hc_poisoned_padding_rows_isolated()
    test_hc_graph_replay_changing_inputs()
    test_hc_op_dispatch()
    test_combine_norm_tile_batch_invariance()
    print("\nPASS: HC rows and admitted combine-norm rows equal the M=1 route.")
    if os.environ.get("SX_TEST_NO_BENCH", "0") in ("", "0"):
        run_benchmark()


if __name__ == "__main__":
    main()
