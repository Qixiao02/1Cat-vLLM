# SPDX-License-Identifier: Apache-2.0
"""b3 lane-core (design_1 MTP-1): the batch-invariant rows kernels and M=1
kernels at MTP verify widths W = B*(k+1), through the production op dispatch
(torch.ops.vllm.qwen38_sm70_fp16_gemv / _gdn_input / _fused_hc and
hc_combine_norm) inside the native-MTP lane's FULL verify-graph context.

GPU: ONE V100 (SM70). Runs inside image 1.8.0-dev2 with the overlay
bind-mounted (uses sx_tests/dense-multirow/_sx_rows_common.py):

  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-lane-core/test_mtp_rows_verify_widths.py
  (SX_TEST_NO_BENCH=1 skips the informational microbenchmark.)

Widths: every B*(k+1) for k in {1,2,3,4} and B in {1,2,3,4,6,8,12,16,20,24}
(2..120), realistic TP4-local Qwen3.8 shapes (hidden 2560, QSA qkv 3584,
o 2560x1536, index 640, router 512 x 2560 (E512), GDN qkvz 4096 + ba 24,
HC down 336x10240 / up 10240x320).

Asserts (bitwise = int16-view equality, NaN/Inf/-0 included):
  * production rows table (gdn_in/qsa_qkv/qsa_index/router 8, gdn_out/qsa_o
    4, hc 2): at every verify width W <= role maximum each output row of the
    op == the unchanged M=1 kernel applied to that row alone; above it the
    op == the old cuBLAS path (F.linear / F.linear + split / cuBLAS HC
    chain) bit for bit, so wide verify steps are unchanged;
  * the same for the implementation table (router 24, hc 8, qsa_o/gdn_out 8)
    so the mechanism is covered up to W=24;
  * SX_OPT_MTP_ROWS=0 in the lane: the old cuBLAS path at every width;
  * outside the FULL verify context (target main backbone = prefill/mixed/
    PW-1 steps, and the drafter) the op never takes the rows kernels;
  * a padded verify request (FULL graph for B_graph requests, B_live < B_graph
    live) with NaN/Inf garbage rows leaves the live rows' bits unchanged;
  * MR9a: hc_combine_norm at admitted HC widths is per-row equal to N=1.
Prints the (role, W) admission table and (unless SX_TEST_NO_BENCH=1) a CUDA
graph microbenchmark rows vs cuBLAS at the rows-admitted verify widths with
the per-verify-step estimate (calls per step x delta).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_lane_common as L  # noqa: E402

C = L.rows_common()

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")

WIDTHS = L.verify_widths()  # 2..120
ROW_WIDTHS = [w for w in WIDTHS if w <= 8]  # 2, 3, 4, 5, 6, 8
IMPL_WIDTHS = [w for w in WIDTHS if w <= 24]
K_HC = 10240
RANK = 320
HC_DIM = 2560


def _role_max(gemv, key: str) -> int:
    return gemv._sx_rows_max_m(key)


def _gemv_op(x, weight, prefix):
    return torch.ops.vllm.qwen38_sm70_fp16_gemv(x, weight, prefix)


def _m1_rows(gemv, x, weight, prefix):
    rows = []
    for r in range(x.shape[0]):
        row = x[r : r + 1].clone()
        assert gemv._runtime_ok(row, weight)
        rows.append(gemv._qwen38_sm70_fp16_gemv(row, weight, prefix))
    return torch.cat(rows)


def _check_gemv(table: str | None, widths, mm: C.Mismatches, report: list):
    gemv = C.gemv_module()
    for name, prefix, shape, _, _ in C.GEMV_ROLES:
        n, k = shape
        key = gemv._sx_role_key(prefix, shape)
        weight = C.make_weight(n, k, 0.05, seed=5000 + n + k)
        for w in widths:
            x = C.make_rows(w, k, 1.0, seed=31 * w + n)
            with L.mtp_verify_ctx(SX_OPT_ROWS_TABLE=table):
                cap = _role_max(gemv, key)
                got = _gemv_op(x, weight, prefix)
            torch.cuda.synchronize()
            if w <= cap:
                want = _m1_rows(gemv, x, weight, prefix)
                for r in range(w):
                    mm.check(
                        f"{table or 'default'} rows",
                        C.bit_equal(got[r], want[r]),
                        f"{name} W={w} row={r}",
                    )
                route = "rows==M1"
            else:
                mm.check(
                    f"{table or 'default'} cublas",
                    C.bit_equal(got, F.linear(x, weight)),
                    f"{name} W={w}",
                )
                route = "cuBLAS"
            report.append([table or "default", name, w, cap, route])


def test_gemv_verify_widths_default_table():
    mm, report = C.Mismatches(), []
    _check_gemv(None, WIDTHS, mm, report)
    admitted = sorted({(r[1], r[2]) for r in report if r[4] == "rows==M1"})
    C.print_table(
        "MTP verify widths: production rows table (asserted bitwise)",
        ["table", "role", "W", "role_max", "route"],
        [r for r in report if r[2] <= 24],
    )
    # Every rows-admitted verify width is one of 2,3,4,5,6,8.
    assert {w for _, w in admitted} <= set(ROW_WIDTHS)
    mm.assert_clean("GEMV at MTP verify widths (production table)")


def test_gemv_verify_widths_impl_table():
    mm, report = C.Mismatches(), []
    _check_gemv(C.IMPL_TABLE, IMPL_WIDTHS, mm, report)
    mm.assert_clean("GEMV at MTP verify widths (implementation table)")


def test_gemv_mtp_rows_switch_and_other_contexts():
    from vllm.compilation import sm70_decode_graph as dg

    mm = C.Mismatches()
    for name, prefix, shape, _, _ in C.GEMV_ROLES:
        n, k = shape
        weight = C.make_weight(n, k, 0.05, seed=6000 + n)
        for w in ROW_WIDTHS:
            x = C.make_rows(w, k, 1.0, seed=w + n)
            ref = F.linear(x, weight)
            with L.mtp_verify_ctx(SX_OPT_MTP_ROWS="0"):
                got = _gemv_op(x, weight, prefix)
            mm.check("SX_OPT_MTP_ROWS=0", C.bit_equal(got, ref), f"{name} W={w}")
            with L.mtp_main_ctx():
                got = _gemv_op(x, weight, prefix)
            mm.check("target main backbone", C.bit_equal(got, ref), f"{name} W={w}")
            # Drafter: lane installed, no FULL target capture context.
            saved = dg.sm70_mtp_lane_installed()
            dg.set_sm70_mtp_lane_installed(True)
            try:
                with C.sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="1"):
                    got = _gemv_op(x, weight, prefix)
            finally:
                dg.set_sm70_mtp_lane_installed(saved)
            mm.check("drafter", C.bit_equal(got, ref), f"{name} W={w}")
    mm.assert_clean("rows kernels never outside the MTP verify graph")


def _gdn_weights(seed=4242):
    return (
        C.make_weight(4096, 2560, 0.05, seed),
        C.make_weight(24, 2560, 0.05, seed + 1),
    )


def _old_gdn_input(gemv, x, qkvz_w, ba_w):
    qkvz = F.linear(x, qkvz_w)
    ba = F.linear(x, ba_w)
    if gemv._can_fuse_gdn_projection_split(qkvz, ba):
        return gemv._split_gdn_projection_outputs(qkvz, ba)
    return (
        qkvz[..., :2560].contiguous(),
        qkvz[..., 2560:].contiguous(),
        ba[..., :12].contiguous(),
        ba[..., 12:].contiguous(),
    )


def test_gdn_input_verify_widths():
    gemv = C.gemv_module()
    qkvz_w, ba_w = _gdn_weights()
    op = torch.ops.vllm.qwen38_sm70_fp16_gdn_input
    mm = C.Mismatches()
    for w in WIDTHS:
        x = C.make_rows(w, 2560, 1.0, seed=77 * w)
        with L.mtp_verify_ctx():
            cap = _role_max(gemv, "gdn_in")
            got = op(x, qkvz_w, ba_w)
        torch.cuda.synchronize()
        if w <= cap:
            want = [
                gemv._qwen38_sm70_fp16_gdn_input(x[r : r + 1].clone(), qkvz_w, ba_w)
                for r in range(w)
            ]
            for r in range(w):
                for part, out, ref in zip(("qkv", "z", "b", "a"), got, want[r]):
                    mm.check("rows", C.bit_equal(out[r], ref[0]), f"{part} W={w} r={r}")
        else:
            old = _old_gdn_input(gemv, x, qkvz_w, ba_w)
            for part, out, ref in zip(("qkv", "z", "b", "a"), got, old):
                mm.check("cublas", C.bit_equal(out, ref), f"{part} W={w}")
        with L.mtp_verify_ctx(SX_OPT_MTP_ROWS="0"):
            got = op(x, qkvz_w, ba_w)
        old = _old_gdn_input(gemv, x, qkvz_w, ba_w)
        for part, out, ref in zip(("qkv", "z", "b", "a"), got, old):
            mm.check("SX_OPT_MTP_ROWS=0", C.bit_equal(out, ref), f"{part} W={w}")
    mm.assert_clean("fused GDN input at MTP verify widths")


def _hc_weights(seed):
    down = C.make_weight(336, K_HC, 0.02, seed)
    down[324:].zero_()
    up = C.make_weight(K_HC, RANK, 0.05, seed + 1)
    return down.contiguous(), up.contiguous()


def _old_hc_chain(x, down, up):
    dai = F.linear(x, down)
    lora = torch.ops.vllm.qwen4_exp_hc_silu(dai[..., :RANK], 4)
    injection = dai[..., RANK : RANK + 4]
    gate = F.linear(lora, up)
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, 4), injection


@pytest.mark.parametrize("table", [None, "hc=8"])
def test_fused_hc_verify_widths(table):
    hc = C.hc_module()
    down, up = _hc_weights(91)
    op = torch.ops.vllm.qwen38_sm70_fp16_fused_hc
    mm = C.Mismatches()
    widths = WIDTHS if table is None else IMPL_WIDTHS
    for w in widths:
        x = C.make_rows(w, K_HC, 1.0, seed=13 * w)
        with L.mtp_verify_ctx(SX_OPT_ROWS_TABLE=table):
            cap = C.gemv_module()._sx_rows_max_m("hc")
            got = op(x, down, up)
        torch.cuda.synchronize()
        if w <= cap:
            want = [
                hc._qwen38_sm70_fp16_fused_hc(x[r : r + 1].clone(), down, up)
                for r in range(w)
            ]
            for r in range(w):
                mm.check(
                    "rows", C.bit_equal(got[0][r], want[r][0][0]), f"block W={w} r={r}"
                )
                mm.check(
                    "rows", C.bit_equal(got[1][r], want[r][1][0]), f"inj W={w} r={r}"
                )
        else:
            old = _old_hc_chain(x, down, up)
            same = all(C.bit_equal(a, b) for a, b in zip(got, old))
            mm.check("cublas", same, f"W={w}")
    mm.assert_clean(f"fused HC at MTP verify widths (table={table or 'default'})")


def test_combine_norm_verify_widths():
    hc = C.hc_module()
    from vllm.models.qwen4_exp.nvidia.ops import hc as hcops

    eps = 1e-6
    saved = hc._SX_HC_FUSED_MODULES
    hc._SX_HC_FUSED_MODULES = max(1, saved)
    try:
        for w in [w for w in WIDTHS if w <= 24]:
            gen = torch.Generator(device="cuda").manual_seed(500 + w)
            residual = (torch.randn((w, K_HC), generator=gen, device="cuda") * 2).half()
            block = torch.randn((w, HC_DIM), generator=gen, device="cuda").half()
            inj = torch.randn((w, 4), generator=gen, device="cuda").half()
            weight = (torch.randn((HC_DIM,), generator=gen, device="cuda") * 0.1).half()
            old_out, old_y = hcops._hc_combine_norm(residual, block, inj, weight, eps, 4)
            with L.mtp_verify_ctx():
                admitted = hc._sx_hc_rows_plan_for_m(w) is not None
                out, y = hcops._hc_combine_norm(residual, block, inj, weight, eps, 4)
            torch.cuda.synchronize()
            assert C.bit_equal(out, old_out)
            if admitted:
                for r in range(w):
                    ref = hcops._hc_combine_norm(
                        residual[r : r + 1].clone(),
                        block[r : r + 1].clone(),
                        inj[r : r + 1].clone(),
                        weight,
                        eps,
                        4,
                    )
                    assert C.bit_equal(y[r], ref[1][0]), f"combine_norm W={w} r={r}"
            else:
                assert C.bit_equal(y, old_y), f"combine_norm W={w} changed"
    finally:
        hc._SX_HC_FUSED_MODULES = saved


@pytest.mark.parametrize("k", L.K_VALUES)
def test_padded_verify_request_isolated(k):
    """FULL graph for B_graph requests with fewer live requests: stale padded
    rows (NaN/Inf garbage) never change the live rows (rows kernels)."""
    gemv = C.gemv_module()
    q = k + 1
    mm = C.Mismatches()
    checked = 0
    for name, prefix, shape, _, _ in C.GEMV_ROLES:
        n, kk = shape
        key = gemv._sx_role_key(prefix, shape)
        weight = C.make_weight(n, kk, 0.05, seed=7000 + n)
        with L.mtp_verify_ctx(SX_OPT_ROWS_TABLE=C.IMPL_TABLE):
            cap = _role_max(gemv, key)
        for b_graph in L.LANE_REQS:
            w = q * b_graph
            if w > cap:
                break
            for b_live in range(1, b_graph):
                x = C.make_rows(w, kk, 1.0, seed=w * 7 + b_live, special="none")
                clean = x.clone()
                C.poison_rows(x, range(q * b_live, w))
                with L.mtp_verify_ctx(SX_OPT_ROWS_TABLE=C.IMPL_TABLE):
                    got = _gemv_op(x, weight, prefix)
                    want = _gemv_op(clean, weight, prefix)
                torch.cuda.synchronize()
                live = q * b_live
                mm.check(
                    "padded",
                    C.bit_equal(got[:live], want[:live]),
                    f"{name} k={k} B={b_live}/{b_graph}",
                )
                checked += 1
    assert checked > 0
    mm.assert_clean("padded verify requests isolated")


def test_zz_microbenchmark():
    """Informational: rows vs cuBLAS in CUDA graphs at rows-admitted verify
    widths (production table), with calls per verify step."""
    if os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0"):
        pytest.skip("SX_TEST_NO_BENCH=1")
    gemv = C.gemv_module()
    rows = []
    for name, prefix, shape, calls, bench in C.GEMV_ROLES:
        if not bench:
            continue
        n, k = shape
        key = gemv._sx_role_key(prefix, shape)
        copies = C.n_copies(n * k * 2)
        weights = [C.make_weight(n, k, 0.05, seed=900 + i) for i in range(copies)]
        for w in ROW_WIDTHS:
            with L.mtp_verify_ctx():
                if w > _role_max(gemv, key):
                    continue
                plan = gemv._plan_for(prefix, shape)
                tile = gemv._sx_rows_tile(C.make_rows(w, k, 1.0, 1), weights[0], key)
            x = C.make_rows(w, k, 1.0, seed=w)
            times = C.bench_graphs(
                {
                    "cublas": lambda i: F.linear(x, weights[i % copies]),
                    "rows": lambda i: gemv._sx_rows_gemv(
                        x, weights[i % copies], plan, tile
                    ),
                },
                calls=32,
            )
            delta_ms = (times["cublas"] - times["rows"]) * calls / 1000.0
            rows.append(
                [name, w, f"{times['cublas']:.2f}", f"{times['rows']:.2f}", calls, f"{delta_ms:+.3f}"]
            )
    qkvz_ws = [_gdn_weights(1200 + 2 * i) for i in range(C.n_copies(4120 * 2560 * 2))]
    for w in ROW_WIDTHS:
        with L.mtp_verify_ctx():
            if w > _role_max(gemv, "gdn_in"):
                continue
            tile = gemv._sx_gdn_rows_tile(
                C.make_rows(w, 2560, 1.0, 1), qkvz_ws[0][0], qkvz_ws[0][1]
            )
        x = C.make_rows(w, 2560, 1.0, seed=w)
        n_ws = len(qkvz_ws)
        times = C.bench_graphs(
            {
                "cublas": lambda i: _old_gdn_input(gemv, x, *qkvz_ws[i % n_ws]),
                "rows": lambda i: gemv._sx_rows_gdn_input(x, *qkvz_ws[i % n_ws], tile),
            },
            calls=32,
        )
        delta_ms = (times["cublas"] - times["rows"]) * 36 / 1000.0
        rows.append(
            ["gdn_in(fused)", w, f"{times['cublas']:.2f}", f"{times['rows']:.2f}", 36, f"{delta_ms:+.3f}"]
        )
    C.print_table(
        "MTP verify rows vs cuBLAS (us/call, CUDA graph); est. ms saved per verify step",
        ["role", "W", "cublas_us", "rows_us", "calls/step", "saved_ms"],
        rows,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
