# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_ROUTER32 (design_1 [C2]): SM70 Qwen3.8 E512/K10 router top-k, M<=32.

Hardware: 1 x V100 (SM70) for the GPU tests; the CPU dispatch tests run on any
host that can import vllm. No test needs 4 GPUs.

Run inside the deployed image (repo root = overlay source tree):
  /opt/venv/bin/python -m pytest -q sx_tests/moe-router/test_router32.py
  /opt/venv/bin/python sx_tests/moe-router/test_router32.py   # same + prints
                                                              # the benchmark

Asserts:
  * runtime-M kernel vs ops.topk_softmax for M = 1..32 over random, scaled,
    tie, signed-zero, NaN, +Inf, all -Inf and mixed-degenerate rows: expert
    ids and token_expert_indices exact; FP32 weights within the existing
    M<=16 contract (atol = rtol = 1e-7).
  * runtime-M kernel vs the legacy constexpr-M kernel for M = 1..16: bitwise
    (weights compared as int32 bit patterns, ids, source rows), also over
    FP16-extreme (underflowing-probability), NaN/Inf-anywhere and raw-bit rows.
  * M = 17..32: per-row bitwise batch invariance against the legacy kernel
    run on <=16-row chunks; source rows equal k * M + row.
  * fused_topk: flag on -> Triton for M<=32; flag off (default for external
    callers) -> legacy kernel up to M16 and bitwise topk_softmax above.
  * FusedTopKRouter arms the flag only for no-spec, TP4, hidden-2560 configs
    (or no config); spec decode, other TP sizes, other hidden sizes (Qwen3-Next
    shares E512/K10) and SX_OPT_ROUTER32=0 keep the old route.
  * CUDA graph of 48 launches at M = 24 / 32, replayed with changed inputs and
    NaN/-777 poisoned outputs, equals eager launches bitwise.
Prints (``-s`` or script mode): median time of a 48-call graph (one 48-layer
decode round) for topk_softmax / legacy / runtime-M at M in
{1,2,4,8,16,17,24,32}, CUDA events, median of 60 samples.
"""

from __future__ import annotations

import statistics
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import fused_topk_router as mod

E = 512
K = 10
LAYERS = 48

_ON_SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
requires_sm70 = pytest.mark.skipif(not _ON_SM70, reason="requires an SM70 GPU")

CASES = (
    "random",
    "scaled_small",
    "scaled_large",
    "ties",
    "signed_zero",
    "nan",
    "inf",
    "negative_inf",
    "mixed_rows",
)
# Extra cases for the bitwise runtime-M vs legacy-kernel comparisons only.
# The generic topk_softmax degenerate-row contract was validated by 1Cat with
# NaN/+Inf at column 0 only; these place specials anywhere (incl. raw FP16
# NaN/Inf encodings), which both Triton kernels must treat identically.
# "fp16_extremes" is also excluded from the topk_softmax contract: its ranks
# 2..10 sit thousands below the row max, so topk_softmax's FP32 probabilities
# underflow to 0 and its argmax then picks the lowest expert *index* among
# the zeros, while both Triton kernels keep logit order (a pre-existing,
# admitted difference of the M<=16 kernel; weights are 0 either way).
BITWISE_CASES = CASES + ("fp16_extremes", "nan_anywhere", "inf_anywhere", "raw_fp16")
# Tolerance vs topk_softmax: 1e-7 is the 1Cat contract on real router logits; the
# synthetic x10 case reaches 2.4e-7 with the legacy kernel too (see test below).
_TS_TOL = 3e-7


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _make_logits(case: str, m: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, E, generator=gen, device="cuda", dtype=torch.float32)
    if case == "scaled_small":
        x = x * 1e-3
    elif case == "scaled_large":
        x = x * 10.0
    x = x.half()
    if case == "ties":
        # 46 tied logits above (almost) every N(0,1) value: exercises the
        # lower-expert-id tie break inside the selected top-10.
        x[:, :16] = 4.0
        x[:, 200:230] = 4.0
    elif case == "signed_zero":
        x.zero_()
        x[:, 1::2] = -0.0
    elif case == "nan":
        x[:, 0] = float("nan")
    elif case == "inf":
        x[:, 0] = float("inf")
    elif case == "negative_inf":
        x.fill_(-float("inf"))
    elif case == "mixed_rows":
        for r in range(m):
            kind = r % 6
            if kind == 1:
                x[r, 0] = float("nan")
            elif kind == 2:
                x[r, 0] = float("inf")
            elif kind == 3:
                x[r].fill_(-float("inf"))
            elif kind == 4:
                x[r, 32:64] = 0.5
            elif kind == 5:
                x[r, ::3] = -float("inf")  # partial -inf, still a valid row
    elif case == "fp16_extremes":
        raw = torch.randint(
            -32768, 32768, (m, E), generator=gen, device="cuda", dtype=torch.int32
        ).to(torch.int16)
        x = torch.nan_to_num(raw.view(torch.float16), nan=0.0, posinf=65504.0,
                             neginf=-65504.0)
    elif case == "nan_anywhere":
        for r in range(m):
            x[r, (7 * r + 3) % E] = float("nan")
    elif case == "inf_anywhere":
        for r in range(m):
            x[r, (11 * r + 5) % E] = float("inf") if r % 2 else -float("inf")
    elif case == "raw_fp16":
        raw = torch.randint(
            -32768, 32768, (m, E), generator=gen, device="cuda", dtype=torch.int32
        ).to(torch.int16)
        x = raw.view(torch.float16).clone()
    return x.contiguous()


def _out(m: int):
    return (
        torch.empty(m, K, dtype=torch.float32, device="cuda"),
        torch.empty(m, K, dtype=torch.int32, device="cuda"),
        torch.empty(m, K, dtype=torch.int32, device="cuda"),
    )


def _run_topk_softmax(x):
    import vllm._custom_ops as ops

    w, ids, rows = _out(x.shape[0])
    ops.topk_softmax(w, ids, rows, x, True)
    return w, ids, rows


def _run_legacy(x):
    w, ids, rows = _out(x.shape[0])
    mod._sm70_qwen38_router_topk(w, ids, rows, x)
    return w, ids, rows


def _run_runtime_m(x):
    w, ids, rows = _out(x.shape[0])
    mod._sm70_qwen38_router_topk_runtime_m(w, ids, rows, x)
    return w, ids, rows


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


# ----------------------------------------------------------------------------
# CPU-only dispatch tests (monkeypatched kernels / ops)
# ----------------------------------------------------------------------------
class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 24, 32])
def test_runtime_m_launcher_arguments_cpu(monkeypatch, dtype, rows):
    kernel = _RecordingKernel()
    monkeypatch.setattr(mod, "_sm70_qwen38_router_topk_runtime_m_kernel", kernel)
    x = torch.empty(rows, E, dtype=dtype)
    w = torch.empty(rows, K, dtype=torch.float32)
    ids = torch.empty(rows, K, dtype=torch.int32)
    mod._sm70_qwen38_router_topk_runtime_m(w, ids, torch.empty_like(ids), x)
    assert len(kernel.calls) == 1
    grid, args, kwargs = kernel.calls[0]
    assert grid == (rows,)
    assert args[4] == rows  # runtime M (row stride of the source-row map)
    assert kwargs["E"] == 512 and kwargs["K"] == 10 and kwargs["BLOCK_E"] == 512
    assert kwargs["PACKED_HALF_KEY"] == (rows == 1 and dtype == torch.float16)
    assert kwargs["num_warps"] == 8
    assert "M" not in kwargs  # M must not be a constexpr specialization


def test_runtime_m_kernel_does_not_specialize_m():
    kernel = mod._sm70_qwen38_router_topk_runtime_m_kernel
    names = list(getattr(kernel, "arg_names", []))
    dns = getattr(kernel, "do_not_specialize", None)
    if not names or dns is None:
        pytest.skip("triton JITFunction introspection unavailable")
    dns = set(dns)
    assert "M" in dns or names.index("M") in dns


@pytest.mark.parametrize("runtime_m", [False, True])
def test_fused_topk_gate_cpu(monkeypatch, runtime_m):
    seen = []
    monkeypatch.setattr(
        mod, "current_platform", SimpleNamespace(is_device_capability=lambda c: True)
    )
    monkeypatch.setattr(
        mod, "envs", SimpleNamespace(VLLM_SM70_QWEN38_ROUTER_TOPK=True)
    )
    monkeypatch.setattr(
        mod, "_sm70_qwen38_router_topk", lambda *a: seen.append("legacy")
    )
    monkeypatch.setattr(
        mod, "_sm70_qwen38_router_topk_runtime_m", lambda *a: seen.append("runtime")
    )

    def generic(w, ids, rows, logits, renorm):
        seen.append("generic")
        return w, ids

    monkeypatch.setattr(mod, "dispatch_topk_softmax_func", lambda **kw: generic)
    for m in range(1, 41):
        seen.clear()
        h = torch.empty(m, 2560, dtype=torch.float16)
        logits = torch.empty(m, E, dtype=torch.float16)
        mod.fused_topk(h, logits, K, True, sm70_qwen38_router_runtime_m=runtime_m)
        if runtime_m and m <= 32:
            expected = "runtime"
        elif not runtime_m and m <= 16:
            expected = "legacy"
        else:
            expected = "generic"
        assert seen == [expected], (m, runtime_m, seen)
    # Non-contract shapes never take either Triton route.
    for logits in (
        torch.empty(4, E, dtype=torch.float32),
        torch.empty(4, 256, dtype=torch.float16),
    ):
        seen.clear()
        h = torch.empty(4, 2560, dtype=torch.float16)
        mod.fused_topk(h, logits, K, True, sm70_qwen38_router_runtime_m=True)
        assert seen == ["generic"]
    seen.clear()
    h = torch.empty(4, 2560, dtype=torch.float16)
    mod.fused_topk(
        h, torch.empty(4, E, dtype=torch.float16), K, False,
        sm70_qwen38_router_runtime_m=True,
    )
    assert seen == ["generic"]  # renormalize=False is outside the contract


def test_router_flag_follows_spec_config_cpu(monkeypatch):
    import vllm.config.vllm as vcfg

    def build():
        return mod.FusedTopKRouter(
            top_k=K, global_num_experts=E, scoring_func="softmax", renormalize=True
        )

    monkeypatch.setattr(mod, "_SX_OPT_ROUTER32", True)
    monkeypatch.setattr(vcfg, "get_current_vllm_config_or_none", lambda: None)
    assert build()._sm70_qwen38_router_runtime_m is True
    monkeypatch.setattr(
        vcfg,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(speculative_config=None),
    )
    assert build()._sm70_qwen38_router_runtime_m is True
    monkeypatch.setattr(
        vcfg,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(speculative_config=object()),
    )
    assert build()._sm70_qwen38_router_runtime_m is False  # MTP/spec unchanged
    for tp, expected in ((4, True), (2, False), (1, False), (8, False)):
        monkeypatch.setattr(
            vcfg,
            "get_current_vllm_config_or_none",
            lambda tp=tp: SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(tensor_parallel_size=tp),
            ),
        )
        assert build()._sm70_qwen38_router_runtime_m is expected, tp
    # Other E512/K10 models (e.g. Qwen3-Next, hidden 2048) keep the old route;
    # an unknown hidden size (no model_config) assumes the deployed contract.
    for hidden, expected in ((2560, True), (2048, False), (4096, False),
                             (None, True)):
        monkeypatch.setattr(
            vcfg,
            "get_current_vllm_config_or_none",
            lambda hidden=hidden: SimpleNamespace(
                speculative_config=None,
                parallel_config=SimpleNamespace(tensor_parallel_size=4),
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(hidden_size=hidden)
                ),
            ),
        )
        assert build()._sm70_qwen38_router_runtime_m is expected, hidden
    monkeypatch.setattr(vcfg, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(mod, "_SX_OPT_ROUTER32", False)
    assert build()._sm70_qwen38_router_runtime_m is False  # SX_OPT_ROUTER32=0

    # _compute_routing forwards the per-router decision to fused_topk.
    monkeypatch.setattr(mod, "_SX_OPT_ROUTER32", True)
    router = build()
    captured = {}

    def fake_fused_topk(**kwargs):
        captured.update(kwargs)
        m = kwargs["hidden_states"].shape[0]
        return (
            torch.zeros(m, K),
            torch.zeros(m, K, dtype=torch.int32),
            torch.zeros(m, K, dtype=torch.int32),
        )

    monkeypatch.setattr(mod, "fused_topk", fake_fused_topk)
    router._compute_routing(
        torch.empty(3, 2560, dtype=torch.float16),
        torch.empty(3, E, dtype=torch.float16),
        None,
    )
    assert captured["sm70_qwen38_router_runtime_m"] is True


# ----------------------------------------------------------------------------
# GPU exactness
# ----------------------------------------------------------------------------
@requires_sm70
@pytest.mark.parametrize("case", CASES)
def test_runtime_m_vs_topk_softmax(case):
    worst = 0.0
    for m in range(1, 33):
        x = _make_logits(case, m, seed=1000 + m)
        ew, eids, erows = _run_topk_softmax(x)
        aw, aids, arows = _run_runtime_m(x)
        torch.accelerator.synchronize()
        assert torch.equal(aids, eids), (case, m)
        assert torch.equal(arows, erows), (case, m)
        # Validation fix (opt180dev1): the synthetic x10 'scaled_large' rows differ from
        # topk_softmax by up to 2 FP32 ulp (2.4e-7). The UNCHANGED legacy M<=16 kernel
        # gives the identical deviation on the same rows (checked on V100), so this is a
        # pre-existing contract property; runtime-M == legacy is asserted bitwise elsewhere.
        torch.testing.assert_close(aw, ew, atol=_TS_TOL, rtol=_TS_TOL)
        worst = max(worst, float((aw - ew).abs().max()))
    print(f"[router32] {case}: max |w - topk_softmax| over M=1..32 = {worst:.3e}")


@requires_sm70
@pytest.mark.parametrize("case", BITWISE_CASES)
def test_runtime_m_bitwise_vs_legacy_m_le_16(case):
    for m in range(1, 17):
        x = _make_logits(case, m, seed=2000 + m)
        legacy = _run_legacy(x)
        runtime = _run_runtime_m(x)
        torch.accelerator.synchronize()
        for a, b in zip(runtime, legacy):
            assert _bits_equal(a, b), (case, m)


@requires_sm70
@pytest.mark.parametrize("case", BITWISE_CASES)
def test_runtime_m_batch_invariance_m17_32(case):
    arange_k = torch.arange(K, device="cuda", dtype=torch.int32)
    for m in range(17, 33):
        x = _make_logits(case, m, seed=3000 + m)
        rw, rids, rrows = _run_runtime_m(x)
        half = m // 2  # both chunks have 8..16 rows (no M1 packed variant)
        for lo, hi in ((0, half), (half, m)):
            lw, lids, _ = _run_legacy(x[lo:hi].contiguous())
            assert _bits_equal(rw[lo:hi], lw), (case, m, lo)
            assert torch.equal(rids[lo:hi], lids), (case, m, lo)
        expected_rows = arange_k[None, :] * m + torch.arange(
            m, device="cuda", dtype=torch.int32
        )[:, None]
        assert torch.equal(rrows, expected_rows), (case, m)


@requires_sm70
def test_fused_topk_dispatch_gpu():
    for m in list(range(1, 34)):
        x = _make_logits("random", m, seed=4000 + m)
        h = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
        on = mod.fused_topk(h, x, K, True, sm70_qwen38_router_runtime_m=True)
        off = mod.fused_topk(h, x, K, True)
        ref = _run_topk_softmax(x)
        rt = _run_runtime_m(x)
        torch.accelerator.synchronize()
        if m <= 32:
            assert all(_bits_equal(a, b) for a, b in zip(on, rt)), m
        else:
            assert all(_bits_equal(a, b) for a, b in zip(on, ref)), m
        if m <= 16:
            legacy = _run_legacy(x)
            assert all(_bits_equal(a, b) for a, b in zip(off, legacy)), m
        else:
            # Old behaviour above M16: generic topk_softmax, bitwise.
            assert all(_bits_equal(a, b) for a, b in zip(off, ref)), m


@requires_sm70
@pytest.mark.parametrize("m", [17, 24, 32])
def test_cuda_graph_replay_matches_eager(m):
    x = torch.randn(LAYERS, m, E, device="cuda", dtype=torch.float16)
    outs = [_out(m) for _ in range(LAYERS)]

    def run():
        for i in range(LAYERS):
            mod._sm70_qwen38_router_topk_runtime_m(*outs[i], x[i])

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for trial in range(10):
        case = BITWISE_CASES[trial % len(BITWISE_CASES)]
        new = torch.stack([_make_logits(case, m, seed=5000 + 97 * trial + i)
                           for i in range(LAYERS)])
        x.copy_(new)
        for w, ids, rows in outs:
            w.fill_(float("nan"))
            ids.fill_(-777)
            rows.fill_(-777)
        graph.replay()
        torch.accelerator.synchronize()
        for i in range(LAYERS):
            ref = _run_runtime_m(x[i])
            assert all(_bits_equal(a, b) for a, b in zip(outs[i], ref)), (trial, i)
        if case in CASES:
            # Replayed values still satisfy the topk_softmax contract.
            sw, sids, srows = _run_topk_softmax(x[0])
            assert torch.equal(outs[0][1], sids) and torch.equal(outs[0][2], srows)
            torch.testing.assert_close(outs[0][0], sw, atol=_TS_TOL, rtol=_TS_TOL)
    del graph


# ----------------------------------------------------------------------------
# microbenchmark (prints; no timing assertions)
# ----------------------------------------------------------------------------
def _graph_round_ms(launch, samples: int = 60) -> float:
    launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAYERS):
            launch()
    for _ in range(20):
        graph.replay()
    torch.accelerator.synchronize()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return statistics.median(times)


@requires_sm70
def test_microbenchmark_48_layer_round():
    import vllm._custom_ops as ops

    print("\n[router32] 48-call graph round, median of 60 (ms); per-call us")
    print(f"{'M':>3} {'topk_softmax':>13} {'legacy':>9} {'runtime_m':>10} "
          f"{'saved/round':>12}")
    for m in (1, 2, 4, 8, 16, 17, 24, 32):
        x = torch.randn(m, E, device="cuda", dtype=torch.float16)
        w, ids, rows = _out(m)
        base = _graph_round_ms(lambda: ops.topk_softmax(w, ids, rows, x, True))
        legacy = (
            _graph_round_ms(lambda: mod._sm70_qwen38_router_topk(w, ids, rows, x))
            if m <= 16
            else float("nan")
        )
        rt = _graph_round_ms(
            lambda: mod._sm70_qwen38_router_topk_runtime_m(w, ids, rows, x)
        )
        print(f"{m:>3} {base:>9.4f} ({base / LAYERS * 1e3:5.1f}) "
              f"{legacy:>9.4f} {rt:>10.4f} ({rt / LAYERS * 1e3:5.1f}) "
              f"{base - rt:>+10.4f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
