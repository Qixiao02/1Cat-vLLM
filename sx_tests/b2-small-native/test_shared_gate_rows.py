# SPDX-License-Identifier: Apache-2.0
"""design_4 [MR6]: exact fused shared-expert gate for M token rows (grid = M).

GPU: ONE V100 (SM70). Needs the rebuilt _C.abi3.so (qwen38_shared_gate_exact_out
accepting (M, 2560)) and the patched vllm/model_executor/models/qwen2_moe.py
installed into the imported vllm.

  /opt/venv/bin/python -m pytest -q -s sx_tests/b2-small-native/test_shared_gate_rows.py
  (SX_TEST_QUICK=1 shrinks the M ladder / weight count; SX_TEST_NO_BENCH=1
  skips the microbenchmark; SX_TEST_MODEL_DIR=<checkpoint dir> selects the
  real gate weights, default /models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4, and
  synthetic weights are used when it is absent.)

Asserts (bitwise = torch.equal on the int16 view, so NaN/Inf/-0 count):
  * every row of the multi-row launch == the M == 1 launch (grid = 1, the
    original contract) on that row alone, for M in {1,2,3,4,8,16,17,20,24,32},
    all real shared_expert_gate weights (48 layers) or synthetic weights,
    activations randn*{0.25,1,3} + special rows (+-0, FP16 subnormals,
    +-65504, tiny), random shared-expert outputs;
  * stale NaN/+-Inf padded rows (FULL-graph padding, e.g. 17 real rows of
    24) never change the bits of the real rows;
  * the gate of each row matches a torch emulation of the single-row kernel
    (exact FP32 FMA chain + shfl_down tree + 8-warp tree + FP16 linear) to
    within 1 FP16 ulp (the kernel uses __expf);
  * CUDA-graph replay with changing activations/outputs == eager per-row M1;
  * Qwen2MoeMLP.forward: SX rows gate -> fused op, bitwise equal to per-row
    M1; the capability probe returns True on the rebuilt extension;
Reports max |fused - unfused (F.linear + sigmoid + mul)| per M (the old
M > 1 path; ULP-level by design) and a CUDA-graph microbenchmark (48 calls =
the 48 MoE layers of one decode step, distinct weights/activations per call)
of the unfused path vs the fused rows kernel at M in {1,2,4,8,16,24,32}.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b2_common as C  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")

SCALES = (0.25, 1.0, 3.0)
DEFAULT_MODEL_DIR = "/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4"


def ops():
    from vllm import _sm70_ops

    if not _sm70_ops.has_qwen38_shared_gate_exact():
        pytest.fail("qwen38_shared_gate_exact_out is not available in this build")
    return _sm70_ops


def gate(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> None:
    ops().qwen38_shared_gate_exact_out(out, x, w)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_WEIGHTS: list[tuple[str, torch.Tensor]] | None = None


def _real_gate_weights() -> list[tuple[str, torch.Tensor]]:
    model_dir = Path(os.environ.get("SX_TEST_MODEL_DIR", DEFAULT_MODEL_DIR))
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        return []
    from safetensors import safe_open

    weight_map = json.loads(index.read_text())["weight_map"]
    keys = sorted(
        key
        for key in weight_map
        if key.endswith("mlp.shared_expert_gate.weight") and "mtp" not in key
    )
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    result = []
    for file, file_keys in by_file.items():
        with safe_open(str(model_dir / file), framework="pt", device="cpu") as f:
            for key in file_keys:
                tensor = f.get_tensor(key)
                if tuple(tensor.shape) != (1, C.HIDDEN):
                    continue
                result.append((key, tensor.to(torch.float16).cuda().contiguous()))
    return sorted(result)


def _synthetic_gate_weights(count: int) -> list[tuple[str, torch.Tensor]]:
    result = []
    for i in range(count):
        gen = torch.Generator(device="cuda").manual_seed(1000 + i)
        scale = (0.01, 0.02, 0.05, 0.2)[i % 4]
        w = (torch.randn((1, C.HIDDEN), generator=gen, device="cuda") * scale).half()
        flat = w.view(-1)
        flat[:: 97 + i] = 0.0
        flat[5 :: 211 + i] = -0.0
        flat[7 :: 301 + i] = 2.0**-24
        result.append((f"synthetic{i}", w.contiguous()))
    return result


def gate_weights() -> list[tuple[str, torch.Tensor]]:
    global _WEIGHTS
    if _WEIGHTS is None:
        weights = _real_gate_weights()
        if weights:
            print(f"\n[b2-gate] {len(weights)} real shared_expert_gate weights")
        else:
            print("\n[b2-gate] checkpoint not found; synthetic gate weights")
            weights = _synthetic_gate_weights(48)
        _WEIGHTS = weights
    limit = 6 if C.quick() else len(_WEIGHTS)
    step = max(1, len(_WEIGHTS) // limit)
    return _WEIGHTS[::step][:limit]


def _special_row(kind: int, k: int, gen: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    if kind == 0:
        return torch.zeros(k, device=dev)
    if kind == 1:
        return torch.full((k,), -0.0, device=dev)
    sign = torch.where(torch.rand(k, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    if kind == 2:
        mant = torch.randint(0, 1024, (k,), generator=gen, device=dev).float()
        return sign * mant * 2.0**-24
    if kind == 3:
        row = torch.randn(k, generator=gen, device=dev)
        row[::97] = 65504.0 * sign[::97]
        row[1::193] = -0.0
        return row
    return sign * 1e-3


def make_rows(m: int, scale: float, seed: int, special: str = "mixed") -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, C.HIDDEN), generator=gen, device="cuda") * scale
    for r in range(m):
        if special == "all":
            x[r] = _special_row(r % 5, C.HIDDEN, gen)
        elif special == "mixed" and r % 5 == 4:
            x[r] = _special_row((r // 5) % 5, C.HIDDEN, gen)
    return x.to(torch.float16).contiguous()


def input_cases(m: int, seed: int) -> list[tuple[str, torch.Tensor]]:
    cases = [(f"randn*{s}", make_rows(m, s, seed + i)) for i, s in enumerate(SCALES)]
    cases.append(("special", make_rows(m, 1.0, seed + 97, special="all")))
    return cases


def make_out(m: int, seed: int) -> torch.Tensor:
    """Shared-expert down-projection output the gate multiplies in place."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = (torch.randn((m, C.HIDDEN), generator=gen, device="cuda") * 0.5).half()
    out.view(-1)[::331] = -0.0
    return out.contiguous()


def poison_rows(t: torch.Tensor, rows) -> None:
    garbage = torch.full((t.shape[1],), float("nan"), device=t.device)
    garbage[1::3] = float("inf")
    garbage[2::3] = float("-inf")
    for r in rows:
        t[r].copy_(garbage.to(t.dtype))


def fused_rows(out0: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = out0.clone()
    gate(out, x, w)
    return out


def fused_m1_per_row(out0: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    result = out0.clone()
    for r in range(x.shape[0]):
        row = out0[r : r + 1].clone()
        gate(row, x[r : r + 1], w)
        result[r].copy_(row[0])
    return result


def unfused(out0: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The old M > 1 path of Qwen2MoeMLP.forward (F.linear, sigmoid, mul)."""
    return torch.sigmoid(F.linear(x, w)) * out0


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", C.m_values())
@torch.inference_mode()
def test_rows_equal_m1_per_row(m: int) -> None:
    failures = []
    checked = 0
    max_unfused = 0.0
    for wi, (name, w) in enumerate(gate_weights()):
        for case, x in input_cases(m, seed=31 * m + wi):
            out0 = make_out(m, seed=7 * m + wi)
            rows = fused_rows(out0, x, w)
            m1 = fused_m1_per_row(out0, x, w)
            checked += 1
            if not C.bit_equal16(rows, m1):
                failures.append(f"{name} {case}: {C.first_mismatch(rows, m1)}")
            ref = unfused(out0, x, w)
            finite = torch.isfinite(rows.float()) & torch.isfinite(ref.float())
            if bool(finite.any()):
                diff = (rows.float() - ref.float()).abs()[finite].max()
                max_unfused = max(max_unfused, float(diff))
    print(f"\n[b2-gate] M={m}: {checked} cases, max|fused-unfused|={max_unfused:.3e}")
    assert not failures, f"M={m}: {len(failures)}/{checked} mismatches; " + (
        "; ".join(failures[:5])
    )


@pytest.mark.parametrize("m,real", ((17, 17), (20, 17), (24, 17), (24, 20), (32, 24)))
@torch.inference_mode()
def test_rows_poisoned_padding(m: int, real: int) -> None:
    for wi, (name, w) in enumerate(gate_weights()[:4]):
        x = make_rows(m, 1.0, seed=500 + wi)
        out0 = make_out(m, seed=600 + wi)
        clean = fused_rows(out0, x, w)
        x_bad, out_bad = x.clone(), out0.clone()
        poison_rows(x_bad, range(real, m))
        poison_rows(out_bad, range(real, m))
        dirty = fused_rows(out_bad, x_bad, w)
        assert C.bit_equal16(dirty[:real], clean[:real]), f"{name}: " + (
            C.first_mismatch(dirty[:real], clean[:real])
        )


def emulate_gate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Torch emulation of qwen38_shared_gate_exact_kernel's gate (per row).

    FP16 x FP16 products are exact in FP32, so product-then-add in FP32
    reproduces __fmaf_rn bit for bit. The warp trees are emulated with the
    same operand pairs; sigmoid uses expf instead of __expf (so the final
    FP16 gate may differ by 1 ulp).
    """
    m = x.shape[0]
    xf = x.float().view(m, 10, 256)  # index = tid + 256 * item
    wf = w.float().view(1, 10, 256)
    value = torch.zeros((m, 256), device=x.device)
    for item in range(10):
        value = xf[:, item, :] * wf[:, item, :] + value
    value = value.view(m, 8, 32)  # [warp][lane]
    for offset in (16, 8, 4, 2, 1):
        value = value[..., :offset] + value[..., offset : 2 * offset]
    partials = torch.zeros((m, 32), device=x.device)
    partials[:, :8] = value[..., 0]
    for offset in (16, 8, 4, 2, 1):
        partials = partials[:, :offset] + partials[:, offset : 2 * offset]
    linear = partials[:, 0].half().float()
    return (1.0 / (1.0 + torch.exp(-linear))).half()


@pytest.mark.parametrize("m", (1, 2, 8, 24, 32))
@torch.inference_mode()
def test_rows_gate_matches_emulation(m: int) -> None:
    worst = 0
    for wi, (name, w) in enumerate(gate_weights()):
        for case, x in input_cases(m, seed=900 + wi):
            ones = torch.ones((m, C.HIDDEN), dtype=torch.float16, device="cuda")
            got = fused_rows(ones, x, w)
            # With out == 1 every element of a row equals that row's gate.
            assert torch.equal(C.bits16(got), C.bits16(got[:, :1]).expand_as(got))
            got_gate = got[:, 0]
            want = emulate_gate(x, w)
            finite = torch.isfinite(got_gate) & torch.isfinite(want)
            assert torch.equal(torch.isnan(got_gate), torch.isnan(want)), (name, case)
            ulps = (
                got_gate[finite].view(torch.int16).int()
                - want[finite].view(torch.int16).int()
            ).abs()
            if ulps.numel():
                worst = max(worst, int(ulps.max()))
    print(f"\n[b2-gate] M={m}: max gate ulp distance to torch emulation = {worst}")
    assert worst <= 1


@torch.inference_mode()
def test_probe_detects_rows_support() -> None:
    from vllm.model_executor.models import qwen2_moe

    assert qwen2_moe._sx_probe_shared_gate_rows() is True


@torch.inference_mode()
def test_probe_rejects_single_row_extension(monkeypatch) -> None:
    from vllm import _sm70_ops
    from vllm.model_executor.models import qwen2_moe

    def old_extension(out, x, w):
        if tuple(x.shape) != (1, C.HIDDEN):
            raise RuntimeError("expected M1/N1/K2560 tensors")

    def row0_only(out, x, w):
        out[:1].mul_(0.5)

    monkeypatch.setattr(_sm70_ops, "qwen38_shared_gate_exact_out", old_extension)
    assert qwen2_moe._sx_probe_shared_gate_rows() is False
    monkeypatch.setattr(_sm70_ops, "qwen38_shared_gate_exact_out", row0_only)
    assert qwen2_moe._sx_probe_shared_gate_rows() is False


# ---------------------------------------------------------------------------
# CUDA graph
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", (2, 4, 8, 16, 24, 32))
@torch.inference_mode()
def test_rows_graph_replay(m: int) -> None:
    _, w = gate_weights()[0]
    x = torch.zeros((m, C.HIDDEN), dtype=torch.float16, device="cuda")
    out = torch.zeros((m, C.HIDDEN), dtype=torch.float16, device="cuda")
    graph = C.capture(lambda: gate(out, x, w))
    for i in range(16):
        new_x = make_rows(m, SCALES[i % 3], seed=2000 + i, special="mixed")
        new_out = make_out(m, seed=3000 + i)
        expected = fused_m1_per_row(new_out, new_x, w)
        x.copy_(new_x)
        out.copy_(new_out)
        graph.replay()
        torch.cuda.synchronize()
        assert C.bit_equal16(out, expected), f"replay {i} M={m}: " + (
            C.first_mismatch(out, expected)
        )


# ---------------------------------------------------------------------------
# Model-level dispatch (Qwen2MoeMLP.forward with the real op)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", (1, 2, 8, 24, 32, 33))
@pytest.mark.parametrize("rows_enabled", (True, False))
@torch.inference_mode()
def test_forward_uses_rows_gate(m: int, rows_enabled: bool, monkeypatch) -> None:
    from vllm.model_executor.models import qwen2_moe

    monkeypatch.setattr(
        qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda label, idx, t: t
    )
    _, w = gate_weights()[0]
    x = make_rows(m, 1.0, seed=4000 + m)
    down = make_out(m, seed=5000 + m)
    calls = []

    class _Gate:
        """ReplicatedLinear stand-in: (F.linear(x, w), None) plus .weight."""

        weight = w

        def __call__(self, inp):
            calls.append("unfused")
            return F.linear(inp, w), None

    layer = NS(
        layer_idx=0,
        _sm70_exact_shared_expert_gate=True,
        _sx_shared_gate_rows=rows_enabled,
        gate_up_proj=NS(forward_fused_silu_and_mul=lambda t: t),
        down_proj=lambda t: (down.clone(), None),
        expert_gate=_Gate(),
    )
    result = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    torch.cuda.synchronize()
    fused_expected = m == 1 or (rows_enabled and m <= 32)
    assert calls == ([] if fused_expected else ["unfused"]), (m, rows_enabled, calls)
    if fused_expected:
        expected = fused_m1_per_row(down, x, w)
        assert C.bit_equal16(result, expected), C.first_mismatch(result, expected)
    else:
        expected = unfused(down, x, w)
        assert C.bit_equal16(result, expected), C.first_mismatch(result, expected)


# ---------------------------------------------------------------------------
# Microbenchmark
# ---------------------------------------------------------------------------
@pytest.mark.skipif(C.no_bench(), reason="SX_TEST_NO_BENCH")
@torch.inference_mode()
def test_bench_shared_gate_rows() -> None:
    calls = 48  # MoE layers per decode step
    weights = [w for _, w in gate_weights()]
    table = []
    for m in C.BENCH_M:
        xs = [make_rows(m, 1.0, seed=6000 + i) for i in range(calls)]
        outs_a = [make_out(m, seed=7000 + i) for i in range(calls)]
        outs_b = [o.clone() for o in outs_a]
        results: list[torch.Tensor | None] = [None] * calls

        def run_unfused(i):
            w = weights[i % len(weights)]
            results[i] = torch.sigmoid(F.linear(xs[i], w)) * outs_a[i]

        def run_fused(i):
            gate(outs_b[i], xs[i], weights[i % len(weights)])

        result = C.bench_graphs({"unfused": run_unfused, "fused": run_fused}, calls=calls)
        saving_ms = (result["unfused"] - result["fused"]) * calls / 1000.0
        table.append(
            [
                m,
                f"{result['unfused']:.2f}",
                f"{result['fused']:.2f}",
                f"{result['unfused'] / result['fused']:.2f}x",
                f"{saving_ms:.3f}",
            ]
        )
        del xs, outs_a, outs_b, results
    C.print_table(
        "Shared-expert gate, us/call (graph of 48 calls, median; unfused = old "
        "M > 1 path: F.linear + sigmoid + mul; runs on the shared-expert "
        "overlap stream in production)",
        ["M", "unfused us", "fused us", "speedup", "ms/step service saved"],
        table,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
