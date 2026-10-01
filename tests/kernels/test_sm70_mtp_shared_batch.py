# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP shared expert retains all three FP16 rounding boundaries.

Ported from upstream 1Cat main@d30469863 (69eac6d8e). In this tree the batch
projection is admitted only inside the FULL verify capture of an installed
native-MTP lane (sm70_fp16_gemv.py, "SX MTP batch routes"); every other width
and context runs the shared expert's own linear + _C.silu_and_mul. Upstream's
sigmoid/multiply epilogue is not ported (the fork's exact multi-row shared
gate serves those rows), so only the projection is checked here.

    /opt/venv/bin/python -m pytest -q tests/kernels/test_sm70_mtp_shared_batch.py
"""

from contextlib import contextmanager

import pytest
import torch

import vllm.envs as envs
from vllm.compilation import sm70_decode_graph as dg
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as batch


@contextmanager
def _mtp_verify_capture():
    saved = dg.sm70_mtp_lane_installed()
    dg.set_sm70_mtp_lane_installed(True)
    try:
        with dg.sm70_decode_graph_compilation(True):
            yield
    finally:
        dg.set_sm70_mtp_lane_installed(saved)


def _require_native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    import vllm._C  # noqa: F401
    if not hasattr(torch.ops._C, "qwen38_shared_up_batch_sm70_out"):
        pytest.skip("Requires source build with shared expert batch kernels")
    # The MTP lane's cuBLAS contract that the kernel reproduces.
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation


def test_pack_preserves_every_weight_bit():
    raw = torch.randint(-(2**15), 2**15, (320, 2560), dtype=torch.int16)
    packed = batch._pack_shared_batch_weight(raw.view(torch.float16))
    restored = packed.permute(0, 3, 1, 2, 4).contiguous().view(320, 2560)
    assert torch.equal(restored.view(torch.int16), raw)


@pytest.mark.parametrize("rows", [1, 2, 5, 10, 17])
@pytest.mark.parametrize("enabled", [False, True])
def test_changed_input_graph(rows, enabled, monkeypatch):
    _require_native()
    monkeypatch.setenv("SX_OPT_MTP_SHARED_BATCH", str(int(enabled)))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    batch._sx_mtp_batch_config.cache_clear()
    torch.manual_seed(20260927)
    weight = torch.randn(320, 2560, device="cuda", dtype=torch.float16) * 0.03
    packed = batch._pack_shared_batch_weight(weight)
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    raw = x.new_empty((rows, 320))
    expected = x.new_empty((rows, 160))
    with _mtp_verify_capture():
        assert batch._shared_batch_runtime_ok(x, packed) == (
            enabled and rows in (5, 10)
        )

    def run():
        torch.mm(x, weight.t(), out=raw)
        torch.ops._C.silu_and_mul(expected, raw)
        with _mtp_verify_capture():
            return batch._qwen38_sm70_shared_up(x, weight, packed)

    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    try:
        for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0, 30.0):
            x.normal_(0, scale)
            actual.fill_(float("nan"))
            graph.replay()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    finally:
        envs.disable_envs_cache()
        batch._sx_mtp_batch_config.cache_clear()


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 6, 8, 12, 15, 16, 20, 24])
def test_other_widths_match_the_original_compiled_path(rows):
    """Verify widths outside M5/M10 (and M1/M2.. decode) keep their numerics.

    Under decode semantics the hook sends every width through the opaque op
    (the width is a symbolic dimension at trace time); outside M5/M10 it must
    return exactly what the module's own F.linear + _C.silu_and_mul return,
    eagerly and under torch.compile with a dynamic batch dimension.
    """
    _require_native()
    torch.manual_seed(rows)
    weight = torch.randn(320, 2560, device="cuda", dtype=torch.float16) * 0.03
    packed = batch._pack_shared_batch_weight(weight)
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)

    def module_path(x, weight):
        gate_up = torch.nn.functional.linear(x, weight)
        out = gate_up.new_empty((*gate_up.shape[:-1], 160))
        torch.ops._C.silu_and_mul(out, gate_up)
        return out

    with _mtp_verify_capture():
        assert not batch._shared_batch_runtime_ok(x, packed)
        actual = batch._qwen38_sm70_shared_up(x, weight, packed)
    expected = module_path(x, weight)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    try:
        compiled = torch.compile(module_path, dynamic=True)(x, weight)
    except Exception as error:  # noqa: BLE001 - no usable compiler here
        pytest.skip(f"torch.compile unavailable: {error}")
    assert torch.equal(actual.view(torch.int16), compiled.view(torch.int16))


@pytest.mark.parametrize("rows", [5, 10])
def test_native_output_canaries(rows):
    _require_native()
    torch.manual_seed(rows)
    weight = torch.randn(320, 2560, device="cuda", dtype=torch.float16) * 0.03
    packed = batch._pack_shared_batch_weight(weight)
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    storage = x.new_full((rows * 160 + 16,), 17)
    out = storage[8:-8].view(rows, 160)
    partial = x.new_empty((8, rows, 320))
    torch.ops._C.qwen38_shared_up_batch_sm70_out(out, partial, x, packed)
    assert torch.all(storage[:8] == 17) and torch.all(storage[-8:] == 17)
    expected = x.new_empty((rows, 160))
    torch.ops._C.silu_and_mul(expected, torch.mm(x, weight.t()))
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [2, 8, 16])
def test_native_rejects_other_widths(rows):
    _require_native()
    x = torch.zeros(rows, 2560, device="cuda", dtype=torch.float16)
    packed = torch.zeros(10, 160, 2, 32, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="M5/M10"):
        torch.ops._C.qwen38_shared_up_batch_sm70_out(
            x.new_empty((rows, 160)), x.new_empty((8, rows, 320)), x, packed
        )
