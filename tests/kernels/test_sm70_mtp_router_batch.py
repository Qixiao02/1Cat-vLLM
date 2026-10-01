# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP router packing and graph replay preserve the original FP16 logits.

Ported from upstream 1Cat main@d30469863 (3b7365925). In this tree the route
is admitted only inside the FULL verify capture of an installed native-MTP
lane (sm70_fp16_gemv.py, "SX MTP batch routes"), and at M5 the SX_OPT_ROWS
router kernel keeps the width unless SX_OPT_MTP_BATCH_OVER_ROWS=1.

    /opt/venv/bin/python -m pytest -q tests/kernels/test_sm70_mtp_router_batch.py
"""

from contextlib import contextmanager

import pytest
import torch

import vllm.envs as envs
from vllm.compilation import sm70_decode_graph as dg
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv

ROLE = "model.layers.0.mlp.gate"


@contextmanager
def _mtp_verify_capture():
    saved = dg.sm70_mtp_lane_installed()
    dg.set_sm70_mtp_lane_installed(True)
    try:
        with dg.sm70_decode_graph_compilation(True):
            yield
    finally:
        dg.set_sm70_mtp_lane_installed(saved)


@pytest.fixture
def mtp_env(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    for name in ("SX_OPT_MTP_ROUTER_BATCH", "SX_OPT_MTP_BATCH_OVER_ROWS"):
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    gemv._sx_mtp_batch_config.cache_clear()
    gemv._sx_rows_config.cache_clear()
    yield monkeypatch
    envs.disable_envs_cache()
    gemv._sx_mtp_batch_config.cache_clear()
    gemv._sx_rows_config.cache_clear()


def test_pack_preserves_every_weight_bit():
    raw = torch.randint(-(2**15), 2**15, (512, 2560), dtype=torch.int16)
    packed = gemv._pack_router_batch_weight(raw.view(torch.float16))
    restored = packed.permute(0, 4, 3, 1, 2, 5).contiguous().view(512, 2560)
    assert torch.equal(restored.view(torch.int16), raw)


def _require_native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")
    import vllm._C  # noqa: F401
    if not hasattr(torch.ops._C, "qwen38_router_batch_sm70_out"):
        pytest.skip("Requires a source build with batch router")
    # The MTP lane's cuBLAS contract that the kernel reproduces.
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation


@pytest.mark.parametrize("rows,over_rows", [(5, "1"), (10, "0"), (10, "1")])
def test_native_and_dispatch_changed_input_graph(mtp_env, rows, over_rows):
    _require_native()
    mtp_env.setenv("SX_OPT_MTP_BATCH_OVER_ROWS", over_rows)
    gemv._sx_mtp_batch_config.cache_clear()
    torch.manual_seed(20260927)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.float16) * 0.03
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    packed = gemv._pack_router_batch_weight(w)
    expected = torch.empty(rows, 512, device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)
    with _mtp_verify_capture():
        tile = gemv._sx_rows_tile(x, w, gemv._sx_role_key(ROLE, (512, 2560)))
        assert gemv._router_batch_runtime_ok(x, packed, tile)  # the batch route

    def run():
        torch.mm(x, w.t(), out=expected)
        torch.ops._C.qwen38_router_batch_sm70_out(actual, x, packed)
        with _mtp_verify_capture():
            return gemv._qwen38_sm70_fp16_gemv(x, w, ROLE, packed)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatched = run()
    for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0, 30.0):
        x.normal_(0, scale)
        actual.fill_(float("nan"))
        dispatched.fill_(float("nan"))
        graph.replay()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(dispatched.view(torch.int16), expected.view(torch.int16))


def test_m5_keeps_the_rows_kernel_by_default(mtp_env):
    """Default precedence: M5 stays on SX_OPT_ROWS (each row == the M1 GEMV)."""
    _require_native()
    torch.manual_seed(20260928)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.float16) * 0.03
    x = torch.randn(5, 2560, device="cuda", dtype=torch.float16)
    packed = gemv._pack_router_batch_weight(w)
    with _mtp_verify_capture():
        dispatched = gemv._qwen38_sm70_fp16_gemv(x, w, ROLE, packed)
    rows = torch.cat(
        [gemv._qwen38_sm70_fp16_gemv(x[r : r + 1].clone(), w, ROLE) for r in range(5)]
    )
    assert torch.equal(dispatched.view(torch.int16), rows.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 15, 20])
def test_other_widths_and_contexts_keep_their_routes(mtp_env, rows):
    _require_native()
    mtp_env.setenv("SX_OPT_MTP_BATCH_OVER_ROWS", "1")
    gemv._sx_mtp_batch_config.cache_clear()
    torch.manual_seed(rows)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.float16) * 0.03
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    packed = gemv._pack_router_batch_weight(w)
    with _mtp_verify_capture():
        assert not gemv._router_batch_runtime_ok(x, packed)
        with_pack = gemv._qwen38_sm70_fp16_gemv(x, w, ROLE, packed)
        without = gemv._qwen38_sm70_fp16_gemv(x, w, ROLE)
    assert torch.equal(with_pack.view(torch.int16), without.view(torch.int16))
    # M5 outside the installed-lane verify capture: never admitted.
    x5 = torch.randn(5, 2560, device="cuda", dtype=torch.float16)
    assert not gemv._router_batch_runtime_ok(x5, packed)
