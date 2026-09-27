# SPDX-License-Identifier: Apache-2.0
"""MTP-8: the drafter's own qkv / o / indexer / router / HC-down projections
and HC modules on the existing checkpoint-FP16 SM70 routes.

GPU: ONE V100 (SM70). Needs the changed mtp.py installed into / bind-mounted
over the imported vllm (sm70_fp16_gemv.py / sm70_fp16_hc.py are unchanged):

  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-draft/test_draft_gemv_gpu.py

Real vLLM ReplicatedLinear layers with the draft prefixes
(mtp.layers.48.*) and the TP4-local shapes are passed through
mtp._sx_install_mtp_draft_fp16_routes, exactly as Qwen4ExpMTP does.

Asserts
  * M = 1 (draft decode at one request with split draft graphs) runs the
    unchanged M=1 row-GEMV kernel: bitwise equal to calling it directly, and
    close to cuBLAS (F.linear);
  * M > 1 outside the dual-compile decode-graph context returns exactly the
    previous F.linear result (bitwise), i.e. draft prefill / padded widths
    are unchanged;
  * inside that context (what the MTP-lane decode wrapper provides) admitted
    widths use the rows kernels: every row bitwise equal to the M=1 kernel;
  * CUDA-graph replay of the M=1 route equals eager;
  * the fused FP16 HC op at M=1 (single GPU: replicated kernels) matches the
    unfused down / SiLU / up / gate-mix chain within FP16 tolerance, and the
    GatedResidual hook admits it once the module is marked;
  * through a real GatedResidual after the install: M=1 close to the previous
    chain, M in {2,5,8,20} bitwise the previous chain.
Reports M=1 route vs cuBLAS (CUDA graph, rotating weights).
"""

from __future__ import annotations

import contextlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_draft_common as C  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")


def _stub_tp_helpers() -> None:
    """Validation fix (b3a): vLLM weight parameters query the TP group even
    with disable_tp=True; this single-process test has no torch.distributed,
    so pin rank 0 / world size 1 for the parameter and linear modules."""
    import vllm.model_executor.parameter as param_mod
    from vllm.model_executor.layers import linear as linear_mod

    for mod in (param_mod, linear_mod):
        for name, fn in (
            ("get_tensor_model_parallel_rank", lambda: 0),
            ("get_tensor_model_parallel_world_size", lambda: 1),
        ):
            if hasattr(mod, name):
                setattr(mod, name, fn)


_stub_tp_helpers()

P = "mtp.layers.48"
ROLES = [
    ("qsa_qkv", f"{P}.self_attn.qkv_proj", 3584, 2560),
    ("qsa_o", f"{P}.self_attn.o_proj", 2560, 1536),
    ("qsa_index", f"{P}.self_attn.indexer.index_qk_proj", 640, 2560),
    ("router", f"{P}.mlp.gate", 512, 2560),
    ("hc_down", f"{P}.attn_hyper_connection.input_mix_weight_down_block_inject", 336, 10240),
]


@pytest.fixture(autouse=True)
def _single_compile_lane(monkeypatch):
    """Tests run the drafter's single-compile semantics unless they opt in."""
    monkeypatch.delenv("VLLM_SM70_QWEN38_DUAL_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    _gemv()._sx_rows_config.cache_clear()
    yield
    _gemv()._sx_rows_config.cache_clear()


def _gemv():
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv

    return sm70_fp16_gemv


def _mtp():
    from vllm.models.qwen4_exp.nvidia import mtp

    return mtp


def _layer(prefix: str, n: int, k: int, seed: int):
    from vllm.model_executor.layers.linear import ReplicatedLinear

    layer = ReplicatedLinear(
        k,
        n,
        bias=False,
        params_dtype=torch.float16,
        prefix=prefix,
        disable_tp=True,
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        layer.weight.copy_(
            (torch.randn((n, k), generator=gen, device="cuda") * 0.05).half()
        )
    return layer


@pytest.fixture(scope="module")
def installed():
    root = torch.nn.Module()
    layers = {}
    for i, (name, prefix, n, k) in enumerate(ROLES):
        layers[name] = _layer(prefix, n, k, 100 + i)
        root.add_module(name, layers[name])
    counts = _mtp()._sx_install_mtp_draft_fp16_routes(root)
    assert counts["gemv"] == len(ROLES), counts
    return layers


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), generator=gen, device="cuda").half()


@contextlib.contextmanager
def _decode_graph_ctx():
    """The dual-compile FULL decode-graph context the MTP-lane wrapper sets."""
    from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation

    gemv = _gemv()
    saved = os.environ.get("VLLM_SM70_QWEN38_DUAL_COMPILE")
    os.environ["VLLM_SM70_QWEN38_DUAL_COMPILE"] = "1"
    gemv._sx_rows_config.cache_clear()
    try:
        with sm70_decode_graph_compilation(True):
            yield
    finally:
        if saved is None:
            os.environ.pop("VLLM_SM70_QWEN38_DUAL_COMPILE", None)
        else:
            os.environ["VLLM_SM70_QWEN38_DUAL_COMPILE"] = saved
        gemv._sx_rows_config.cache_clear()


@pytest.mark.parametrize("name", [r[0] for r in ROLES])
@torch.inference_mode()
def test_m1_uses_gemv_route(installed, name: str) -> None:
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import Qwen38SM70FP16LinearMethod

    layer = installed[name]
    assert type(layer.quant_method) is Qwen38SM70FP16LinearMethod
    for seed in range(4):
        x = _x(1, layer.weight.shape[1], seed)
        out, _ = layer(x)
        direct = _gemv()._qwen38_sm70_fp16_gemv(x, layer.weight, layer.prefix)
        assert C.bit_equal16(out, direct), name
        ref = F.linear(x, layer.weight)
        scale = max(1.0, float(ref.abs().max()))
        torch.testing.assert_close(out.float(), ref.float(), atol=5e-3 * scale, rtol=1e-2)


@pytest.mark.parametrize("name", [r[0] for r in ROLES])
@pytest.mark.parametrize("m", [2, 3, 4, 5, 8, 10, 20, 40, 120])
@torch.inference_mode()
def test_m_gt1_outside_decode_ctx_is_previous(installed, name: str, m: int) -> None:
    layer = installed[name]
    x = _x(m, layer.weight.shape[1], m)
    out, _ = layer(x)
    assert C.bit_equal16(out, F.linear(x, layer.weight)), (name, m)


@pytest.mark.parametrize("name", [r[0] for r in ROLES])
@pytest.mark.parametrize("m", [2, 4, 5, 8])
@torch.inference_mode()
def test_rows_inside_decode_ctx_match_m1(installed, name: str, m: int) -> None:
    gemv = _gemv()
    layer = installed[name]
    x = _x(m, layer.weight.shape[1], 7 * m)
    with _decode_graph_ctx():
        key = gemv._sx_role_key(layer.prefix, tuple(layer.weight.shape))
        admitted = m <= gemv._sx_rows_max_m(key) if key else False
        out, _ = layer(x)
    if not admitted:
        assert C.bit_equal16(out, F.linear(x, layer.weight)), (name, m)
        return
    for r in range(m):
        row = gemv._qwen38_sm70_fp16_gemv(x[r : r + 1].clone(), layer.weight, layer.prefix)
        assert C.bit_equal16(out[r : r + 1], row), (name, m, r)


@torch.inference_mode()
def test_native_mtp_lane_keeps_drafter_route(installed, monkeypatch) -> None:
    """With the batch-3a MTP lane installed (dual-compile target) the drafter
    keeps decode semantics, so its M=1 route stays active; the target's main
    backbone context turns it off (previous F.linear)."""
    from vllm.compilation import sm70_decode_graph as dg

    if not hasattr(dg, "set_sm70_mtp_lane_installed"):
        pytest.skip("batch-3a lane-core sm70_decode_graph.py not mounted")
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")
    previous = dg.sm70_mtp_lane_installed()
    dg.set_sm70_mtp_lane_installed(True)
    try:
        layer = installed["router"]
        x = _x(1, 2560, 3)
        out, _ = layer(x)
        direct = _gemv()._qwen38_sm70_fp16_gemv(x, layer.weight, layer.prefix)
        assert C.bit_equal16(out, direct)
        with dg.sm70_target_main_backbone():
            out_main, _ = layer(x)
        assert C.bit_equal16(out_main, F.linear(x, layer.weight))
    finally:
        dg.set_sm70_mtp_lane_installed(previous)


@torch.inference_mode()
def test_dual_compile_without_lane_is_previous(installed, monkeypatch) -> None:
    """Dual compile on but no lane: drafter semantics are off, so the draft
    projections run the previous F.linear even at M=1."""
    from vllm.compilation import sm70_decode_graph as dg

    if hasattr(dg, "sm70_mtp_lane_installed") and dg.sm70_mtp_lane_installed():
        pytest.skip("lane installed in this process")
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")
    layer = installed["qsa_qkv"]
    x = _x(1, 2560, 4)
    out, _ = layer(x)
    assert C.bit_equal16(out, F.linear(x, layer.weight))


@pytest.mark.parametrize("name", [r[0] for r in ROLES])
@torch.inference_mode()
def test_m1_graph_replay(installed, name: str) -> None:
    layer = installed[name]
    static = _x(1, layer.weight.shape[1], 1)
    holder = {}

    def fn():
        holder["out"], _ = layer(static)

    graph = C.capture(fn)
    for it in range(3):
        new = _x(1, layer.weight.shape[1], 50 + it)
        static.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        eager, _ = layer(new)
        assert C.bit_equal16(holder["out"], eager), (name, it)


@torch.inference_mode()
def test_fused_hc_m1_matches_unfused_chain() -> None:
    import vllm.models.qwen4_exp.nvidia.ops.hc  # noqa: F401  (registers HC ops)
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import (
        maybe_apply_qwen38_sm70_fp16_fused_hc,
    )

    down = _layer(f"{P}.attn_hyper_connection.input_mix_weight_down_block_inject", 336, 10240, 7)
    up = _layer(f"{P}.attn_hyper_connection.input_mix_weight_up", 10240, 320, 8)
    with torch.no_grad():
        down.weight.mul_(0.4)
        up.weight.mul_(0.4)
    for seed in range(3):
        x = _x(1, 10240, 900 + seed)
        block, injection = torch.ops.vllm.qwen38_sm70_fp16_fused_hc(x, down.weight, up.weight)
        down_out = F.linear(x, down.weight)
        lora = torch.ops.vllm.qwen4_exp_hc_silu(down_out[..., :320], 4)
        ref_injection = down_out[..., 320:324]
        gate = F.linear(lora, up.weight)
        ref_block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, 4)
        for got, ref in ((block, ref_block), (injection, ref_injection)):
            scale = max(1.0, float(ref.abs().max()))
            torch.testing.assert_close(got.float(), ref.float(), atol=5e-3 * scale, rtol=2e-2)
        hooked = maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, True)
        assert hooked is not None
        assert C.bit_equal16(hooked[0], block) and C.bit_equal16(hooked[1], injection)
        assert maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, False) is None


def _real_gated_residual(monkeypatch, prefix: str, seed: int):
    """A real Qwen4Exp GatedResidual (combine) HC module, single GPU.

    Its input_mix_weight_up is a TP-aware ReplicatedLinear; the TP helpers are
    stubbed to rank 0 / size 1 so no distributed init is needed.
    """
    from vllm.model_executor.layers import linear as linear_mod
    from vllm.models.qwen4_exp.nvidia.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    monkeypatch.setattr(linear_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=2560,
        params_dtype=torch.float16,
        hc_lowrank=320,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    module = GatedResidual(config, use_combine=True, prefix=prefix).cuda()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(
                (torch.randn(param.shape, generator=gen, device="cuda") * 0.02).to(
                    param.dtype
                )
            )
    return module


@torch.inference_mode()
def test_real_gated_residual_project(monkeypatch) -> None:
    """MTP-8 HC through the real module: the install marks the combine HC and
    puts its down projection on the GEMV route; M = 1 takes the fused FP16 HC
    kernels (close to the previous chain), M > 1 stays bitwise the previous
    down / SiLU / up / gate-mix chain (draft prefill, padded widths)."""
    import vllm.models.qwen4_exp.nvidia.ops.hc  # noqa: F401  (registers HC ops)

    module = _real_gated_residual(monkeypatch, f"{P}.attn_hyper_connection", 11)
    widths = (1, 2, 5, 8, 20)
    xs = {m: _x(m, 10240, 600 + m) * 0.5 for m in widths}
    before = {m: module._project(xs[m]) for m in widths}
    counts = _mtp()._sx_install_mtp_draft_fp16_routes(module)
    assert counts == {"gemv": 1, "hc": 1}, counts
    assert module._sm70_qwen38_fp16_fused_hc
    for m in widths:
        block, injection = module._project(xs[m])
        ref_block, ref_injection = before[m]
        assert injection is not None and ref_injection is not None
        if m == 1:
            for got, ref in ((block, ref_block), (injection, ref_injection)):
                scale = max(1.0, float(ref.abs().max()))
                torch.testing.assert_close(
                    got.float(), ref.float(), atol=5e-3 * scale, rtol=2e-2
                )
        else:
            assert C.bit_equal16(block, ref_block), m
            assert C.bit_equal16(injection, ref_injection), m


@pytest.mark.skipif(C.no_bench(), reason="SX_TEST_NO_BENCH=1")
@torch.inference_mode()
def test_bench_m1_route_vs_cublas() -> None:
    rows = []
    for name, prefix, n, k in ROLES:
        copies = 8
        layers = [_layer(prefix, n, k, 300 + i) for i in range(copies)]
        root = torch.nn.Module()
        for i, lyr in enumerate(layers):
            root.add_module(f"c{i}", lyr)
        _mtp()._sx_install_mtp_draft_fp16_routes(root)
        xs = [_x(1, k, 400 + i) for i in range(copies)]
        calls = 32

        def route():
            for i in range(calls):
                layers[i % copies](xs[i % copies])

        def cublas():
            for i in range(calls):
                F.linear(xs[i % copies], layers[i % copies].weight)

        graphs = {"cublas": C.capture(cublas), "route": C.capture(route)}
        us = C.bench_alternating(graphs, calls)
        rows.append([name, f"{n}x{k}", f"{us['cublas']:.2f}", f"{us['route']:.2f}", f"{us['cublas'] / us['route']:.2f}x"])
        del graphs
        torch.cuda.synchronize()
    C.print_table(
        "Draft M=1 projection: cuBLAS vs SM70 FP16 GEMV route (us/call, CUDA graph)",
        ["role", "N x K", "cuBLAS", "route", "speedup"],
        rows,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
