# SPDX-License-Identifier: Apache-2.0
"""Item 2 of "mtp-port-moe": exact FP16 draft MoE projections (upstream
0930fd3b6 / 6bcffbb79) as the native-MTP lane arms them.

GPU: ONE V100 (SM70). Needs torch.ops._C.sm70_mtp_moe_fp16_out from a vllm._C
rebuilt from this branch. Before a rebuild, SX_TEST_JIT_MTP_MOE_FP16=1 builds
csrc/sm70_turbomind/ops/mtp_moe_fp16_sm70.cu alone into $TORCH_EXTENSIONS_DIR
(needs nvcc) and registers the op next to the installed _C.

  /opt/venv/bin/python -m pytest -q sx_tests/mtp-port-moe/test_draft_moe_fp16_gpu.py
  /opt/venv/bin/python -m pytest -q -s sx_tests/mtp-port-moe/test_draft_moe_fp16_gpu.py -k bench

Knobs of sx_tests/b3-draft/_b3_draft_common.py apply (SX_TEST_MTP_WEIGHTS_DIR
and SX_TEST_MTP_TP_RANK use the checkpoint's real MTP experts).

The lane is reproduced as the drafter builds it: the SX draft tile table
armed (arm_sm70_mtp_draft_moe_tiles) and arm_sm70_mtp_draft_moe_fp16(True)
with VLLM_SM70_MTP_MOE_FP16_EXACT unset.

Asserts:
  * fused_experts (classic path) and TritonExperts.apply (the modular experts
    the drafter runs) at M1 and M5: the native op is called for W13 and W2
    when armed and never when not armed; the whole MoE output is bitwise
    equal to the Triton projections over changing activations (six scales),
    routes and an invalid expert id, also under CUDA-graph replay;
  * every other draft width (2, 3, 4, 8, 10, 15, 20, 40) never calls it;
  * an explicit VLLM_SM70_MTP_MOE_FP16_EXACT=0 keeps the lane off it.
Prints (-s, -k bench): one draft round's MoE (M5 + 3 x M1) in a CUDA graph,
Triton vs native.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

os.environ.pop("VLLM_SM70_MTP_MOE_FP16_EXACT", None)  # lane default

B3_DRAFT = Path(__file__).resolve().parents[1] / "b3-draft"
sys.path.insert(0, str(B3_DRAFT))

import _b3_draft_common as C  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")
REPO = Path(__file__).resolve().parents[2]
OP = "sm70_mtp_moe_fp16_out"
SCALES = (0.0, 0.001, 0.1, 1.0, 3.0, 30.0)


def _ensure_op() -> None:
    import vllm._C  # noqa: F401

    if not C.fm()._SX_OPT_MTP_MOE_FP16_EXACT:
        pytest.skip("SX_OPT_MTP_MOE_FP16_EXACT=0")
    if hasattr(torch.ops._C, OP):
        return
    if os.environ.get("SX_TEST_JIT_MTP_MOE_FP16", "0") in ("", "0"):
        pytest.skip(f"vllm._C lacks {OP}; rebuild from this branch or set "
                    "SX_TEST_JIT_MTP_MOE_FP16=1")
    from torch.utils.cpp_extension import load

    load(
        name="sx_mtp_moe_fp16_sm70",
        sources=[str(REPO / "csrc/sm70_turbomind/ops/mtp_moe_fp16_sm70.cu")],
        extra_cuda_cflags=["-O3", "-std=c++17",
                           "-gencode=arch=compute_70,code=sm_70"],
        is_python_module=False,
    )
    assert hasattr(torch.ops._C, OP)


@pytest.fixture(scope="module")
def experts():
    _ensure_op()
    torch.manual_seed(0)
    w1, w2 = C.experts_for_test()
    yield w1, w2
    del w1, w2
    torch.cuda.empty_cache()


@contextlib.contextmanager
def lane(armed: bool):
    """The drafter's arming in the native-MTP lane (tile table + native op)."""
    mod = C.fm()
    saved = mod._sx_mtp_moe_fp16_exact_armed
    with C.table_ctx():
        try:
            mod.arm_sm70_mtp_draft_moe_fp16(armed)
            assert mod._sx_mtp_moe_fp16_exact_armed == (
                armed
                and mod._SX_OPT_MTP_MOE_FP16_EXACT
                and "VLLM_SM70_MTP_MOE_FP16_EXACT" not in os.environ
            )
            yield
        finally:
            mod._sx_mtp_moe_fp16_exact_armed = saved


@pytest.fixture
def hits(monkeypatch):
    native = getattr(torch.ops._C, OP)
    calls: list[bool] = []

    def tracked(*args):
        calls.append(bool(args[-1]))
        native(*args)

    monkeypatch.setattr(torch.ops._C, OP, tracked)
    return calls


def _inputs(m: int, seed: int, scale: float):
    x = C.make_hidden(m, seed, scale=1.0)
    x = x * scale if scale else torch.zeros_like(x)
    weights, ids = C.make_routing(m, seed)
    ids = ids.to(torch.int32).contiguous()
    ids.view(-1)[0] = -1  # invalid expert slot (writes zeros, as Triton)
    return x.contiguous(), weights.float().contiguous(), ids


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("m", [1, 5])
@torch.inference_mode()
def test_fused_experts_m1_m5_bitwise(experts, hits, m):
    w1, w2 = experts
    for trial, scale in enumerate(SCALES):
        x, weights, ids = _inputs(m, 100 + trial, scale)
        with lane(False):
            hits.clear()
            ref = C.run_moe(x, w1, w2, weights, ids)
            assert hits == []
        with lane(True):
            got = C.run_moe(x, w1, w2, weights, ids)
            assert hits == [False, True], (m, scale)
        torch.accelerator.synchronize()
        assert torch.equal(_bits(got), _bits(ref)), (m, scale)


def _moe_config():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    return FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=1,
        intermediate_size_per_partition=1,
        num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


@pytest.mark.parametrize("m", [1, 5])
@torch.inference_mode()
def test_modular_experts_lane_graph(experts, hits, m):
    """Upstream's test_modular_experts_route_and_graph with the lane arming
    instead of VLLM_SM70_MTP_MOE_FP16_EXACT."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )

    w1, w2 = experts
    mod = TritonExperts(_moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    ids = torch.zeros(m, 10, device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
    shapes = mod.workspace_shapes(m, 320, 2560, 10, 512, 512, None,
                                  MoEActivation.SILU)
    workspaces = [x.new_empty(shape) for shape in shapes[:2]]
    outputs = [torch.empty_like(x) for _ in range(2)]

    def run(arm):
        mod.apply(outputs[arm], x, w1, w2, weights, ids, MoEActivation.SILU,
                  512, None, None, None, *workspaces, None, False)

    graphs = []
    for arm in range(2):
        with lane(bool(arm)):
            hits.clear()
            run(arm)
            assert hits == ([False, True] if arm else [])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(arm)
            graphs.append(graph)
    for scale in SCALES:
        x.normal_(0, scale)
        ids.random_(0, 512)
        ids[0, 0] = -1
        weights.copy_(torch.softmax(torch.randn_like(weights), -1))
        for workspace in workspaces:
            workspace.fill_(float("nan"))
        for output in outputs:
            output.fill_(float("nan"))
        for graph in graphs:
            graph.replay()
        assert torch.equal(_bits(outputs[0]), _bits(outputs[1])), scale


@pytest.mark.parametrize("m", [2, 3, 4, 8, 10, 15, 20, 40])
@torch.inference_mode()
def test_other_draft_widths_keep_triton(experts, hits, m):
    w1, w2 = experts
    x, weights, ids = _inputs(m, 300 + m, 1.0)
    with lane(True):
        C.run_moe(x, w1, w2, weights, ids)
    assert hits == []


@torch.inference_mode()
def test_explicit_env_zero_keeps_lane_off(experts, hits, monkeypatch):
    w1, w2 = experts
    monkeypatch.setenv("VLLM_SM70_MTP_MOE_FP16_EXACT", "0")
    from vllm import envs

    getattr(envs, "disable_envs_cache", lambda: None)()
    x, weights, ids = _inputs(5, 400, 1.0)
    with lane(True):
        C.run_moe(x, w1, w2, weights, ids)
    assert hits == []


@torch.inference_mode()
def test_bench(experts):
    w1, w2 = experts
    rounds = [_inputs(m, 500 + i, 1.0) for i, m in enumerate((5, 1, 1, 1))]

    def one_round():
        for x, weights, ids in rounds:
            C.run_moe(x, w1, w2, weights, ids)

    graphs = {}
    for name, armed in (("triton", False), ("native", True)):
        with lane(armed):
            graphs[name] = C.capture(one_round)
    us = C.bench_alternating(graphs, calls=1)
    print(f"\n[mtp-port-moe] draft MoE per round (M5 + 3 x M1, W13 + W2), "
          f"CUDA graph, median us: Triton {us['triton']:.1f}, native "
          f"{us['native']:.1f} ({us['triton'] - us['native']:+.1f})")
