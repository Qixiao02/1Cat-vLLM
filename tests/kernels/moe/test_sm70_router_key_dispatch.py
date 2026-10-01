# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route guards for the lossless FP16 E512/K10 sort-key specialization.

Ported from upstream 1Cat 8a99ccb4e / 1ef9f45a5. Fork deviation: the packed
key for M2..16 and the top-16 selection are admitted only for routers built
in the SM70 Qwen3.8 native-MTP lane (FusedTopKRouter passes their
``sm70_qwen38_router_key_args``); a launch without them keeps the packed key
at M1 only, as before, and selects top-16 keys only with an explicit
VLLM_SM70_MTP_ROUTER_TOP16=1 (upstream's meaning).
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import fused_topk_router as mod

LANE_KEY_ARGS = (16, False)


@pytest.mark.parametrize("launcher", ["constexpr_m", "runtime_m"])
@pytest.mark.parametrize("lane", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 5, 10, 16, 17])
def test_packed_half_key_preserves_dtype_and_batch_guards(
    monkeypatch, launcher, lane, dtype, rows
):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, kwargs))

            return launch

    monkeypatch.delenv("VLLM_SM70_MTP_ROUTER_TOP16", raising=False)
    if launcher == "constexpr_m":
        monkeypatch.setattr(mod, "_sm70_qwen38_router_topk_kernel", Kernel())
        run = mod._sm70_qwen38_router_topk
    else:
        monkeypatch.setattr(mod, "_sm70_qwen38_router_topk_runtime_m_kernel", Kernel())
        run = mod._sm70_qwen38_router_topk_runtime_m
    x = torch.empty(rows, 512, dtype=dtype)
    weights = torch.empty(rows, 10, dtype=torch.float32)
    ids = torch.empty(rows, 10, dtype=torch.int32)
    run(weights, ids, torch.empty_like(ids), x, *(LANE_KEY_ARGS if lane else ()))
    assert len(calls) == 1
    grid, kwargs = calls[0]
    assert grid == (rows,)
    max_rows = 16 if lane else 1
    assert kwargs["PACKED_HALF_KEY"] == (rows <= max_rows and dtype == torch.float16)
    assert kwargs["SELECT_TOP16"] is False
    assert kwargs["num_warps"] == 8  # Keep the FP32 normalization reduction.


@pytest.mark.parametrize("lane", [False, True])
@pytest.mark.parametrize("explicit", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 4, 5, 8, 10, 15, 16])
def test_top16_selection_guards(monkeypatch, lane, explicit, dtype, rows):
    """Top-16 selection: FP16 M5/M10 only, always with the packed key."""
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append(kwargs)

    monkeypatch.setattr(mod, "_sm70_qwen38_router_topk_kernel", Kernel())
    monkeypatch.setattr(mod.envs, "VLLM_SM70_MTP_ROUTER_TOP16", bool(explicit))
    x = torch.empty(rows, 512, dtype=dtype)
    weights = torch.empty(rows, 10, dtype=torch.float32)
    ids = torch.empty(rows, 10, dtype=torch.int32)
    # A lane router passes its decision; other launches read the env (None).
    select = (True if explicit is None else explicit) if lane else None
    key_args = (16, select) if lane else ()
    mod._sm70_qwen38_router_topk(weights, ids, torch.empty_like(ids), x, *key_args)
    (kwargs,) = calls
    wanted = bool(lane and select or not lane and explicit)
    top16 = wanted and dtype == torch.float16 and rows in (5, 10)
    assert kwargs["SELECT_TOP16"] is top16
    if top16:
        assert kwargs["PACKED_HALF_KEY"]


@pytest.mark.parametrize("rows", [5, 10])
@pytest.mark.parametrize("select_top16", [False, True])
@pytest.mark.parametrize("lane", [False, True])
def test_mtp_public_route_preserves_graph_weights_ids_and_source_rows(
    monkeypatch, rows, select_top16, lane
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    monkeypatch.setenv("VLLM_SM70_QWEN38_ROUTER_TOPK", "1")
    monkeypatch.setattr(mod.envs, "VLLM_SM70_MTP_ROUTER_TOP16", select_top16)
    x = torch.zeros(rows, 512, device="cuda", dtype=torch.float16)
    hidden = torch.zeros(rows, 2560, device="cuda", dtype=torch.float16)
    ref = (
        torch.empty(rows, 10, device="cuda", dtype=torch.float32),
        torch.empty(rows, 10, device="cuda", dtype=torch.int32),
        torch.empty(rows, 10, device="cuda", dtype=torch.int32),
    )

    def control():
        mod._sm70_qwen38_router_topk_kernel[(rows,)](
            x,
            *ref,
            E=512,
            K=10,
            M=rows,
            BLOCK_E=512,
            PACKED_HALF_KEY=False,
            num_warps=8,
        )

    def candidate():
        if not lane:
            return mod.fused_topk(hidden, x, 10, True)
        # A lane router: runtime-M kernel, packed key M<=16, its top-16 choice.
        return mod.fused_topk(
            hidden,
            x,
            10,
            True,
            sm70_qwen38_router_runtime_m=True,
            sm70_qwen38_router_key_args=(16, select_top16),
        )

    for _ in range(3):
        control()
        candidate()
    torch.accelerator.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    with torch.cuda.graph(graphs[0]):
        control()
    with torch.cuda.graph(graphs[1]):
        actual = candidate()
    for replay in range(12):
        x.normal_(0, (0.001, 0.1, 1.0, 30.0)[replay % 4])
        if replay == 0:
            x.zero_()
            x[:, ::2] = -0.0
        elif replay == 1:
            x.copy_((torch.arange(512, device="cuda") % 7).half())
        elif replay == 2:
            x[:, 0] = float("nan")
        elif replay == 3:
            x.fill_(-float("inf"))
        for result in actual:
            result.fill_(-777)
        for graph in graphs:
            graph.replay()
        for got, expected in zip(actual, ref):
            assert torch.equal(got.view(torch.int32), expected.view(torch.int32))
