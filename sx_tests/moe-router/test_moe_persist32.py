# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_MOE_PERSIST32 (design_1 [C8]) and SX_OPT_MOE_EAGER_IOTA (design_3 [P5].4).

Hardware: 1 x V100 (SM70) with ~3 GB free for the GPU tests (one TP4-local
Qwen3.8 MoE layer with synthetic NVFP4 weights: E512, hidden 2560, local
intermediate 160, top-k 10). No test needs 4 GPUs. CPU tests run anywhere
vllm imports.

The layer is built through the production ModelOptNvFp4SM70MoEMethod
process_weights_after_loading and exercised through the production apply().
The environment mirrors the deployment: VLLM_SM70_NVFP4_MOE_GROUPED_DECODE=1
(when the native op exists), VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS=240, raw
scales off, all other flags at their defaults.

Run inside the deployed image (repo root = overlay source tree):
  /opt/venv/bin/python -m pytest -q sx_tests/moe-router/test_moe_persist32.py
  /opt/venv/bin/python sx_tests/moe-router/test_moe_persist32.py  # + prints

Asserts:
  * CPU gates: capacity 32 / shared arange only for the Qwen3.8 TP4 contract
    without DBO and without speculative decoding (MTP etc. keep 18 / None).
  * persistent arange views (token_expert_indices, compact_offsets, dense
    expert ids) and the shared eager arange are never written by any call or
    graph replay; a narrower call after a wider one still equals eager.
  * load time: persistent capacity 32, warmup/tuning contract
    (sm70_nvfp4_graph_safe_max_tokens) still 18, buffer shapes sized for 32,
    shared eager arange sized max_num_batched_tokens*10+1.
  * apply() at M in {17,19,20,24,25,31,32}: persistent buffers (capacity 32)
    == eager per-call buffers, torch.equal, for random / clustered /
    all-rows-share-10-experts / reversed routes; repeat-determinism checked;
    persistent scratch poisoned with NaN / -1 before the persistent call.
  * FULL CUDA graph of apply() at M in {19, 24, 32}: replay with changed x,
    ids, weights and NaN-poisoned persistent scratch == eager-buffer apply()
    outside the graph, torch.equal.
  * eager widths M in {33, 48, 127, 128, 784}: shared-arange path == per-call
    torch.arange path, torch.equal; the shared arange is never written.
Prints (-s or script mode):
  * memory: persistent bytes per layer at capacity 18 vs 32 and x48 layers.
  * graph microbenchmark: 48 sequential apply() calls captured in one graph
    at M24 / M32, persistent (new) vs eager-buffers-in-graph (old),
    CUDA events, median of 60 replays.
  * eager microbenchmark: host enqueue time and GPU time per apply() at
    M24 (capacity 32 vs 18) and M33 / M784 (shared arange on vs off).
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS", "240")
os.environ.setdefault("VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE", "0")
os.environ.setdefault("VLLM_SM70_NVFP4_MOE_GROUPED_DECODE", "1")

import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm import envs  # noqa: E402
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe  # noqa: E402

E, H, I, K = 512, 2560, 160, 10
LAYERS = 48
MAX_BATCHED_TOKENS = 8192

_ON_SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
requires_sm70 = pytest.mark.skipif(not _ON_SM70, reason="requires an SM70 GPU")

_SCRATCH_FLOAT = (
    "_nvfp4_sm70_output",
    "_nvfp4_sm70_permuted_input",
    "_nvfp4_sm70_gate_up",
    "_nvfp4_sm70_intermediate",
    "_nvfp4_sm70_sorted_output",
)
_SCRATCH_INT = (
    "_nvfp4_sm70_input_row_indices",
    "_nvfp4_sm70_expert_offsets",
    "_nvfp4_sm70_expert_offsets64",
    "_nvfp4_sm70_inv_permuted_idx",
    "_nvfp4_sm70_topk_ids",
    "_nvfp4_sm70_permuted_idx",
    "_nvfp4_sm70_permuted_experts_id",
    "_nvfp4_sm70_sorted_row_idx",
    "_nvfp4_sm70_topk_ids_for_sort",
    "_nvfp4_sm70_active_expert_ids",
)


def _fake_config(max_tokens=MAX_BATCHED_TOKENS, ubatching=False, spec=None):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        parallel_config=SimpleNamespace(use_ubatching=ubatching),
        speculative_config=spec,
    )


def _moe_cfg(tp=4, hidden=H, inter=I, experts=E, top_k=K):
    return SimpleNamespace(
        num_experts=experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size_per_partition=inter,
        tp_size=tp,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )


# ----------------------------------------------------------------------------
# CPU tests
# ----------------------------------------------------------------------------
def _contract_layer(tp=4, experts=E, hidden=H, inter=I, top_k=K):
    return SimpleNamespace(
        moe_config=SimpleNamespace(tp_size=tp),
        sm70_nvfp4_num_experts=experts,
        sm70_nvfp4_hidden_size=hidden,
        sm70_nvfp4_intermediate_size=inter,
        sm70_nvfp4_top_k=top_k,
    )


def test_persistent_capacity_gate_cpu(monkeypatch):
    monkeypatch.setattr(moe, "_SX_OPT_MOE_PERSIST32", True)
    monkeypatch.setattr(moe, "get_current_vllm_config_or_none", lambda: None)
    assert moe._persistent_max_tokens_for(_contract_layer()) == 32
    monkeypatch.setattr(
        moe, "get_current_vllm_config_or_none", lambda: _fake_config()
    )
    assert moe._persistent_max_tokens_for(_contract_layer()) == 32
    # Other contracts / TP sizes keep the original capacity.
    for layer in (
        _contract_layer(tp=2, inter=320),
        _contract_layer(tp=1, inter=640),
        _contract_layer(experts=256, hidden=2048, inter=128, top_k=8),  # 35B
        _contract_layer(tp=8, experts=288, hidden=4096, inter=256, top_k=8),  # GLM
    ):
        assert moe._persistent_max_tokens_for(layer) == 18
    monkeypatch.setattr(
        moe, "get_current_vllm_config_or_none", lambda: _fake_config(ubatching=True)
    )
    assert moe._persistent_max_tokens_for(_contract_layer()) == 18  # DBO
    monkeypatch.setattr(
        moe,
        "get_current_vllm_config_or_none",
        lambda: _fake_config(spec=SimpleNamespace(method="mtp")),
    )
    assert moe._persistent_max_tokens_for(_contract_layer()) == 18  # MTP/spec
    monkeypatch.setattr(moe, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(moe, "_SX_OPT_MOE_PERSIST32", False)
    assert moe._persistent_max_tokens_for(_contract_layer()) == 18  # switch off


def test_get_buffers_threshold_cpu(monkeypatch):
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    calls = []
    monkeypatch.setattr(
        moe.ModelOptNvFp4SM70MoEMethod,
        "_persistent_buffers",
        staticmethod(lambda layer, m: calls.append(("p", m))),
    )
    monkeypatch.setattr(
        moe.ModelOptNvFp4SM70MoEMethod,
        "_eager_buffers",
        staticmethod(lambda layer, m, idx: calls.append(("e", m))),
    )
    layer = SimpleNamespace(sm70_nvfp4_persistent_max_tokens=32)
    for m in (1, 18, 19, 24, 32, 33, 784):
        method._get_buffers(layer, m, False)
    assert calls == [("p", 1), ("p", 18), ("p", 19), ("p", 24), ("p", 32),
                     ("e", 33), ("e", 784)]
    calls.clear()
    legacy = SimpleNamespace()  # attribute absent -> original 18
    for m in (18, 19, 24):
        method._get_buffers(legacy, m, False)
    assert calls == [("p", 18), ("e", 19), ("e", 24)]


def test_eager_iota_buffers_cpu(monkeypatch):
    monkeypatch.setattr(moe, "_SX_OPT_MOE_EAGER_IOTA", True)
    monkeypatch.setattr(
        moe, "get_current_vllm_config_or_none", lambda: _fake_config(max_tokens=64)
    )
    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        lambda slots, experts: 16 * slots + experts,
        raising=False,
    )
    monkeypatch.setattr(moe, "_qwen38_eager_iotas", {})
    monkeypatch.setattr(moe, "_sort_workspace_sizes", {})
    layer = _contract_layer()
    layer.global_num_experts = E
    layer.w13_tm_weight = torch.empty(1)
    layer._nvfp4_sm70_dense_expert_ids = torch.arange(E, dtype=torch.int32)
    cpu0 = torch.device("cpu", 0)  # explicit index: no accelerator lookup
    iota = moe._get_qwen38_eager_iota(layer, cpu0)
    assert iota is not None and iota.numel() == 64 * K + 1
    assert torch.equal(iota, torch.arange(64 * K + 1, dtype=torch.int32))
    # Not the Qwen3.8 TP4 contract -> old path.
    assert moe._get_qwen38_eager_iota(_contract_layer(tp=2), cpu0) is None
    # Speculative decoding -> old path; unknown token budget -> old path.
    monkeypatch.setattr(
        moe,
        "get_current_vllm_config_or_none",
        lambda: _fake_config(max_tokens=64, spec=SimpleNamespace(method="mtp")),
    )
    assert moe._get_qwen38_eager_iota(layer, cpu0) is None
    monkeypatch.setattr(moe, "get_current_vllm_config_or_none", lambda: None)
    assert moe._get_qwen38_eager_iota(layer, cpu0) is None
    monkeypatch.setattr(
        moe, "get_current_vllm_config_or_none", lambda: _fake_config(max_tokens=64)
    )
    assert moe._get_qwen38_eager_iota(layer, cpu0) is iota  # per-device reuse

    eager = moe.ModelOptNvFp4SM70MoEMethod._eager_buffers
    for m in (5, 9, 40, 64, 65):
        slots = m * K
        layer._nvfp4_sm70_eager_iota = None
        old = eager(layer, m, False)
        layer._nvfp4_sm70_eager_iota = iota
        new = eager(layer, m, False)
        assert set(old) == set(new)
        for name in old:
            assert old[name].shape == new[name].shape, (m, name)
            assert old[name].dtype == new[name].dtype, (m, name)
        for name in ("token_expert_indices", "compact_offsets", "dense_expert_ids"):
            assert torch.equal(old[name], new[name]), (m, name)
        assert new["sort_workspace"].numel() == 16 * slots + E
        shares = new["token_expert_indices"].data_ptr() == iota.data_ptr()
        assert shares == (slots + 1 <= iota.numel()), m
        compact_shares = new["compact_offsets"].data_ptr() == iota.data_ptr()
        assert compact_shares == (slots + 1 <= iota.numel() and slots > 80), m
    assert torch.equal(iota, torch.arange(64 * K + 1, dtype=torch.int32))


# ----------------------------------------------------------------------------
# GPU layer fixture
# ----------------------------------------------------------------------------
def _synthetic_tensors(device, seed=20260926):
    gen = torch.Generator(device=device).manual_seed(seed)

    def fp8_scales(*shape):
        return (torch.rand(*shape, generator=gen, device=device) + 0.5).to(
            torch.float8_e4m3fn
        )

    return dict(
        w13_weight=torch.randint(
            0, 256, (E, 2 * I, H // 2), generator=gen, device=device,
            dtype=torch.uint8,
        ),
        w13_weight_scale=fp8_scales(E, 2 * I, H // 16),
        w13_weight_scale_2=torch.full((E, 2), 0.02, device=device),
        w2_weight=torch.randint(
            0, 256, (E, H, I // 2), generator=gen, device=device, dtype=torch.uint8
        ),
        w2_weight_scale=fp8_scales(E, H, I // 16),
        w2_weight_scale_2=torch.full((E,), 0.02, device=device),
    )


@pytest.fixture(scope="module")
def qwen38_layer():
    if not _ON_SM70:
        pytest.skip("requires an SM70 GPU")
    from vllm import _sm70_ops as sm70_ops

    if not sm70_ops.has_nvfp4_grouped_decode_dispatch():
        os.environ["VLLM_SM70_NVFP4_MOE_GROUPED_DECODE"] = "0"
    getattr(envs, "disable_envs_cache", lambda: None)()
    cfg = _moe_cfg()
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    method.moe = cfg
    layer = SimpleNamespace(
        moe_config=cfg,
        local_num_experts=E,
        global_num_experts=E,
        activation=moe.MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
        swiglu_limit=None,
        w13_input_scale=None,
        w2_input_scale=None,
        **_synthetic_tensors(torch.device("cuda")),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe, "_SX_OPT_MOE_PERSIST32", True)
        mp.setattr(moe, "_SX_OPT_MOE_EAGER_IOTA", True)
        mp.setattr(moe, "get_current_vllm_config_or_none", lambda: _fake_config())
        method.process_weights_after_loading(layer)
    torch.accelerator.synchronize()
    yield method, layer
    del layer
    torch.accelerator.empty_cache()


def _routes(kind: str, m: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(m, E, generator=gen, device="cuda")
    if kind == "clustered":  # realistic cross-row reuse: 40-expert pool
        pool = torch.randperm(E, generator=gen, device="cuda")[:40]
        mask = torch.full((E,), -1e4, device="cuda")
        mask[pool] = 0.0
        scores = scores + mask
    vals, ids = torch.topk(scores, K, dim=-1)
    if kind == "shared10":  # every row routes to the same 10 experts
        base = torch.tensor([3, 77, 150, 151, 222, 300, 301, 402, 480, 511],
                            device="cuda")
        ids = torch.stack([base.roll(r) for r in range(m)])
    elif kind == "reversed":
        ids, _ = torch.sort(ids, dim=-1, descending=True)
    weights = torch.softmax(vals.float(), dim=-1).contiguous()
    x = torch.randn(m, H, generator=gen, device="cuda").half().contiguous()
    return x, weights, ids.to(torch.int32).contiguous()


def _assert_read_only_aranges(layer):
    """Persistent arange views are shared by every width 1..32 and never
    rewritten per call: a write at M19..32 would corrupt later M1..18 steps."""
    slots = layer._nvfp4_sm70_token_expert_indices.numel()
    ref = torch.arange(slots + 1, device="cuda", dtype=torch.int32)
    assert torch.equal(layer._nvfp4_sm70_token_expert_indices.flatten(), ref[:-1])
    assert torch.equal(layer._nvfp4_sm70_compact_offsets, ref)
    assert torch.equal(
        layer._nvfp4_sm70_dense_expert_ids,
        torch.arange(E, device="cuda", dtype=torch.int32),
    )
    iota = layer._nvfp4_sm70_eager_iota
    if iota is not None:
        assert torch.equal(
            iota, torch.arange(iota.numel(), device="cuda", dtype=torch.int32)
        )


def _poison(layer):
    for name in _SCRATCH_FLOAT:
        getattr(layer, name).fill_(float("nan"))
    for name in _SCRATCH_INT:
        getattr(layer, name).fill_(-1)
    layer._nvfp4_sm70_sort_workspace.random_(-128, 127)


def _apply(method, layer, x, w, ids, capacity):
    """capacity 32: new persistent route; 0: exact old eager route
    (per-call buffers and per-call torch.arange, shared arange disabled)."""
    iota = layer._nvfp4_sm70_eager_iota
    layer.sm70_nvfp4_persistent_max_tokens = capacity
    if capacity == 0:
        layer._nvfp4_sm70_eager_iota = None
    try:
        return method.apply(layer, x, w, ids, None, None).clone()
    finally:
        layer.sm70_nvfp4_persistent_max_tokens = 32
        layer._nvfp4_sm70_eager_iota = iota


# ----------------------------------------------------------------------------
# GPU tests
# ----------------------------------------------------------------------------
@requires_sm70
def test_load_time_capacity(qwen38_layer):
    _, layer = qwen38_layer
    assert layer.sm70_nvfp4_persistent_max_tokens == 32
    assert layer.sm70_nvfp4_graph_safe_max_tokens == 18  # warmup unchanged
    assert tuple(layer._nvfp4_sm70_output.shape) == (32, H)
    assert tuple(layer._nvfp4_sm70_permuted_input.shape) == (320, H)
    assert tuple(layer._nvfp4_sm70_sorted_output.shape) == (320, H)
    assert tuple(layer._nvfp4_sm70_token_expert_indices.shape) == (32, K)
    assert torch.equal(
        layer._nvfp4_sm70_token_expert_indices.flatten(),
        torch.arange(320, device="cuda", dtype=torch.int32),
    )
    for m in range(1, 33):
        need = moe._moe_permute_sort_workspace_size(m * K, E)
        assert layer._nvfp4_sm70_sort_workspace.numel() >= need, m
    iota = layer._nvfp4_sm70_eager_iota
    assert iota is not None and iota.numel() == MAX_BATCHED_TOKENS * K + 1


@requires_sm70
@pytest.mark.parametrize("kind", ["random", "clustered", "shared10", "reversed"])
@pytest.mark.parametrize("m", [17, 19, 20, 24, 25, 31, 32])
def test_persistent_equals_eager(qwen38_layer, kind, m):
    method, layer = qwen38_layer
    x, w, ids = _routes(kind, m, seed=100 * m + len(kind))
    _apply(method, layer, x, w, ids, 0)  # warm TurboMind dispatch caches
    _apply(method, layer, x, w, ids, 32)
    eager_a = _apply(method, layer, x, w, ids, 0)
    _poison(layer)
    persist_a = _apply(method, layer, x, w, ids, 32)
    eager_b = _apply(method, layer, x, w, ids, 0)
    _poison(layer)
    persist_b = _apply(method, layer, x, w, ids, 32)
    torch.accelerator.synchronize()
    assert torch.isfinite(eager_a).all()
    assert torch.equal(eager_a, eager_b), "eager path not deterministic"
    assert torch.equal(persist_a, eager_a), (kind, m)
    assert torch.equal(persist_b, eager_a), (kind, m)
    _assert_read_only_aranges(layer)
    # A narrower persistent call after a wider one (stale rows beyond M in
    # every persistent buffer) still equals the eager route.
    if m > 17:
        xs, ws, idss = x[:17].contiguous(), w[:17].contiguous(), ids[:17].contiguous()
        narrow = _apply(method, layer, xs, ws, idss, 32)
        narrow_ref = _apply(method, layer, xs, ws, idss, 0)
        torch.accelerator.synchronize()
        assert torch.equal(narrow, narrow_ref), (kind, m)


@requires_sm70
@pytest.mark.parametrize("m", [19, 24, 32])
def test_graph_replay_persistent_equals_eager(qwen38_layer, m):
    method, layer = qwen38_layer
    x0, w0, ids0 = _routes("random", m, seed=7000 + m)
    sx, sw, sids = x0.clone(), w0.clone(), ids0.clone()
    method.apply(layer, sx, sw, sids, None, None)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = method.apply(layer, sx, sw, sids, None, None)
    assert out.data_ptr() == layer._nvfp4_sm70_output.data_ptr()  # persistent
    for trial, kind in enumerate(
        ("random", "clustered", "shared10", "reversed", "random", "clustered")
    ):
        x, w, ids = _routes(kind, m, seed=7100 + 31 * trial + m)
        sx.copy_(x)
        sw.copy_(w)
        sids.copy_(ids)
        _poison(layer)
        graph.replay()
        got = out.clone()
        ref = _apply(method, layer, sx, sw, sids, 0)
        torch.accelerator.synchronize()
        assert torch.equal(got, ref), (m, trial, kind)
    _assert_read_only_aranges(layer)
    del graph


@requires_sm70
@pytest.mark.parametrize("m", [33, 48, 127, 128, 784])
def test_eager_iota_equals_arange(qwen38_layer, m):
    method, layer = qwen38_layer
    iota = layer._nvfp4_sm70_eager_iota
    assert iota is not None
    x, w, ids = _routes("random", m, seed=9000 + m)
    try:
        layer._nvfp4_sm70_eager_iota = None
        method.apply(layer, x, w, ids, None, None)  # warm
        old = method.apply(layer, x, w, ids, None, None).clone()
        layer._nvfp4_sm70_eager_iota = iota
        new = method.apply(layer, x, w, ids, None, None).clone()
        torch.accelerator.synchronize()
    finally:
        layer._nvfp4_sm70_eager_iota = iota
    assert torch.isfinite(old).all()
    assert torch.equal(new, old), m
    assert torch.equal(
        iota, torch.arange(iota.numel(), device="cuda", dtype=torch.int32)
    ), "shared eager arange was written"


# ----------------------------------------------------------------------------
# memory estimate + microbenchmarks (prints; no timing assertions)
# ----------------------------------------------------------------------------
def _persistent_bytes(ns) -> int:
    skip = {"_nvfp4_sm70_eager_iota", "_nvfp4_sm70_dense_expert_ids"}
    total = 0
    for name, value in vars(ns).items():
        if (
            name.startswith("_nvfp4_sm70_")
            and name not in skip
            and isinstance(value, torch.Tensor)
        ):
            total += value.numel() * value.element_size()
    return total


@requires_sm70
def test_memory_footprint_estimate(qwen38_layer):
    method, layer = qwen38_layer
    sizes = {}
    for capacity in (18, 32):
        ns = SimpleNamespace(
            w13_tm_weight=layer.w13_tm_weight,
            sm70_nvfp4_top_k=K,
            sm70_nvfp4_num_experts=E,
            sm70_nvfp4_hidden_size=H,
            sm70_nvfp4_intermediate_size=I,
            global_num_experts=E,
            sm70_nvfp4_persistent_max_tokens=capacity,
        )
        torch.accelerator.synchronize()
        before = torch.cuda.memory_allocated()
        method._allocate_graph_safe_decode_buffers(ns)
        torch.accelerator.synchronize()
        sizes[capacity] = (_persistent_bytes(ns), torch.cuda.memory_allocated() - before)
        del ns
    delta = sizes[32][0] - sizes[18][0]
    per_slot = 5120 + 51200 + 6400 + 3200 + 51200 + 10 * 40  # 117,520 B/token
    print(
        "\n[persist32] persistent decode buffers per layer: "
        f"cap18 {sizes[18][0] / 2**20:.3f} MiB (alloc {sizes[18][1] / 2**20:.3f}), "
        f"cap32 {sizes[32][0] / 2**20:.3f} MiB (alloc {sizes[32][1] / 2**20:.3f}); "
        f"delta {delta / 2**20:.3f} MiB/layer, x{LAYERS} layers = "
        f"{delta * LAYERS / 1e6:.1f} MB per rank "
        f"(analytic 14 x {per_slot} B x {LAYERS} = "
        f"{14 * per_slot * LAYERS / 1e6:.1f} MB); shared eager arange "
        f"{layer._nvfp4_sm70_eager_iota.numel() * 4 / 1024:.0f} KiB per device"
    )
    assert abs(delta - 14 * per_slot) <= 64 * 1024  # sort workspace slack


def _graph_round_ms(method, layer, m, capacity, samples=60):
    """capacity 32: new; 18: old (eager buffers + aranges inside the graph)."""
    x, w, ids = _routes("clustered", m, seed=11000 + m)
    iota = layer._nvfp4_sm70_eager_iota
    layer.sm70_nvfp4_persistent_max_tokens = capacity
    if capacity == 18:
        layer._nvfp4_sm70_eager_iota = None
    try:
        method.apply(layer, x, w, ids, None, None)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(LAYERS):
                method.apply(layer, x, w, ids, None, None)
    finally:
        layer.sm70_nvfp4_persistent_max_tokens = 32
        layer._nvfp4_sm70_eager_iota = iota
    for _ in range(5):
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
    torch.accelerator.empty_cache()
    return statistics.median(times)


def _eager_call_times(fn, calls=20, blocks=7):
    host, device = [], []
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    for _ in range(blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        t1 = time.perf_counter()
        end.record()
        end.synchronize()
        host.append((t1 - t0) / calls * 1e3)
        device.append(start.elapsed_time(end) / calls)
    return statistics.median(host), statistics.median(device)


@requires_sm70
def test_microbenchmarks(qwen38_layer):
    method, layer = qwen38_layer
    print("\n[persist32] 48 x apply() in one CUDA graph, median of 60 (ms)")
    for m in (24, 32):
        new = _graph_round_ms(method, layer, m, 32)
        old = _graph_round_ms(method, layer, m, 18)
        print(f"  M{m}: persistent {new:.3f}  eager-buffers-in-graph {old:.3f}  "
              f"saved {old - new:+.3f} ms/round")
    print("[persist32] eager apply(): host enqueue ms/call, GPU ms/call")
    x, w, ids = _routes("clustered", 24, seed=12024)
    iota = layer._nvfp4_sm70_eager_iota
    for capacity in (32, 18):
        layer.sm70_nvfp4_persistent_max_tokens = capacity
        if capacity == 18:
            layer._nvfp4_sm70_eager_iota = None
        try:
            hst, dev = _eager_call_times(
                lambda: method.apply(layer, x, w, ids, None, None)
            )
        finally:
            layer.sm70_nvfp4_persistent_max_tokens = 32
            layer._nvfp4_sm70_eager_iota = iota
        print(f"  M24 capacity {capacity}: host {hst:.3f}  gpu {dev:.3f}")
    for m in (33, 784):
        x, w, ids = _routes("random", m, seed=13000 + m)
        for label, value in (("shared arange", iota), ("per-call arange", None)):
            layer._nvfp4_sm70_eager_iota = value
            try:
                hst, dev = _eager_call_times(
                    lambda: method.apply(layer, x, w, ids, None, None)
                )
            finally:
                layer._nvfp4_sm70_eager_iota = iota
            print(f"  M{m} {label}: host {hst:.3f}  gpu {dev:.3f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
