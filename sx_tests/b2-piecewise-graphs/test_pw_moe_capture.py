# SPDX-License-Identifier: Apache-2.0
"""PW-1: the routed-MoE prefill routes are CUDA-graph safe at the new
PIECEWISE sizes (never captured before PW-1; M>=128 is the indexed-A route).

Hardware: 1 x V100 (SM70), ~6 GB free. One TP4-local Qwen3.8 MoE layer with
synthetic NVFP4 weights (E512, hidden 2560, local intermediate 160, top-k 10),
built through the production ModelOptNvFp4SM70MoEMethod
process_weights_after_loading and run through the production apply(), i.e.
exactly the code the moe_forward custom op executes inside a PIECEWISE piece.
No test needs 4 GPUs.

  /opt/venv/bin/python -m pytest -q sx_tests/b2-piecewise-graphs/test_pw_moe_capture.py
  /opt/venv/bin/python sx_tests/b2-piecewise-graphs/test_pw_moe_capture.py   # + prints

Environment mirrors the deployment: VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS=240,
VLLM_SM70_NVFP4_MOE_GROUPED_DECODE=1, raw scales off, other flags default.

Asserts:
  * eager apply() at M in {32, 48, 64, 96, 128, 256, 512, 832, 1024} issues no
    PyTorch-visible host synchronisation (torch.cuda.set_sync_debug_mode
    "error").
  * the same apply() captured in a CUDA graph (vLLM pattern: warm-up on the
    capture stream, then capture on it, shared private pool across sizes,
    largest first) replays bitwise equal (FP16 bit patterns) to eager apply() at the
    same M, for 4 routing patterns per size, with the capture-time inputs
    replaced before every replay.
  * padded rows are isolated: a graph at the padded size whose rows beyond
    the real ones hold garbage (random x incl. +-inf/NaN rows, random routes)
    gives real rows bitwise equal to the same graph with different garbage,
    and bitwise equal to eager apply() at the padded size.
  * padded vs unpadded (the numerics change PW-1 introduces vs today's eager
    step at the unpadded M): real rows of the padded graph vs eager apply() at
    the real M are reported (bitwise-equal row fraction, max |diff|) and must
    be finite and within 2e-2 relative.
Prints (-s or script mode): 48 x apply() per "forward" at M in {128, 512,
1024}: eager host enqueue ms and GPU ms vs graph replay host/GPU ms.
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
PW_SIZES = (1024, 832, 512, 256, 128, 112, 96, 64, 48, 32)  # capture order: largest first (112: PAD_CASES)
PAD_CASES = ((25, 32), (100, 112), (474, 512), (800, 832), (1000, 1024))

_ON_SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
requires_sm70 = pytest.mark.skipif(not _ON_SM70, reason="requires an SM70 GPU")


def _fake_config(max_tokens=MAX_BATCHED_TOKENS):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=None,
    )


def _moe_cfg():
    return SimpleNamespace(
        num_experts=E,
        experts_per_token=K,
        hidden_dim=H,
        intermediate_size_per_partition=I,
        tp_size=4,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )


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


def _garbage_tail(x, w, ids, real: int, seed: int, nonfinite: bool):
    """Overwrite rows [real:] like stale padded rows of a PIECEWISE step."""
    if real >= x.shape[0]:
        return
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tail = x.shape[0] - real
    x[real:] = (torch.randn(tail, H, generator=gen, device="cuda") * 50).half()
    if nonfinite:
        x[real] = float("nan")
        if tail > 1:
            x[real + 1] = float("inf")
    ids[real:] = torch.randint(0, E, (tail, K), generator=gen, device="cuda",
                               dtype=torch.int32)
    w[real:] = torch.softmax(torch.randn(tail, K, generator=gen, device="cuda"), -1)


class _Captured:
    """One apply() per graph, captured like CudaGraphManager.capture():
    static inputs, warm-up on the capture stream, capture on the same stream,
    one private pool shared by every size (largest first)."""

    def __init__(self, method, layer, m, stream, pool):
        self.x, self.w, self.ids = _routes("random", m, seed=4242 + m)
        with torch.cuda.stream(stream):
            method.apply(layer, self.x, self.w, self.ids, None, None)  # warm-up
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.out = method.apply(layer, self.x, self.w, self.ids, None, None)

    def run(self, x, w, ids):
        self.x.copy_(x)
        self.w.copy_(w)
        self.ids.copy_(ids)
        self.graph.replay()
        return self.out.clone()


@pytest.fixture(scope="module")
def captured(qwen38_layer):
    method, layer = qwen38_layer
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    pool = torch.cuda.graph_pool_handle()
    graphs = {m: _Captured(method, layer, m, stream, pool) for m in PW_SIZES}
    torch.accelerator.synchronize()
    yield graphs
    graphs.clear()
    torch.accelerator.empty_cache()


def _eager(method, layer, x, w, ids):
    return method.apply(layer, x, w, ids, None, None).clone()


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of FP16 tensors, NaN/Inf safe.

    torch.equal treats NaN != NaN, and garbage padded rows (x * 50 overflows
    the FP16 SwiGLU intermediate) do produce Inf/NaN outputs, so compare the
    raw 16-bit patterns instead.
    """
    return (
        a.shape == b.shape
        and a.dtype == b.dtype == torch.float16
        and torch.equal(
            a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        )
    )


# ----------------------------------------------------------------------------
# GPU tests
# ----------------------------------------------------------------------------
@requires_sm70
@pytest.mark.parametrize("m", sorted(PW_SIZES))
def test_no_host_sync(qwen38_layer, m):
    method, layer = qwen38_layer
    x, w, ids = _routes("random", m, seed=100 + m)
    _eager(method, layer, x, w, ids)  # first call may query/tune on the host
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = method.apply(layer, x, w, ids, None, None)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.accelerator.synchronize()
    assert torch.isfinite(out).all()


@requires_sm70
@pytest.mark.parametrize("m", sorted(PW_SIZES))
def test_graph_replay_equals_eager(qwen38_layer, captured, m):
    method, layer = qwen38_layer
    graph = captured[m]
    for trial, kind in enumerate(("random", "clustered", "shared10", "reversed")):
        x, w, ids = _routes(kind, m, seed=7000 + 31 * trial + m)
        got = graph.run(x, w, ids)
        ref = _eager(method, layer, x, w, ids)
        torch.accelerator.synchronize()
        assert torch.isfinite(ref).all(), (m, kind)
        assert _same_bits(got, ref), (m, kind, (got.float() - ref.float()).abs().max())


@requires_sm70
@pytest.mark.parametrize("real, padded", PAD_CASES)
def test_padded_rows_isolated(qwen38_layer, captured, real, padded):
    method, layer = qwen38_layer
    graph = captured[padded]
    x, w, ids = _routes("clustered", padded, seed=9100 + real)
    xa, wa, ida = x.clone(), w.clone(), ids.clone()
    xb, wb, idb = x.clone(), w.clone(), ids.clone()
    _garbage_tail(xa, wa, ida, real, seed=1, nonfinite=False)
    _garbage_tail(xb, wb, idb, real, seed=2, nonfinite=True)
    out_a = graph.run(xa, wa, ida)
    out_b = graph.run(xb, wb, idb)
    eager_pad = _eager(method, layer, xa, wa, ida)
    eager_real = _eager(
        method, layer, x[:real].contiguous(), w[:real].contiguous(),
        ids[:real].contiguous(),
    )
    torch.accelerator.synchronize()
    # Real rows never depend on the padded rows (row-independent MoE).
    assert _same_bits(out_a[:real], out_b[:real]), (real, padded)
    # Graph == eager at the same padded size, including the garbage rows
    # (bit patterns: those rows may hold Inf/NaN).
    assert _same_bits(out_a, eager_pad), (real, padded)
    # Padded vs today's unpadded eager step: report, bound loosely.
    diff = (out_a[:real].float() - eager_real.float()).abs()
    same_rows = int((diff.amax(dim=1) == 0).sum())
    scale = eager_real.float().abs().max().item()
    print(
        f"\n[PW-1 moe] real {real} -> padded {padded}: bitwise-equal rows "
        f"{same_rows}/{real}, max|diff| {diff.max().item():.3e} "
        f"(max|ref| {scale:.3e})"
    )
    assert torch.isfinite(out_a[:real]).all()
    assert diff.max().item() <= 2e-2 * max(scale, 1e-3), (real, padded)


# ----------------------------------------------------------------------------
# microbenchmark (prints only)
# ----------------------------------------------------------------------------
def _time_eager(fn, reps=5):
    host, gpu = [], []
    for _ in range(reps + 1):
        torch.accelerator.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        end.record()
        end.synchronize()
        host.append((t1 - t0) * 1e3)
        gpu.append(start.elapsed_time(end))
    return statistics.median(host[1:]), statistics.median(gpu[1:])


@requires_sm70
def test_microbenchmark_48_layers(qwen38_layer):
    method, layer = qwen38_layer
    print("\n[PW-1 moe] 48 x apply() (one forward's routed MoE), median of 5")
    for m in (128, 512, 1024):
        x, w, ids = _routes("clustered", m, seed=12000 + m)

        def eager_round():
            for _ in range(LAYERS):
                method.apply(layer, x, w, ids, None, None)

        eh, eg = _time_eager(eager_round)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            eager_round()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            eager_round()
        gh, gg = _time_eager(graph.replay)
        print(
            f"  M{m:5d}: eager host {eh:7.2f} ms  gpu {eg:7.2f} ms | "
            f"graph host {gh:6.3f} ms  gpu {gg:7.2f} ms"
        )
        del graph
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
