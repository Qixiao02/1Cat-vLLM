# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_MOE_GROUPED32 / SX_OPT_MOE_GROUPED_MASK through the production MoE.

Hardware: 1 x V100 (SM70) with ~3 GB free (one TP4-local Qwen3.8 MoE layer,
synthetic NVFP4 weights: E512, hidden 2560, local intermediate 160, top-k 10).
No test needs 4 GPUs. Needs the image's vllm._C plus the v2 ops (rebuilt _C,
$SX_OPT_MOE_GROUPED32_LIBRARY, or an on-the-fly sidecar build via nvcc).

The layer is built by ModelOptNvFp4SM70MoEMethod.process_weights_after_loading
and run through apply(), with the deployment environment
(VLLM_SM70_NVFP4_MOE_GROUPED_DECODE=1, TUNE_MAX_TOKENS=240, persist32 on) and a
pure-decode forward context whose metadata carries a query_start_loc view of a
persistent buffer (runner semantics: tail = live token count).

  /opt/venv/bin/python -m pytest -q sx_tests/b2-moe-m32/test_grouped32_moe_apply.py
  /opt/venv/bin/python -m pytest -q -s ... -k microbench

Asserts:
  * load time: B17-B32 admitted, masking on, split 8, metadata sized 320
    routes, v2 ops cached on the layer.
  * apply() at M in {17,20,24,32}: every row torch.equal to apply() at M16 on
    the same rows, both through the old op (mask off) and v2 (mask on).
  * apply() with live < M (M8/M16/M24/M32): live rows torch.equal to the
    unmasked call, padded rows exactly zero, NaN padded activations harmless.
  * FULL CUDA graph of apply() at M24/M32: replay with changed x/ids/weights
    and live counts written into the persistent query_start_loc buffer, and
    NaN-poisoned scratch, equals eager apply(); planner metadata of the replay
    matches the live routes only.
  * switches: grouped max tokens 16 -> M24 is exactly the TurboMind route;
    mask off -> M16 uses the old op and still equals v2 on every row.
  * numerics vs the previous M24/M32 route (TurboMind): printed max-abs and
    rel-L2 (same FP32-association class as M16 grouped vs control), bounded.
Prints (microbench): 48 x apply() in one CUDA graph at M24/M32, grouped32
vs the generic TurboMind route (after the production warmup tuning at 240 /
320 slots), masked M24 at live 17/20, and M16 old op vs v2+mask.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS", "240")
os.environ.setdefault("VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE", "0")
os.environ["VLLM_SM70_NVFP4_MOE_GROUPED_DECODE"] = "1"

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import _grouped32_common as common  # noqa: E402
from _grouped32_common import E, H, I, K, LAYERS, graph_ms, routes  # noqa: E402

pytestmark = pytest.mark.skipif(not common.ON_SM70, reason="requires an SM70 GPU")

MAX_BATCHED_TOKENS = 8192
QSL_CAPACITY = 33  # buffer of max_num_reqs + 1 entries (max_num_reqs = 32)


def _moe():
    from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

    return moe


def _fake_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_BATCHED_TOKENS),
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
    common.ensure_v2_ops()
    from vllm import _sm70_ops as sm70_ops
    from vllm import envs

    if not sm70_ops.has_nvfp4_grouped_decode_dispatch():
        pytest.skip("image vllm._C lacks the original grouped decode ops")
    getattr(envs, "disable_envs_cache", lambda: None)()
    moe = _moe()
    moe._grouped_v2_state.clear()
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
        mp.setattr(moe, "_SX_OPT_MOE_GROUPED32", True)
        mp.setattr(moe, "_SX_OPT_MOE_GROUPED_MASK", True)
        mp.setattr(moe, "get_current_vllm_config_or_none", _fake_config)
        method.process_weights_after_loading(layer)
    torch.accelerator.synchronize()
    yield method, layer
    del layer
    torch.accelerator.empty_cache()


# ----------------------------------------------------------------------------
# pure-decode forward context
def _qsl_buffer() -> torch.Tensor:
    return torch.zeros(QSL_CAPACITY + 1, device="cuda", dtype=torch.int32)


def _write_live(buf: torch.Tensor, live: int) -> None:
    """Model-runner semantics: cumsum of live decodes, tail padded to live."""
    host = torch.arange(buf.numel(), dtype=torch.int32).clamp_(max=live)
    buf.copy_(host)


@contextlib.contextmanager
def decode_forward(qsl_view: torch.Tensor | None, tokens: int):
    """One pure-decode forward: QSA/short-conv style metadata share the
    query_start_loc view; GDN-style metadata has none."""
    moe = _moe()
    shared = SimpleNamespace(max_query_len=1, query_start_loc=qsl_view)
    other = SimpleNamespace(max_query_len=1, query_start_loc=qsl_view)
    gdn = SimpleNamespace(
        num_prefills=0, num_prefill_tokens=0, num_decodes=tokens,
        num_decode_tokens=tokens,
    )
    metadata = {"layers.0.qsa": shared, "layers.1.qsa": shared,
                "layers.2.conv": other, "layers.3.gdn": gdn}
    ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe, "is_forward_context_available", lambda: True)
        mp.setattr(moe, "get_forward_context", lambda: ctx)
        yield ctx


def _apply_decode(method, layer, x, w, ids, live=None):
    """apply() inside a pure-decode forward whose tail says `live` rows."""
    m = x.shape[0]
    buf = _qsl_buffer()
    _write_live(buf, m if live is None else live)
    with decode_forward(buf[: m + 1], m):
        return method.apply(layer, x, w, ids, None, None).clone()


def _apply_plain(method, layer, x, w, ids):
    """apply() outside a forward context: the pre-existing (non-grouped) route."""
    return method.apply(layer, x, w, ids, None, None).clone()


@contextlib.contextmanager
def layer_attrs(layer, **values):
    old = {k: getattr(layer, k) for k in values}
    try:
        for k, v in values.items():
            setattr(layer, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(layer, k, v)


def _poison(layer):
    for name in ("_nvfp4_sm70_output", "_nvfp4_sm70_intermediate",
                 "_nvfp4_sm70_sorted_output", "_nvfp4_sm70_gate_up",
                 "_nvfp4_sm70_permuted_input"):
        getattr(layer, name).fill_(float("nan"))
    for name in ("_nvfp4_grouped_rows", "_nvfp4_grouped_experts",
                 "_nvfp4_grouped_sizes", "_nvfp4_grouped_total"):
        getattr(layer, name).fill_(-9999)


# ----------------------------------------------------------------------------
def test_load_time_attributes(qwen38_layer):
    _, layer = qwen38_layer
    assert layer.sm70_nvfp4_grouped_decode
    assert layer.sm70_nvfp4_grouped_max_tokens == 32
    assert layer.sm70_nvfp4_grouped_mask
    assert layer.sm70_nvfp4_grouped32_split == 8
    assert layer._nvfp4_grouped_v2_ops is not None
    assert tuple(layer._nvfp4_grouped_rows.shape) == (320, 8)
    assert layer._nvfp4_grouped_experts.numel() == 320
    assert layer._nvfp4_grouped_sizes.numel() == 320
    assert layer.sm70_nvfp4_persistent_max_tokens == 32


@pytest.mark.parametrize("kind", ["random", "pool113", "shared10", "reversed"])
@pytest.mark.parametrize("m", [17, 20, 24, 32])
def test_apply_rows_equal_m16(qwen38_layer, m, kind):
    method, layer = qwen38_layer
    x, w, ids = routes(kind, m, seed=50 * m + len(kind))
    _poison(layer)
    got = _apply_decode(method, layer, x, w, ids)
    assert torch.isfinite(got).all()
    for lo, hi in ((0, 16), (m - 16, m)):
        xs, ws, idss = (t[lo:hi].contiguous() for t in (x, w, ids))
        v2_ref = _apply_decode(method, layer, xs, ws, idss)  # M16, v2 + mask
        with layer_attrs(layer, sm70_nvfp4_grouped_mask=False):
            old_ref = _apply_decode(method, layer, xs, ws, idss)  # M16 old op
        torch.accelerator.synchronize()
        assert torch.equal(got[lo:hi], old_ref), (m, kind, lo)
        assert torch.equal(got[lo:hi], v2_ref), (m, kind, lo)
    # repeat determinism through apply()
    again = _apply_decode(method, layer, x, w, ids)
    assert torch.equal(got, again)


@pytest.mark.parametrize(
    "m,lives", [(8, [5]), (16, [9, 15]), (24, [17, 20, 23]), (32, [25, 31])]
)
def test_apply_masking(qwen38_layer, m, lives):
    method, layer = qwen38_layer
    x, w, ids = routes("pool113", m, seed=300 + m)
    full = _apply_decode(method, layer, x, w, ids)
    for live in lives:
        xp = x.clone()
        xp[live:] = float("nan")
        _poison(layer)
        got = _apply_decode(method, layer, xp, w, ids, live=live)
        torch.accelerator.synchronize()
        assert torch.equal(got[:live], full[:live]), (m, live)
        assert torch.equal(got[live:], torch.zeros_like(got[live:])), (m, live)


@pytest.mark.parametrize("m", [24, 32])
def test_apply_full_graph_replay(qwen38_layer, m):
    method, layer = qwen38_layer
    x0, w0, ids0 = routes("pool113", m, seed=7000 + m)
    sx, sw, sids = x0.clone(), w0.clone(), ids0.clone()
    buf = _qsl_buffer()
    _write_live(buf, m)
    view = buf[: m + 1]
    with decode_forward(view, m):
        method.apply(layer, sx, sw, sids, None, None)  # warm (eager)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with decode_forward(view, m), torch.cuda.graph(graph):
        out = method.apply(layer, sx, sw, sids, None, None)
    assert out.data_ptr() == layer._nvfp4_sm70_output.data_ptr()
    trials = [("random", m), ("pool113", 17), ("shared10", m - 1),
              ("clustered", 20), ("reversed", 1), ("pool113", m)]
    for trial, (kind, live) in enumerate(trials):
        x, w, ids = routes(kind, m, seed=7100 + 31 * trial + m)
        sx.copy_(x)
        sw.copy_(w)
        sids.copy_(ids)
        _write_live(buf, live)
        _poison(layer)
        graph.replay()
        torch.accelerator.synchronize()
        got = out.clone()
        groups = common.expected_plan(ids.view(-1).tolist(), live * K)
        assert int(layer._nvfp4_grouped_total.item()) == len(groups), trial
        ref = _apply_decode(method, layer, x, w, ids)  # eager, unmasked
        torch.accelerator.synchronize()
        assert torch.equal(got[:live], ref[:live]), (m, trial, kind, live)
        assert torch.equal(got[live:], torch.zeros_like(got[live:])), (m, trial)
    del graph


def test_switches_restore_previous_routes(qwen38_layer):
    method, layer = qwen38_layer
    x, w, ids = routes("pool113", 24, seed=424)
    with layer_attrs(layer, sm70_nvfp4_grouped_max_tokens=16):
        tm_in_context = _apply_decode(method, layer, x, w, ids)
    tm_plain = _apply_plain(method, layer, x, w, ids)
    torch.accelerator.synchronize()
    assert torch.equal(tm_in_context, tm_plain)  # exactly the TurboMind route
    x16, w16, ids16 = routes("pool113", 16, seed=416)
    with layer_attrs(layer, sm70_nvfp4_grouped_mask=False):
        old = _apply_decode(method, layer, x16, w16, ids16, live=9)
    new = _apply_decode(method, layer, x16, w16, ids16, live=9)
    torch.accelerator.synchronize()
    assert torch.equal(old[:9], new[:9])  # old op computes padded rows too
    assert torch.equal(new[9:], torch.zeros_like(new[9:]))


@pytest.mark.parametrize("m", [16, 24, 32])
def test_numerics_vs_previous_route(qwen38_layer, m):
    """Grouped vs the route M used before (TurboMind at 24/32, direct QPN at
    16): FP32 association only. Printed; bounded loosely."""
    method, layer = qwen38_layer
    worst = 0.0
    for kind in ("random", "pool113", "shared10"):
        x, w, ids = routes(kind, m, seed=800 + m + len(kind))
        grouped = _apply_decode(method, layer, x, w, ids).float()
        with layer_attrs(layer, sm70_nvfp4_grouped_max_tokens=16):
            previous = _apply_plain(method, layer, x, w, ids).float()
        diff = (grouped - previous).abs()
        rel = float(diff.norm() / previous.norm().clamp_min(1e-30))
        worst = max(worst, rel)
        print(f"\n[grouped32] M{m} {kind}: max_abs {float(diff.max()):.3e} "
              f"rel_l2 {rel:.3e} (|out| max {float(previous.abs().max()):.3e})")
        assert torch.isfinite(grouped).all()
    # FP16-ULP-level association differences only; a wrong expert/row would
    # give rel_l2 ~ 1.
    assert worst < 5e-3, worst


# ----------------------------------------------------------------------------
def _warm_turbomind(layer, widths):
    try:
        from vllm.model_executor.warmup.awq_sm70_warmup import (
            _warmup_nvfp4_moe_decode_layers,
        )

        with torch.inference_mode():
            _warmup_nvfp4_moe_decode_layers([layer], list(widths))
        torch.accelerator.synchronize()
        return True
    except Exception as exc:  # noqa: BLE001 - benchmark context only
        print(f"[grouped32] TurboMind warmup skipped: {exc}")
        return False


def _graph_round(method, layer, m, live=None, **attrs):
    x, w, ids = routes("pool113", m, seed=11000 + m)
    buf = _qsl_buffer()
    _write_live(buf, m if live is None else live)
    view = buf[: m + 1]
    with layer_attrs(layer, **attrs):

        def fn():
            with decode_forward(view, m):
                for _ in range(LAYERS):
                    method.apply(layer, x, w, ids, None, None)

        return graph_ms(fn)


def test_microbench(qwen38_layer):
    method, layer = qwen38_layer
    tuned = _warm_turbomind(layer, (16, 24, 32))
    print(f"\n[grouped32] 48 x apply() in one CUDA graph, median of 60 (ms); "
          f"pool113 routes; TurboMind warmup {'on' if tuned else 'off'}")
    for m in (24, 32):
        new = _graph_round(method, layer, m)
        old = _graph_round(method, layer, m, sm70_nvfp4_grouped_max_tokens=16)
        print(f"  M{m}: grouped32 {new:.3f}  TurboMind {old:.3f}  "
              f"saved {old - new:+.3f} ms/step ({(old - new) / old * 100:+.1f}%)")
    full = _graph_round(method, layer, 24)
    for live in (17, 20):
        masked = _graph_round(method, layer, 24, live=live)
        unmasked = _graph_round(method, layer, 24, live=live,
                                sm70_nvfp4_grouped_mask=False)
        print(f"  M24 live {live}: masked {masked:.3f}  unmasked {unmasked:.3f}  "
              f"(full live 24: {full:.3f})")
    old16 = _graph_round(method, layer, 16, sm70_nvfp4_grouped_mask=False)
    new16 = _graph_round(method, layer, 16)
    old16_live9 = _graph_round(method, layer, 16, live=9,
                               sm70_nvfp4_grouped_mask=False)
    new16_live9 = _graph_round(method, layer, 16, live=9)
    print(f"  M16: old op {old16:.3f}  v2 {new16:.3f}  |  live 9: old "
          f"{old16_live9:.3f}  v2 masked {new16_live9:.3f}")
