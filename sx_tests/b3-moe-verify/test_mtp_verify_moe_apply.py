# SPDX-License-Identifier: Apache-2.0
"""Batch 3a "moe-verify": NVFP4 MoE routes at uniform MTP verify widths.

GPUs: 1 x V100 (SM70), ~4 GB free (two, briefly three, TP4-local Qwen3.8 MoE
layers with synthetic NVFP4 weights: E512, hidden 2560, local intermediate
160, top-k 10).
Needs the image's vllm._C (1.8.0-dev2: grouped v1/v2 ops, QPN M1/MTP5/batch
ops) with this group's files bind-mounted over the installed package.

  /opt/venv/bin/python -m pytest -q sx_tests/b3-moe-verify/test_mtp_verify_moe_apply.py
  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-moe-verify/test_mtp_verify_moe_apply.py -k microbench

Two layers are built by ModelOptNvFp4SM70MoEMethod.process_weights_after_loading
from the same synthetic tensors: one under an MTP-lane config (method "mtp",
k = 4, exact Qwen3.8 TP4 target; the real lane contract runs), one under the
no-MTP config (production: PERSIST32/EAGER_IOTA/GROUPED32/MASK on). The
forward contexts mirror the V2 runner (see _mtp_common.py): uniform verify
metadata (QSA max_query_len = q, PLE/GDN spec counters, one persistent
query_start_loc view of W / q + 1 entries whose tail is the live token count).

Asserts:
  * load time: lane layer stamped q = 5, grouped + masking + v2 ops, MTP5 and
    dynamic QPN defaults, eager iota on, PERSIST32 off (18); the no-MTP layer
    carries no lane admission; route table for k = 4.
  * grouped verify widths (k = 1..7 layouts, W 8..32): every row bitwise
    equal to the no-MTP lane's grouped decode on that row at the same split
    (M8 windows for split 4, M16 windows for split 8) and, for W > 16, the
    whole output bitwise equal to the no-MTP lane at the same width.
  * masking: live < W (multiples of q) -> live rows bitwise equal to the
    unmasked call, padded rows exactly zero, NaN padded activations harmless;
    equal to the no-MTP lane at the same width/live count where one exists.
  * direct routes (W5 MTP5 + batch W2, W3/W6/W7 dynamic QPN): bitwise equal
    to the 1Cat env opt-ins (VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE=1 /
    ..._DYNAMIC_DECODE=1) they default; numerics vs TurboMind printed.
  * FULL CUDA graphs of apply() at W5/10/20/30 (k = 4), W12 (k = 3), W32
    (k = 3): replay with new x/ids/weights and live counts, NaN-poisoned
    scratch, equals eager apply on live rows; padded rows zero.
  * mixed / non-uniform / pure-decode contexts and the switch-off state keep
    exactly the previous (TurboMind) route.
  * SX_OPT_MTP_LANE=0 (lane-wide master): the lane config builds the layer as
    1.8.0-dev2 did (no lane attributes, no v2 ops, no eager iota, capacity 18)
    and every verify width keeps the previous route bitwise (review addition).
Prints (-s): 48 x apply() in one CUDA graph at verify widths 5, 10, 15, 20,
25 (padded to 30 and to 32), 30, 32: dev2 MTP-lane route vs the new route
and the alternatives (1Cat MTP5 with separate W2, grouped at W5, dynamic QPN
at 10/15, split 8 at W10, split 4 above 16, masking off).
"""

from __future__ import annotations

import contextlib
import os
import sys

os.environ.pop("SX_OPT_MTP_LANE", None)  # the lane master must be on here
os.environ.setdefault("VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS", "240")
os.environ.setdefault("VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE", "0")
os.environ["VLLM_SM70_NVFP4_MOE_GROUPED_DECODE"] = "1"
# The lane defaults of SX_OPT_MTP_MOE_DIRECT apply only when these are absent.
for _name in (
    "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE",
    "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE",
):
    os.environ.pop(_name, None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402

sys.path.insert(0, str(C.B2_MOE_DIR))

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.ON_SM70, reason="requires an SM70 GPU")

E, H, I, K, LAYERS = C.E, C.H, C.I, C.K, C.LAYERS


def _moe():
    from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe

    return moe


def _routes(kind, m, seed):
    import _grouped32_common as b2

    return b2.routes(kind, m, seed)


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


def _build(config, lane_master=True):
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router

    moe = _moe()
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
        for name in ("_SX_OPT_MOE_PERSIST32", "_SX_OPT_MOE_EAGER_IOTA",
                     "_SX_OPT_MOE_GROUPED32", "_SX_OPT_MOE_GROUPED_MASK",
                     "_SX_OPT_MTP_MOE_GROUPED", "_SX_OPT_MTP_MOE_DIRECT"):
            mp.setattr(moe, name, True)
        mp.setattr(moe, "_SX_MTP_MOE_GROUPED_MIN_TOKENS", 8)
        mp.setattr(moe, "_SX_MTP_MOE_GROUPED_SPLIT_SMALL", 4)
        mp.setattr(fused_topk_router, "_SX_OPT_MTP_MOE_ROUTES", True)
        if lane_master:
            mp.delenv("SX_OPT_MTP_LANE", raising=False)
        else:
            mp.setenv("SX_OPT_MTP_LANE", "0")
        mp.setattr(moe, "get_current_vllm_config_or_none", lambda: config)
        method.process_weights_after_loading(layer)
    torch.accelerator.synchronize()
    return method, layer


@pytest.fixture(scope="module")
def layers():
    import _grouped32_common as b2

    b2.ensure_v2_ops()
    from vllm import _sm70_ops as sm70_ops
    from vllm import envs

    if not sm70_ops.has_nvfp4_grouped_decode_dispatch():
        pytest.skip("image vllm._C lacks the grouped decode ops")
    getattr(envs, "disable_envs_cache", lambda: None)()
    moe = _moe()
    moe._grouped_v2_state.clear()
    lane_method, lane = _build(C.lane_config(4))
    nomtp_method, nomtp = _build(C.nomtp_config())
    yield SimpleNamespace(lane_method=lane_method, lane=lane,
                          nomtp_method=nomtp_method, nomtp=nomtp)
    del lane, nomtp
    torch.accelerator.empty_cache()


# ----------------------------------------------------------------------------
# forward helpers
def _apply_verify(method, layer, x, w, ids, q, live=None, metadata=None):
    m = x.shape[0]
    buf = C.qsl_buffer()
    C.write_live(buf, m if live is None else live, q)
    meta = metadata or C.verify_metadata(buf[: m // q + 1], m, q)
    with C.forward_context(_moe(), meta):
        return method.apply(layer, x, w, ids, None, None).clone()


def _apply_decode(method, layer, x, w, ids, live=None):
    m = x.shape[0]
    buf = C.qsl_buffer()
    C.write_live(buf, m if live is None else live, 1)
    with C.forward_context(_moe(), C.decode_metadata(buf[: m + 1], m)):
        return method.apply(layer, x, w, ids, None, None).clone()


def _apply_plain(method, layer, x, w, ids):
    """No forward context: the route the 1.8.0-dev2 MTP lane used for every
    verify width (TurboMind generic; static QPN at 2/4/8/16)."""
    return method.apply(layer, x, w, ids, None, None).clone()


@contextlib.contextmanager
def as_q(layer, q):
    """Treat the lane layer as built for k = q - 1 (route logic reads the
    attributes; the MTP5 default only exists for k = 4)."""
    with C.layer_attrs(layer, sx_mtp_verify_q=q,
                       sx_mtp_qpn_mtp5=bool(q == 5 and layer.sx_mtp_qpn_mtp5)):
        yield


def _poison(layer):
    for name in ("_nvfp4_sm70_output", "_nvfp4_sm70_intermediate",
                 "_nvfp4_sm70_sorted_output", "_nvfp4_sm70_gate_up",
                 "_nvfp4_sm70_permuted_input"):
        getattr(layer, name).fill_(float("nan"))
    for name in ("_nvfp4_grouped_rows", "_nvfp4_grouped_experts",
                 "_nvfp4_grouped_sizes", "_nvfp4_grouped_total"):
        getattr(layer, name).fill_(-9999)


def _windows(x, w, ids, start, rows, ref_m, seed):
    """Rows [start, start + rows) padded to ref_m rows with unrelated routes."""
    px, pw, pids = _routes("random", ref_m, seed)
    take = slice(start, start + rows)
    return (
        torch.cat([x[take], px[: ref_m - rows]]).contiguous(),
        torch.cat([w[take], pw[: ref_m - rows]]).contiguous(),
        torch.cat([ids[take], pids[: ref_m - rows]]).contiguous(),
    )


# ----------------------------------------------------------------------------
def test_load_time(layers):
    moe = _moe()
    lane, nomtp = layers.lane, layers.nomtp
    assert lane.sx_mtp_verify_q == 5
    assert lane.sx_mtp_grouped and lane.sm70_nvfp4_grouped_mask
    assert lane.sm70_nvfp4_grouped_max_tokens == 32
    assert lane._nvfp4_grouped_v2_ops is not None
    assert tuple(lane._nvfp4_grouped_rows.shape) == (320, 8)
    assert lane.sx_mtp_qpn_mtp5 and lane.sx_mtp_qpn_dynamic
    assert lane._nvfp4_sm70_eager_iota is not None
    assert lane.sm70_nvfp4_persistent_max_tokens == 18  # PERSIST32 stays off
    assert nomtp.sx_mtp_verify_q == 0 and not nomtp.sx_mtp_grouped
    assert not nomtp.sx_mtp_qpn_mtp5 and not nomtp.sx_mtp_qpn_dynamic
    assert nomtp.sm70_nvfp4_persistent_max_tokens == 32
    table = moe._mtp_verify_route_table(lane)
    print(f"\n[b3-moe] k=4 verify routes: {table}")
    assert table[5].startswith("qpn-mtp5-split4")
    assert table[10] == "grouped-split4" and table[15] == "grouped-split8"
    assert all(table[w] == "grouped-split8" for w in (20, 25, 30))
    assert table[35] == "turbomind"


GROUPED_CASES = [
    (5, 10), (5, 15), (5, 20), (5, 25), (5, 30),
    (4, 8), (4, 12), (4, 16), (4, 24), (4, 32),
    (3, 9), (3, 12), (3, 15), (3, 18), (3, 30),
    (2, 10), (2, 14), (2, 16), (2, 32),
    (6, 12), (7, 14), (7, 28), (8, 8), (8, 24),
]


@pytest.mark.parametrize("kind", ["random", "pool113", "shared10", "reversed"])
@pytest.mark.parametrize("q,width", GROUPED_CASES)
def test_grouped_verify_rows_bitwise_vs_nomtp(layers, q, width, kind):
    moe = _moe()
    lane, nomtp = layers.lane, layers.nomtp
    x, w, ids = _routes(kind, width, seed=97 * width + 13 * q + len(kind))
    with as_q(lane, q):
        route = moe._mtp_verify_route_table(lane)[width]
        got = _apply_verify(layers.lane_method, lane, x, w, ids, q)
        again = _apply_verify(layers.lane_method, lane, x, w, ids, q)
    assert route.startswith("grouped-split"), route
    split = int(route.removeprefix("grouped-split"))
    torch.accelerator.synchronize()
    assert torch.isfinite(got).all()
    assert torch.equal(got, again)  # deterministic
    if width > 16:
        ref = _apply_decode(layers.nomtp_method, nomtp, x, w, ids)
        assert torch.equal(got, ref), (q, width, kind)  # same no-MTP width
    ref_m = 8 if split == 4 else 16
    for start in range(0, width, ref_m):
        rows = min(ref_m, width - start)
        xs, ws, idss = _windows(x, w, ids, start, rows, ref_m, seed=start + width)
        ref = _apply_decode(layers.nomtp_method, nomtp, xs, ws, idss)
        torch.accelerator.synchronize()
        assert torch.equal(got[start:start + rows], ref[:rows]), (
            q, width, kind, start)


@pytest.mark.parametrize(
    "q,width,lives",
    [(5, 10, [5]), (5, 15, [5, 10]), (5, 20, [5, 10, 15]), (5, 30, [25, 20, 5]),
     (4, 32, [24, 28, 4]), (3, 18, [9, 15]), (2, 16, [2, 10])],
)
def test_verify_masking(layers, q, width, lives):
    lane, nomtp = layers.lane, layers.nomtp
    x, w, ids = _routes("pool113", width, seed=300 + width + q)
    with as_q(lane, q):
        full = _apply_verify(layers.lane_method, lane, x, w, ids, q)
        for live in lives:
            xp = x.clone()
            xp[live:] = float("nan")
            _poison(lane)
            got = _apply_verify(layers.lane_method, lane, xp, w, ids, q, live=live)
            torch.accelerator.synchronize()
            assert torch.equal(got[:live], full[:live]), (q, width, live)
            assert torch.equal(got[live:], torch.zeros_like(got[live:])), (
                q, width, live)
            if width > 16 or width in (8, 16):
                ref = _apply_decode(layers.nomtp_method, nomtp, xp, w, ids,
                                    live=live)
                assert torch.equal(got, ref), (q, width, live)


def _optin_envs(**over):
    from vllm import envs

    return C.EnvsProxy(envs, **over)


@pytest.mark.parametrize("q,width", [(5, 5), (3, 3), (3, 6), (6, 6), (7, 7), (2, 6)])
def test_direct_routes_equal_env_optins(layers, q, width):
    moe = _moe()
    lane = layers.lane
    worst = 0.0
    for kind in ("random", "pool113", "shared10"):
        x, w, ids = _routes(kind, width, seed=500 + 7 * width + len(kind))
        with as_q(lane, q):
            route = moe._mtp_verify_route_table(lane)[width]
            got = _apply_verify(layers.lane_method, lane, x, w, ids, q)
            # The same forward with the lane defaults off but 1Cat's global
            # env opt-ins on (what the lane defaults stand for).
            with C.layer_attrs(lane, sx_mtp_qpn_mtp5=False, sx_mtp_qpn_dynamic=False), \
                    pytest.MonkeyPatch.context() as mp:
                mp.setattr(moe, "envs", _optin_envs(
                    VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE=(width == 5),
                    VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=True,
                ))
                optin = _apply_verify(layers.lane_method, lane, x, w, ids, q)
        previous = _apply_plain(layers.lane_method, lane, x, w, ids)  # TurboMind
        torch.accelerator.synchronize()
        assert route.startswith("qpn"), route
        assert torch.equal(got, optin), (q, width, kind)
        diff = (got.float() - previous.float()).abs()
        rel = float(diff.norm() / previous.float().norm().clamp_min(1e-30))
        worst = max(worst, rel)
        print(f"\n[b3-moe] W{width} q{q} {kind} {route} vs TurboMind: max_abs "
              f"{float(diff.max()):.3e} rel_l2 {rel:.3e}")
    assert worst < 5e-3, worst


def test_w5_mtp5_variants(layers):
    """W5: MTP5 + batch W2 (lane default) vs 1Cat's MTP5 alone (separate W2
    + weighted reduce) vs no-MTP QPN M8 rows (informational)."""
    lane, nomtp = layers.lane, layers.nomtp
    x, w, ids = _routes("pool113", 5, seed=555)
    got = _apply_verify(layers.lane_method, lane, x, w, ids, 5)
    with C.layer_attrs(lane, sx_mtp_qpn_dynamic=False):
        mtp5_only = _apply_verify(layers.lane_method, lane, x, w, ids, 5)
    xs, ws, idss = _windows(x, w, ids, 0, 5, 8, seed=8)
    with C.layer_attrs(nomtp, sm70_nvfp4_grouped_decode=False):
        qpn8 = _apply_decode(layers.nomtp_method, nomtp, xs, ws, idss)[:5]
    torch.accelerator.synchronize()
    d1 = float((got.float() - mtp5_only.float()).abs().max())
    d2 = float((got.float() - qpn8.float()).abs().max())
    print(f"\n[b3-moe] W5 lane vs 1Cat MTP5-only: max_abs {d1:.3e} "
          f"(bitwise {torch.equal(got, mtp5_only)}); vs no-MTP QPN M8 rows: "
          f"max_abs {d2:.3e} (bitwise {torch.equal(got, qpn8)})")
    scale = float(got.float().abs().max())
    assert d1 <= 1e-2 * scale and d2 <= 1e-2 * scale


@pytest.mark.parametrize("q,width", [(5, 5), (5, 10), (5, 20), (5, 30), (4, 12), (4, 32)])
def test_verify_full_graph_replay(layers, q, width):
    moe = _moe()
    lane = layers.lane
    with as_q(lane, q):
        route = moe._mtp_verify_route_table(lane)[width]
        grouped = route.startswith("grouped")
        x0, w0, ids0 = _routes("pool113", width, seed=7000 + width)
        sx, sw, sids = x0.clone(), w0.clone(), ids0.clone()
        buf = C.qsl_buffer()
        C.write_live(buf, width, q)
        meta = C.verify_metadata(buf[: width // q + 1], width, q)
        with C.forward_context(moe, meta):
            layers.lane_method.apply(lane, sx, sw, sids, None, None)  # warm
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with C.forward_context(moe, meta), torch.cuda.graph(graph):
            out = layers.lane_method.apply(lane, sx, sw, sids, None, None)
        lives = [width, width - q, q, width] if grouped else [width, width]
        kinds = ["random", "pool113", "shared10", "reversed"]
        for trial, live in enumerate(lives):
            x, w, ids = _routes(kinds[trial % 4], width, seed=7100 + 31 * trial)
            sx.copy_(x)
            sw.copy_(w)
            sids.copy_(ids)
            C.write_live(buf, live, q)
            _poison(lane)
            graph.replay()
            torch.accelerator.synchronize()
            got = out.clone()
            ref = _apply_verify(layers.lane_method, lane, x, w, ids, q)
            torch.accelerator.synchronize()
            assert torch.equal(got[:live], ref[:live]), (q, width, trial, live)
            if grouped:
                assert torch.equal(got[live:], torch.zeros_like(got[live:]))
        del graph


def test_rejected_contexts_keep_previous_route(layers):
    lane = layers.lane
    for width, q in ((10, 5), (5, 5), (15, 5), (20, 5), (6, 3)):
        x, w, ids = _routes("pool113", width, seed=900 + width)
        previous = _apply_plain(layers.lane_method, lane, x, w, ids)
        buf = C.qsl_buffer()
        C.write_live(buf, width, q)
        view = buf[: width // q + 1]
        with as_q(lane, q):
            cases = {
                "mixed": C.mixed_metadata(view, q),
                "non_uniform": C.verify_metadata(view, width, q,
                                                 spec_tokens=width - 1),
                "plain_decode_mix": C.verify_metadata(view, width, q,
                                                      plain_decodes=1),
            }
            for name, meta in cases.items():
                got = _apply_verify(layers.lane_method, lane, x, w, ids, q,
                                    metadata=meta)
                torch.accelerator.synchronize()
                assert torch.equal(got, previous), (width, q, name)
            # SX_OPT_MTP_MOE_ROUTES=0 (no lane stamping): previous route. Only
            # the stamped q is cleared; every other lane attribute is inert.
            with C.layer_attrs(lane, sx_mtp_verify_q=0):
                got = _apply_verify(layers.lane_method, lane, x, w, ids, q)
            torch.accelerator.synchronize()
            assert torch.equal(got, previous), (width, q, "switch off")


def test_lane_master_off_builds_dev2_mtp_layer(layers):
    """SX_OPT_MTP_LANE=0 (lane-wide master): the MTP-lane layer is built as
    1.8.0-dev2 built it, so every verify width keeps the previous route
    bitwise (TurboMind, static QPN at 2/4/8/16), in every context."""
    method, dev2 = _build(C.lane_config(4), lane_master=False)
    try:
        assert dev2.sx_mtp_verify_q == 0 and not dev2.sx_mtp_grouped
        assert not dev2.sx_mtp_qpn_mtp5 and not dev2.sx_mtp_qpn_dynamic
        assert dev2._nvfp4_grouped_v2_ops is None  # v2 contract refused
        assert dev2.sm70_nvfp4_grouped_max_tokens == 16
        assert dev2._nvfp4_sm70_eager_iota is None
        assert dev2.sm70_nvfp4_persistent_max_tokens == 18
        for width in (5, 8, 10, 15, 16, 20, 30):
            x, w, ids = _routes("pool113", width, seed=1300 + width)
            previous = _apply_plain(method, dev2, x, w, ids)
            got = _apply_verify(method, dev2, x, w, ids, 5 if width % 5 == 0 else 4)
            torch.accelerator.synchronize()
            assert torch.equal(got, previous), width
    finally:
        del dev2
        torch.accelerator.empty_cache()


@pytest.mark.parametrize("width", [10, 15, 20, 30])
def test_numerics_vs_previous_route(layers, width):
    lane = layers.lane
    worst = 0.0
    for kind in ("random", "pool113", "shared10"):
        x, w, ids = _routes(kind, width, seed=800 + width + len(kind))
        new = _apply_verify(layers.lane_method, lane, x, w, ids, 5).float()
        previous = _apply_plain(layers.lane_method, lane, x, w, ids).float()
        diff = (new - previous).abs()
        rel = float(diff.norm() / previous.norm().clamp_min(1e-30))
        worst = max(worst, rel)
        print(f"\n[b3-moe] W{width} {kind} grouped vs TurboMind: max_abs "
              f"{float(diff.max()):.3e} rel_l2 {rel:.3e}")
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
        print(f"[b3-moe] TurboMind warmup skipped: {exc}")
        return False


def _graph_round(layers, tokens, q, live, **attrs):
    moe = _moe()
    lane = layers.lane
    x, w, ids = _routes("pool113", tokens, seed=11000 + tokens)
    buf = C.qsl_buffer()
    C.write_live(buf, live, q)
    meta = C.verify_metadata(buf[: tokens // q + 1], tokens, q)
    with as_q(lane, q), C.layer_attrs(lane, **attrs):

        def fn():
            with C.forward_context(moe, meta):
                for _ in range(LAYERS):
                    layers.lane_method.apply(lane, x, w, ids, None, None)

        return C.graph_ms(fn)


def test_microbench(layers):
    tuned = _warm_turbomind(layers.lane, (5, 10, 15, 20, 25, 30, 32))
    print(f"\n[b3-moe] 48 x apply() in one CUDA graph, median of 60 (ms), pool113 "
          f"routes, uniform verify context; TurboMind warmup "
          f"{'on' if tuned else 'off'}")
    print(f"  {'width':<12}{'dev2':>9}{'new':>9}{'saved':>9}  alternatives")
    for label, tokens, q, live in C.BENCH_WIDTHS:
        old = _graph_round(layers, tokens, q, live, sx_mtp_verify_q=0)
        new = _graph_round(layers, tokens, q, live)
        alts = []
        if tokens == 5:
            alts.append(("mtp5+sepW2", dict(sx_mtp_qpn_dynamic=False)))
            alts.append(("grouped-s4", dict(sx_mtp_grouped_min_tokens=5)))
        if tokens in (10, 15):
            alts.append(("qpn-dyn", dict(sx_mtp_grouped_min_tokens=16)))
        if tokens == 10:
            alts.append(("grouped-s8", dict(sx_mtp_grouped_split_small=8)))
        if tokens > 16:
            alts.append(("grouped-s4", dict(sm70_nvfp4_grouped32_split=4)))
        if live < tokens:
            alts.append(("unmasked", dict(sm70_nvfp4_grouped_mask=False)))
        text = "  ".join(
            f"{name} {_graph_round(layers, tokens, q, live, **a):.3f}"
            for name, a in alts
        )
        print(f"  {label:<12}{old:>9.3f}{new:>9.3f}{old - new:>+9.3f}  {text}")
