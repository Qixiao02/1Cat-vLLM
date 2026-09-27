# SPDX-License-Identifier: Apache-2.0
"""b3 lane-core (design_1 MTP-1/MTP-6, design_4 MTP-K5): CUDA-graph capture of
the new native-MTP lane sizes on ONE V100, through the production capture
loop (vllm/v1/worker/gpu/cudagraph_utils.CudaGraphManager.capture: PIECEWISE
first, then FULL largest-first into one shared pool, SM70 pre-capture warmup,
dispatch + run_fullgraph replay).

GPU: 1 x V100 (SM70), ~3 GB. The TP collective context is replaced by a plain
capture stream (graph_capture is patched; no torch.distributed init), so no
test needs 4 GPUs.

  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-lane-core/test_mtp_lane_graph_capture.py

The "verify stack" run per descriptor is the dense part of one Qwen3.8 TP4
layer pair with the production SM70 ops and realistic shapes: fused GDN input
(qkvz 4096 + ba 24 x 2560), GDN out (2560x1536), QSA qkv (3584x2560), QSA o
(2560x1536), QSA indexer (640x2560), router (512x2560, E512), fused HC mix
(336x10240 / 10240x320) and hc_combine_norm, with the forward wrapped exactly
like ModelCudaGraphManager (sm70_decode_graph_compilation(cg_mode == FULL))
and the native-MTP lane installed.

Asserts, for k in {1,2,3,4} and the MTP-K5 verify widths q*B, B in
{1,2,3,4,6,8,12,16,20,24} (W up to 120):
  * every FULL verify descriptor captures without a host sync or an
    allocation error, largest first, into one pool;
  * dispatch(B, q*B, q) for every B in 1..24 picks a captured FULL graph;
  * replay with fresh inputs (and NaN-poisoned outputs) == eager execution
    of the same stack at the same width inside the same context, bitwise;
  * the rows kernels are captured exactly at the admitted widths (GEMV/GDN
    input W <= 8 or 4 by role, HC W <= 2) and never above;
  * PW-1 sizes (target main backbone semantics, no decode context): raw
    CUDA-graph capture at the lane grid sizes (128..1024 for k=4) replays
    bitwise equal to eager, and never enters the rows kernels.
"""

from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_lane_common as L  # noqa: E402

C = L.rows_common()

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")

K_HC = 10240
RANK = 320
HC_DIM = 2560
MAX_W = 8 * 24  # widest verify width if k were 7; buffers sized once


class VerifyStack:
    """Persistent inputs/outputs + the op sequence of one verify forward."""

    ROLES = {
        "qsa_qkv": ("model.layers.3.self_attn.qkv_proj", (3584, 2560)),
        "qsa_o": ("model.layers.3.self_attn.o_proj", (2560, 1536)),
        "qsa_index": ("model.layers.3.self_attn.indexer.index_qk_proj", (640, 2560)),
        "router": ("model.layers.3.mlp.gate", (512, 2560)),
        "gdn_out": ("model.layers.0.linear_attn.out_proj", (2560, 1536)),
    }

    def __init__(self, max_w: int = MAX_W, seed: int = 1):
        self.max_w = max_w
        self.weights = {
            name: C.make_weight(shape[0], shape[1], 0.05, seed=seed + i)
            for i, (name, (_, shape)) in enumerate(self.ROLES.items())
        }
        self.qkvz_w = C.make_weight(4096, 2560, 0.05, seed + 10)
        self.ba_w = C.make_weight(24, 2560, 0.05, seed + 11)
        down = C.make_weight(336, K_HC, 0.02, seed + 12)
        down[324:].zero_()
        self.hc_down = down.contiguous()
        self.hc_up = C.make_weight(K_HC, RANK, 0.05, seed + 13)
        self.norm_w = (torch.randn(HC_DIM, device="cuda") * 0.1).half()
        self.hidden = torch.zeros(max_w, 2560, dtype=torch.float16, device="cuda")
        self.hc_in = torch.zeros(max_w, K_HC, dtype=torch.float16, device="cuda")
        self.outs: dict[str, torch.Tensor] = {}
        self.refill(0)

    def refill(self, seed: int) -> None:
        self.hidden.copy_(C.make_rows(self.max_w, 2560, 1.0, seed=seed + 101))
        self.hc_in.copy_(C.make_rows(self.max_w, K_HC, 1.0, seed=seed + 202))

    def run(self, w: int) -> dict[str, torch.Tensor]:
        ops = torch.ops.vllm
        x = self.hidden[:w]
        qkv, z, b, a = ops.qwen38_sm70_fp16_gdn_input(x, self.qkvz_w, self.ba_w)
        res = {"gdn_qkv": qkv, "gdn_z": z, "gdn_b": b, "gdn_a": a}
        z_in = z.reshape(w, 1536).contiguous()
        for name, (prefix, shape) in self.ROLES.items():
            inp = z_in if shape[1] == 1536 else x
            res[name] = ops.qwen38_sm70_fp16_gemv(inp, self.weights[name], prefix)
        hc_x = self.hc_in[:w]
        block, injection = ops.qwen38_sm70_fp16_fused_hc(hc_x, self.hc_down, self.hc_up)
        res["hc_block"], res["hc_inj"] = block, injection
        out, y = ops.qwen4_exp_hc_combine_norm(
            hc_x, block, injection.contiguous(), self.norm_w, 1e-6, 4
        )
        res["norm_out"], res["norm_y"] = out, y
        return res

    def run_into_persistent(self, w: int) -> None:
        res = self.run(w)
        for name, value in res.items():
            buf = self.outs.get(name)
            if buf is None:
                buf = torch.empty(
                    (self.max_w, *value.shape[1:]), dtype=value.dtype, device="cuda"
                )
                self.outs[name] = buf
            buf[:w].copy_(value)


@contextlib.contextmanager
def _fake_graph_capture(device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield SimpleNamespace(stream=stream)
    torch.cuda.current_stream().wait_stream(stream)


class _Spy:
    """Record (kernel, weight rows, width) whenever a rows kernel is entered."""

    def __init__(self, monkeypatch):
        gemv = C.gemv_module()
        hc = C.hc_module()
        self.calls: set[tuple[str, int, int]] = set()
        self.enabled = False
        for mod, name in (
            (gemv, "_sx_rows_gemv"),
            (gemv, "_sx_rows_gdn_input"),
            (hc, "_sx_hc_rows_forward"),
        ):
            real = getattr(mod, name)

            def wrapper(x, weight, *args, _real=real, _name=name, **kwargs):
                if self.enabled:
                    self.calls.add((_name, int(weight.shape[0]), int(x.shape[0])))
                return _real(x, weight, *args, **kwargs)

            monkeypatch.setattr(mod, name, wrapper)

    def widths(self, name, weight_rows):
        return sorted(
            w for kind, rows, w in self.calls if kind == name and rows == weight_rows
        )


@contextlib.contextmanager
def _lane_installed():
    from vllm.compilation import sm70_decode_graph as dg

    saved = dg.sm70_mtp_lane_installed()
    dg.set_sm70_mtp_lane_installed(True)
    try:
        yield
    finally:
        dg.set_sm70_mtp_lane_installed(saved)


@pytest.fixture
def lane(monkeypatch):
    import vllm.config.vllm as cfg_mod
    from vllm.v1.worker.gpu import cudagraph_utils as cgu

    C.gemv_module()
    C.hc_module()  # registers the HC custom ops used by the stack
    monkeypatch.setattr(cgu, "graph_capture", _fake_graph_capture)
    monkeypatch.setattr(cgu, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(
        cgu, "get_pp_group", lambda: MagicMock(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(cfg_mod, "_sx_piecewise_platform_ok", lambda: True)
    # Validation fix (b3a): every test builds its own manager; once the previous
    # test's graphs are destroyed the shared global pool id is released, and
    # capturing into it again trips the caching allocator's use_count assert.
    # Give each test a fresh platform graph pool.
    from vllm.platforms import current_platform

    monkeypatch.setattr(type(current_platform), "_global_graph_pool", None)
    with L.sx_env(**L.LANE_ENV):
        yield cgu


@pytest.mark.parametrize("k", L.K_VALUES)
def test_capture_and_replay_verify_widths(lane, monkeypatch, k):
    cgu = lane
    from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation

    FULL = cgu.CUDAGraphMode.FULL
    q = k + 1
    widths = [q * b for b in L.LANE_REQS]
    cfg = L.lane_config(k)
    mgr = cgu.CudaGraphManager(
        cfg, torch.device("cuda"), cgu.CUDAGraphMode.FULL_AND_PIECEWISE, q
    )
    full_descs = list(mgr._capture_descs[FULL])
    assert sorted(d.num_tokens for d in full_descs) == widths
    stack = VerifyStack()
    spy = _Spy(monkeypatch)
    torch.cuda.synchronize()

    def create_forward_fn(desc):
        def forward_fn(cg_mode):
            # Same wrapper as ModelCudaGraphManager.capture's forward_fn.
            with sm70_decode_graph_compilation(desc.cg_mode == FULL):
                stack.run_into_persistent(desc.num_tokens)

        return forward_fn, cgu.CapturedAttentionState(None, {})

    # CudaGraphManager.capture runs under inference_mode (as in production),
    # so the persistent buffers it creates are inference tensors: keep the
    # replay/compare in inference mode too.
    with (
        torch.inference_mode(),
        _lane_installed(),
        C.sx_env(SX_OPT_ROWS_TABLE=None, SX_OPT_ROWS=None, SX_OPT_MTP_ROWS=None),
    ):
        spy.enabled = True
        mgr.capture(create_forward_fn, progress_bar_desc="b3 verify")
        spy.enabled = False
        assert set(mgr.graphs) == set(full_descs)
        # Rows kernels were entered exactly at the admitted widths.
        gemv = C.gemv_module()
        for role, (prefix, shape) in VerifyStack.ROLES.items():
            cap = gemv._sx_rows_max_m(gemv._sx_role_key(prefix, shape))
            if shape[0] == 2560:  # qsa_o and gdn_out share the shape and cap
                assert role in ("qsa_o", "gdn_out") and cap == gemv._sx_rows_max_m(
                    "qsa_o"
                ) == gemv._sx_rows_max_m("gdn_out")
            assert spy.widths("_sx_rows_gemv", shape[0]) == [
                w for w in widths if w <= cap
            ], (role, cap)
        gdn_cap = gemv._sx_rows_max_m("gdn_in")
        hc_cap = gemv._sx_rows_max_m("hc")
        assert spy.widths("_sx_rows_gdn_input", 4096) == [
            w for w in widths if w <= gdn_cap
        ]
        assert spy.widths("_sx_hc_rows_forward", 336) == [
            w for w in widths if w <= hc_cap
        ]
        # Dispatch every live request count to a captured FULL graph.
        for b in range(1, L.MAX_SEQS + 1):
            desc = mgr.dispatch(b, b * q, q)
            assert desc.cg_mode == FULL and desc in mgr.graphs, (k, b, desc)
        # Replay == eager at the same width, fresh inputs, poisoned outputs.
        for step, desc in enumerate(sorted(full_descs, key=lambda d: d.num_tokens)):
            w = desc.num_tokens
            stack.refill(1000 * k + step)
            for buf in stack.outs.values():
                buf.fill_(float("nan"))
            mgr.run_fullgraph(desc)
            torch.cuda.synchronize()
            replayed = {name: buf[:w].clone() for name, buf in stack.outs.items()}
            with sm70_decode_graph_compilation(True):
                eager = stack.run(w)
            torch.cuda.synchronize()
            for name, value in eager.items():
                assert C.bit_equal(replayed[name], value), (k, w, name)


def test_pw_sizes_capture_replay(lane, monkeypatch):
    """PW-1 sizes run the target main backbone: rows kernels never, replay ==
    eager (raw CUDA graphs; production captures them piecewise)."""
    del lane
    import vllm.config.vllm as cfg_mod

    k = 4
    cfg = L.lane_config(k)
    sizes = cfg_mod._sm70_qwen38_mixed_piecewise_sizes(
        cfg, cfg.compilation_config.cudagraph_capture_sizes
    )
    assert sizes and sizes[0] == 128 and sizes[-1] == 1024
    probe = [s for s in sizes if s in (128, 160, 512, 1024)]
    stack = VerifyStack(max_w=1024)
    spy = _Spy(monkeypatch)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    with torch.inference_mode(), L.mtp_main_ctx():
        spy.enabled = True
        for w in sorted(probe, reverse=True):
            with _fake_graph_capture(torch.device("cuda")):
                stack.run_into_persistent(w)  # warmup on the capture stream
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool):
                    stack.run_into_persistent(w)
            graphs[w] = graph
        spy.enabled = False
        assert not spy.calls, spy.calls
        for step, w in enumerate(probe):
            stack.refill(77 + step)
            for buf in stack.outs.values():
                buf.fill_(float("nan"))
            graphs[w].replay()
            torch.cuda.synchronize()
            replayed = {n: b[:w].clone() for n, b in stack.outs.items()}
            eager = stack.run(w)
            torch.cuda.synchronize()
            for name, value in eager.items():
                assert C.bit_equal(replayed[name], value), (w, name)
            # Main-backbone semantics == the old cuBLAS path.
            ref = F.linear(stack.hidden[:w], stack.weights["router"])
            assert C.bit_equal(replayed["router"], ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
