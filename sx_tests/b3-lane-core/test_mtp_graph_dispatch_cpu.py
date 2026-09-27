# SPDX-License-Identifier: Apache-2.0
"""b3 lane-core: CUDA-graph descriptors and dispatch of the SM70 Qwen3.8
native-MTP lane for verify widths B*(k+1) (design_1 MTP-0/MTP-6, design_2 PF2,
design_4 MTP-K5).

GPU: none. The real ModelCudaGraphManager / PrefillEagleCudaGraphManager /
DecodeEagleCudaGraphManager constructors run on CPU with the platform, PP
group and graph-pool handles patched (no CUDA context):

  /opt/venv/bin/python -m pytest -q sx_tests/b3-lane-core/test_mtp_graph_dispatch_cpu.py

For k in {1,2,3,4}, max_num_seqs 24, split verify/draft graphs (lane default)
and the shared (non-split) fallback:
  * target FULL descriptors = the MTP-K5 verify widths q*B, B in
    {1,2,3,4,6,8,12,16,20,24}, uniform_token_count q, num_reqs B; the Eagle
    draft-prefill manager has the identical FULL set; the split draft-decode
    manager has FULL B for the same B list (B17..24 included);
  * PW-1: target PIECEWISE = capture sizes + the grid above q*24 up to 1024;
    the draft-prefill manager owns exactly the same PIECEWISE descriptors
    (its capture looks the target's captured PIECEWISE states up by
    descriptor), SX_OPT_MTP_PW_DRAFT=0 keeps it on the old list,
    SX_OPT_MTP_PW=0 / SX_OPT_MTP_LANE=0 restore the previous lists;
  * dispatch, every B in 1..24: uniform verify -> FULL at the next graph B
    (never eager), non-uniform verify (one request short of drafts) and
    mixed steps (B verify requests + a prefill chunk up to 1100 tokens) ->
    PIECEWISE at the next size <= 1024, NONE above; the draft prefill replays
    the target's padded size with the same mode; draft decode B -> FULL;
  * no-MTP target manager: identical to batch 2 (PW grid 32..1024).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_lane_common as L  # noqa: E402

import vllm.config.vllm as cfg_mod  # noqa: E402
from vllm.compilation import sm70_decode_graph as dg  # noqa: E402
from vllm.config.compilation import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils as cgu  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.eagle import cudagraph as eagle_cg  # noqa: E402

FULL = CUDAGraphMode.FULL
PIECEWISE = CUDAGraphMode.PIECEWISE
NONE = CUDAGraphMode.NONE
K_VALUES = L.K_VALUES
LANE_REQS = L.LANE_REQS
GRID_1024 = L.GRID_1024
MAX_SEQS = L.MAX_SEQS
sx_env = L.sx_env
lane_config = L.lane_config


@pytest.fixture(autouse=True)
def _platform(monkeypatch):
    platform = cgu.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        platform,
        "is_device_capability",
        lambda capability, *a, **k: capability in ((7, 0), 70),
    )
    monkeypatch.setattr(platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(
        cgu, "get_pp_group", lambda: MagicMock(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(cfg_mod, "_sx_piecewise_platform_ok", lambda: True)
    saved_installed = dg.sm70_mtp_lane_installed()
    with sx_env(**L.LANE_ENV, VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS="0"):
        try:
            yield
        finally:
            dg.set_sm70_mtp_lane_installed(saved_installed)


def managers(cfg, k=4, captured=True):
    """(target, draft_prefill, draft_decode) as the V2 runner builds them."""
    q = k + 1
    mode = CUDAGraphMode.FULL_AND_PIECEWISE
    target = cgu.ModelCudaGraphManager(cfg, torch.device("cpu"), mode, q)
    prefill = eagle_cg.PrefillEagleCudaGraphManager(cfg, torch.device("cpu"), mode, q)
    decode = eagle_cg.DecodeEagleCudaGraphManager(
        cfg, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 1
    )
    for mgr in (target, prefill, decode):
        mgr._graphs_captured = captured
    return target, prefill, decode


def _next(n, sizes):
    return next((s for s in sorted(sizes) if s >= n), None)


def _descs(mgr, mode):
    return list(mgr._capture_descs.get(mode, []))


def _assert_draft_lookup_ok(target, prefill):
    """PrefillEagleCudaGraphManager.capture looks every one of its descriptors
    up in the target's captured attention states (full_cg_attn_states[desc]);
    a descriptor the target did not capture is a KeyError at startup."""
    captured = {d for descs in target._capture_descs.values() for d in descs}
    missing = [
        d for descs in prefill._capture_descs.values() for d in descs
        if d not in captured
    ]  # fmt: skip
    assert not missing, missing


@pytest.mark.parametrize("k", K_VALUES)
def test_descriptors_split_lane(k):
    q = k + 1
    cfg = lane_config(k)
    target, prefill, decode = managers(cfg, k)
    verify = [q * b for b in LANE_REQS]
    # FULL verify graphs: exact widths, B17..24 included.
    full = _descs(target, FULL)
    assert sorted(d.num_tokens for d in full) == verify
    assert all(
        d.uniform_token_count == q and d.num_reqs * q == d.num_tokens for d in full
    )
    assert _descs(prefill, FULL) == full
    # Split draft decode: exact request counts.
    dfull = _descs(decode, FULL)
    assert sorted(d.num_tokens for d in dfull) == list(LANE_REQS)
    assert all(d.uniform_token_count == 1 and d.num_reqs == d.num_tokens for d in dfull)
    assert PIECEWISE not in decode._capture_descs
    # PW-1 (target + draft prefill, identical descriptors).
    grid = [s for s in GRID_1024 if s > q * MAX_SEQS]
    assert target._sx_piecewise_only_sizes == tuple(grid)
    assert prefill._sx_piecewise_only_sizes == tuple(grid)
    assert decode._sx_piecewise_only_sizes == ()
    pw = _descs(target, PIECEWISE)
    assert [d.num_tokens for d in pw] == sorted(verify + grid, reverse=True)
    assert set(_descs(prefill, PIECEWISE)) == set(pw)
    assert all(d.num_reqs is None and d.uniform_token_count is None for d in pw)
    _assert_draft_lookup_ok(target, prefill)
    # The shared config list is never widened by PW sizes.
    assert cfg.compilation_config.cudagraph_capture_sizes == verify
    assert cfg.compilation_config.max_cudagraph_capture_size == q * MAX_SEQS


@pytest.mark.parametrize("k", K_VALUES)
def test_descriptors_switches(k):
    q = k + 1
    verify = [q * b for b in LANE_REQS]
    with sx_env(SX_OPT_MTP_PW_DRAFT="0"):
        target, prefill, _ = managers(lane_config(k), k)
    assert target._sx_piecewise_only_sizes and prefill._sx_piecewise_only_sizes == ()
    _assert_draft_lookup_ok(target, prefill)
    assert [d.num_tokens for d in _descs(prefill, PIECEWISE)] == sorted(
        verify, reverse=True
    )
    for env in (dict(SX_OPT_MTP_PW="0"), dict(SX_OPT_MTP_LANE="0")):
        with sx_env(**env):
            target, prefill, _ = managers(lane_config(k), k)
        assert target._sx_piecewise_only_sizes == ()
        assert prefill._sx_piecewise_only_sizes == ()
        assert [d.num_tokens for d in _descs(target, PIECEWISE)] == sorted(
            verify, reverse=True
        )
    # Other speculative methods never get PW-1 sizes.
    target, prefill, _ = managers(lane_config(k, spec="eagle3"), k)
    assert target._sx_piecewise_only_sizes == () == prefill._sx_piecewise_only_sizes


@pytest.mark.parametrize("k", K_VALUES)
def test_descriptors_shared_fallback(k):
    """VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=0 (bounded shared list)."""
    q = k + 1
    with sx_env(VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS="0"):
        lane = cfg_mod._sm70_qwen38_mtp_lane_capture_sizes(MAX_SEQS, q)
        sizes = sorted({1, 2, 4, 8, 9, 18} | set(lane))
        cfg = lane_config(k, sizes=sizes)
        target, prefill, decode = managers(cfg, k)
    rounded = cfg.compilation_config.cudagraph_capture_sizes
    assert all(size % q == 0 for size in rounded) and max(rounded) == q * MAX_SEQS
    assert target._sx_piecewise_only_sizes == prefill._sx_piecewise_only_sizes
    assert target._sx_piecewise_only_sizes[0] > q * MAX_SEQS
    assert set(_descs(prefill, PIECEWISE)) == set(_descs(target, PIECEWISE))
    assert _descs(prefill, FULL) == _descs(target, FULL)
    _assert_draft_lookup_ok(target, prefill)
    # Shared draft decode pads to multiples of q (previous behaviour).
    assert all(d.num_tokens % q == 0 for d in _descs(decode, FULL))


def _mixed(num_verify_reqs, q, chunk):
    """B uniform verify requests + one prefill chunk (chunk >= 2 tokens)."""
    num_reqs = num_verify_reqs + 1
    num_tokens = num_verify_reqs * q + chunk
    max_query_len = max(q if num_verify_reqs else 0, chunk)
    return num_reqs, num_tokens, cgu.get_uniform_token_count(
        num_reqs, num_tokens, max_query_len
    )


@pytest.mark.parametrize("k", K_VALUES)
def test_dispatch_verify_every_batch(k):
    q = k + 1
    target, prefill, decode = managers(lane_config(k), k)
    for b in range(1, MAX_SEQS + 1):
        graph_b = _next(b, LANE_REQS)
        desc = target.dispatch(b, b * q, q)
        assert (desc.cg_mode, desc.num_tokens, desc.num_reqs) == (
            FULL,
            graph_b * q,
            graph_b,
        ), (k, b, desc)
        # Draft prefill (step 0) replays at the target's padded size.
        pdesc = prefill.dispatch(b, desc.num_tokens, q)
        assert pdesc == desc
        # Draft decode (steps 1..k-1): exact B graphs.
        ddesc = decode.dispatch(b, b, 1)
        assert (ddesc.cg_mode, ddesc.num_tokens, ddesc.num_reqs) == (
            FULL,
            graph_b,
            graph_b,
        ), (k, b, ddesc)
        # Non-uniform verify (one request short of drafts): PIECEWISE.
        if b >= 2:
            tokens = b * q - 1
            desc = target.dispatch(b, tokens, None)
            assert desc.cg_mode == PIECEWISE and desc.num_tokens >= tokens


@pytest.mark.parametrize("k", K_VALUES)
def test_dispatch_mixed_steps(k):
    q = k + 1
    target, prefill, _ = managers(lane_config(k), k)
    with sx_env(SX_OPT_MTP_PW="0"):
        old_target, old_prefill, _ = managers(lane_config(k), k)
    verify = [q * b for b in LANE_REQS]
    grid = [s for s in GRID_1024 if s > q * MAX_SEQS]
    pw_sizes = verify + grid
    for b in (0, 1, 4, 16, 23):
        for chunk in (2, 3, 7, 33, 100, 450, 784, 1000, 1100):
            num_reqs, tokens, uniform = _mixed(b, q, chunk)
            desc = target.dispatch(num_reqs, tokens, uniform)
            padded = _next(tokens, pw_sizes)
            if uniform == q:
                # Every request has q rows: indistinguishable from a verify
                # batch, so it replays the FULL verify graph.
                graph_b = _next(num_reqs, LANE_REQS)
                assert (desc.cg_mode, desc.num_tokens) == (FULL, q * graph_b)
            elif padded is None:
                assert desc.cg_mode == NONE and desc.num_tokens == tokens
            else:
                assert (desc.cg_mode, desc.num_tokens) == (PIECEWISE, padded), (
                    k, b, chunk, desc,
                )  # fmt: skip
            # Draft prefill follows the target's padded token count.
            pdesc = prefill.dispatch(num_reqs, desc.num_tokens, uniform)
            assert (pdesc.cg_mode, pdesc.num_tokens) == (desc.cg_mode, desc.num_tokens)
            # SX_OPT_MTP_PW=0: previous behaviour (eager above q*24).
            old = old_target.dispatch(num_reqs, tokens, uniform)
            if tokens <= q * MAX_SEQS:
                assert old == desc
            else:
                assert old.cg_mode == NONE and old.num_tokens == tokens
            old_p = old_prefill.dispatch(num_reqs, old.num_tokens, uniform)
            assert old_p.cg_mode == old.cg_mode


def test_nomtp_target_unchanged():
    cfg = lane_config(spec=None, sizes=[1, 2, 4, 8, 16, 24])
    target = cgu.ModelCudaGraphManager(
        cfg, torch.device("cpu"), CUDAGraphMode.FULL_AND_PIECEWISE, 1
    )
    assert target._sx_piecewise_only_sizes == tuple(GRID_1024)
    assert [d.num_tokens for d in _descs(target, FULL)] == [24, 16, 8, 4, 2, 1]
    for env in (dict(SX_OPT_MTP_PW="0"), dict(SX_OPT_MTP_LANE="0")):
        with sx_env(**env):
            again = cgu.ModelCudaGraphManager(
                cfg, torch.device("cpu"), CUDAGraphMode.FULL_AND_PIECEWISE, 1
            )
        assert again._sx_piecewise_only_sizes == tuple(GRID_1024)
        assert again._capture_descs == target._capture_descs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
