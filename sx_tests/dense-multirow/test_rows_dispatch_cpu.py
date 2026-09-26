# SPDX-License-Identifier: Apache-2.0
"""dense-multirow: admission, config and launch-geometry checks (no GPU).

GPU: none (meta tensors + recording fake kernels); runs on any host where the
overlaid vllm imports.

  /opt/venv/bin/python -m pytest -q sx_tests/dense-multirow/test_rows_dispatch_cpu.py

Asserts: SX_OPT_ROWS* parsing and clamping; token-tile split; role keys;
the multi-row kernels are launched with the role's M=1 BLOCK_K / num_warps /
LOAD_POLICY (the invariant that makes rows bitwise equal to M=1); the M=1
launch is unchanged; admission requires the FULL decode-graph context,
fp16 CUDA packed inputs, VLLM_BATCH_INVARIANT off and M within the table;
hc_combine_norm picks the M=1 tile only when the HC rows route is admitted.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sx_rows_common as C  # noqa: E402


class _Recorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((grid, args, kwargs))


def _meta(*shape):
    return torch.empty(*shape, dtype=torch.float16, device="meta")


def test_config_parsing_and_split():
    gemv = C.gemv_module()
    with C.sx_env(
        SX_OPT_ROWS=None,
        SX_OPT_ROWS_MAX_M=None,
        SX_OPT_ROWS_TABLE=None,
        SX_OPT_ROWS_FUSED_REDUCE=None,
    ):
        cfg = gemv._sx_rows_config()
        assert cfg.enabled and cfg.max_m == 24 and cfg.fused_reduce
        assert cfg.table == gemv._SX_ROWS_DEFAULT_MAX_M
        assert (cfg.gemv_tile, cfg.hc_down_tile, cfg.hc_down_nw, cfg.hc_up_tile) == (
            8,
            4,
            4,
            8,
        )
        assert gemv._sx_rows_max_m("router") == gemv._SX_ROWS_DEFAULT_MAX_M["router"]
        assert gemv._sx_rows_max_m("gdn_in") == 8
        assert gemv._sx_rows_max_m("unknown") == 0
    with C.sx_env(
        SX_OPT_ROWS_TABLE="hc=16, router=8,bogus=3,gdn_out=x",
        SX_OPT_ROWS_MAX_M="12",
        SX_OPT_ROWS_HC_DOWN_NW="3",
        SX_OPT_ROWS_GEMV_TILE="99",
    ):
        cfg = gemv._sx_rows_config()
        assert cfg.table["hc"] == 16 and cfg.table["router"] == 8
        assert cfg.table["gdn_out"] == gemv._SX_ROWS_DEFAULT_MAX_M["gdn_out"]  # invalid entry ignored
        assert gemv._sx_rows_max_m("hc") == 12  # global cap
        assert cfg.hc_down_nw == 2 and cfg.gemv_tile == 8
    with C.sx_env(SX_OPT_ROWS="0"):
        assert gemv._sx_rows_max_m("router") == 0
    split = gemv._sx_split_rows
    assert [split(m, 8) for m in (1, 2, 4, 8, 12, 16, 17, 24, 32)] == [
        1, 2, 4, 8, 6, 8, 6, 8, 8,
    ]  # fmt: skip
    assert [split(m, 4) for m in (2, 4, 8, 16, 24)] == [2, 4, 4, 4, 4]


def test_role_keys():
    gemv = C.gemv_module()
    for name, prefix, shape, _, _ in C.GEMV_ROLES:
        key = gemv._sx_role_key(prefix, shape)
        assert key is not None, name
    assert gemv._sx_role_key("", (512, 2560)) is None
    assert gemv._sx_role_key("model.layers.3.mlp.gate", (513, 2560)) is None


@pytest.mark.parametrize(
    "prefix,shape,m,rows,grid0",
    [
        ("model.layers.0.linear_attn.out_proj", (2560, 1536), 4, 4, 1),
        ("model.layers.3.self_attn.o_proj", (2560, 1536), 2, 2, 1),
        ("model.layers.3.self_attn.qkv_proj", (3584, 2560), 8, 8, 1),
        ("model.layers.3.mlp.gate", (512, 2560), 24, 8, 3),
        ("model.layers.3.mlp.gate", (512, 2560), 16, 8, 2),
    ],
)
def test_gemv_rows_launch_uses_m1_plan(monkeypatch, prefix, shape, m, rows, grid0):
    gemv = C.gemv_module()
    rec = _Recorder()
    monkeypatch.setattr(gemv, "_qwen38_fp16_rows_gemv_kernel", rec)
    monkeypatch.setattr(gemv, "_sx_rows_inputs_ok", lambda *a: True)
    monkeypatch.setattr(gemv, "_sx_decode_graph_active", lambda: True)
    with C.sx_env(SX_OPT_ROWS=None, SX_OPT_ROWS_TABLE=C.IMPL_TABLE, SX_OPT_ROWS_MAX_M=None):
        out = gemv._qwen38_sm70_fp16_gemv(_meta(m, shape[1]), _meta(*shape), prefix)
    plan = gemv._plan_for(prefix, shape)
    assert out.shape == (m, shape[0])
    assert len(rec.launches) == 1
    grid, args, kwargs = rec.launches[0]
    assert grid == (grid0, shape[0])
    assert args[3] == m
    assert kwargs["BLOCK_K"] == plan.block_k
    assert kwargs["num_warps"] == plan.num_warps
    assert kwargs["LOAD_POLICY"] == plan.load_policy
    assert kwargs["ROWS"] == rows and kwargs["N"] == shape[0]
    assert kwargs["MASK_ROWS"] is (m % rows != 0)


def test_gemv_rows_admission_gates(monkeypatch):
    gemv = C.gemv_module()
    prefix, shape = "model.layers.0.linear_attn.out_proj", (2560, 1536)
    x4, w = _meta(4, 1536), _meta(*shape)
    key = gemv._sx_role_key(prefix, shape)
    real_inputs_ok = gemv._sx_rows_inputs_ok  # the monkeypatch below lasts until teardown
    with C.sx_env(SX_OPT_ROWS=None, SX_OPT_ROWS_TABLE=C.IMPL_TABLE, SX_OPT_ROWS_MAX_M=None):
        monkeypatch.setattr(gemv, "_sx_rows_inputs_ok", lambda *a: True)
        # No decode-graph context: never admitted.
        monkeypatch.setattr(gemv, "_sx_decode_graph_active", lambda: False)
        assert gemv._sx_rows_tile(x4, w, key) == 0
        monkeypatch.setattr(gemv, "_sx_decode_graph_active", lambda: True)
        assert gemv._sx_rows_tile(x4, w, key) == 4
        assert gemv._sx_rows_tile(_meta(1, 1536), w, key) == 0
        assert gemv._sx_rows_tile(_meta(16, 1536), w, key) == 0  # table max 8
        assert gemv._sx_rows_tile(x4, w, None) == 0  # role-less legacy call
        # setenv (not setattr on vllm.envs): a module attribute would outlive
        # the test and shadow the env lookup for the rest of the session.
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        assert gemv._sx_rows_tile(x4, w, key) == 0
        monkeypatch.delenv("VLLM_BATCH_INVARIANT")
        assert gemv._sx_rows_tile(x4, w, key) == 4
    # Real input checks: CPU tensors are never admitted.
    assert not real_inputs_ok(
        torch.empty(4, 8, dtype=torch.float16), torch.empty(3, 8, dtype=torch.float16)
    )


def test_gemv_m1_launch_unchanged(monkeypatch):
    gemv = C.gemv_module()
    rec = _Recorder()
    monkeypatch.setattr(gemv, "_runtime_ok", lambda *a: True)
    monkeypatch.setattr(gemv, "_qwen38_fp16_row_gemv_kernel", rec)
    out = gemv._qwen38_sm70_fp16_gemv(
        _meta(1, 1536), _meta(2560, 1536), "model.layers.0.linear_attn.out_proj"
    )
    assert out.shape == (1, 2560)
    assert rec.launches == [
        (
            (2560,),
            rec.launches[0][1],
            {"K": 1536, "BLOCK_K": 512, "LOAD_POLICY": 1, "num_warps": 4},
        )
    ]


def test_gdn_rows_launch(monkeypatch):
    gemv = C.gemv_module()
    rec = _Recorder()
    monkeypatch.setattr(gemv, "_qwen38_fp16_gdn_input_rows_kernel", rec)
    monkeypatch.setattr(gemv, "_sx_rows_inputs_ok", lambda *a: True)
    monkeypatch.setattr(gemv, "_sx_decode_graph_active", lambda: True)
    with C.sx_env(SX_OPT_ROWS=None, SX_OPT_ROWS_TABLE=C.IMPL_TABLE, SX_OPT_ROWS_MAX_M=None):
        outs = gemv._qwen38_sm70_fp16_gdn_input(
            _meta(8, 2560), _meta(4096, 2560), _meta(24, 2560)
        )
    assert [tuple(o.shape) for o in outs] == [(8, 2560), (8, 1536), (8, 12), (8, 12)]
    grid, args, kwargs = rec.launches[0]
    assert grid == (1, 4096 + 24)
    assert kwargs["BLOCK_K"] == 512 and kwargs["num_warps"] == 2
    assert kwargs["ROWS"] == 8 and kwargs["K"] == 2560


def test_hc_rows_launch(monkeypatch):
    hc = C.hc_module()
    down_rec, up_rec = _Recorder(), _Recorder()
    monkeypatch.setattr(hc, "_qwen38_hc_down_silu_inject_rows_kernel", down_rec)
    monkeypatch.setattr(hc, "_qwen38_hc_up_gate_mix_row4_rows_kernel", up_rec)
    with C.sx_env(SX_OPT_ROWS=None, SX_OPT_ROWS_TABLE=C.IMPL_TABLE, SX_OPT_ROWS_MAX_M=None):
        assert hc._sx_hc_rows_plan_for_m(1) is None
        assert hc._sx_hc_rows_plan_for_m(16) is None  # default hc max 8
        plan = hc._sx_hc_rows_plan_for_m(8)
    assert plan == hc._SxHcRowsPlan(4, 4, 8, True)
    block, injection = hc._sx_hc_rows_forward(
        _meta(8, 10240), _meta(336, 10240), _meta(10240, 320), plan
    )
    assert block.shape == (8, 2560) and injection.shape == (8, 4)
    assert injection.is_contiguous()
    grid, _, kwargs = down_rec.launches[0]
    assert grid == (2, 81)
    assert kwargs["BLOCK_K"] == 256 and kwargs["num_warps"] == 4
    assert kwargs["NW"] == 4 and kwargs["ROWS"] == 4
    grid, _, kwargs = up_rec.launches[0]
    assert grid == (1, 640)
    assert kwargs["BLOCK_K"] == 512 and kwargs["BLOCK_N"] == 4
    assert kwargs["num_warps"] == 8 and kwargs["ROWS"] == 8


@pytest.mark.parametrize("admitted", [False, True])
def test_combine_norm_tile_follows_hc_admission(monkeypatch, admitted):
    hc = C.hc_module()
    from vllm.models.qwen4_exp.nvidia.ops import hc as hcops

    kernel = MagicMock()
    monkeypatch.setattr(hcops, "_hc_combine_norm_kernel", kernel)
    monkeypatch.setattr(hcops.current_platform, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(hc, "_SX_HC_FUSED_MODULES", 1)
    monkeypatch.setattr(hcops, "is_sm70_decode_graph_compiling", lambda: True)
    monkeypatch.setattr(hc, "_sx_decode_graph_active", lambda: admitted)
    with C.sx_env(SX_OPT_ROWS=None, SX_OPT_ROWS_TABLE=C.IMPL_TABLE, SX_OPT_ROWS_MAX_M=None):
        hcops._hc_combine_norm(
            torch.empty(4, 10240, dtype=torch.float16),
            torch.empty(4, 2560, dtype=torch.float16),
            torch.empty(4, 4, dtype=torch.float16),
            torch.empty(2560, dtype=torch.float16),
            1e-6,
            4,
        )
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    # _sx_decode_graph_active (patched) carries the SM70 + FULL-graph check.
    assert kwargs["BLOCK_SIZE"] == (1024 if admitted else 512)
    assert kwargs["PREFETCH_WEIGHT"] is admitted


@pytest.mark.parametrize(
    "n,fused_modules,env",
    [
        (16, 1, {}),  # above the default hc table (8)
        (4, 0, {}),  # fused FP16 HC not enabled for this model
        (4, 1, {"SX_OPT_ROWS_HC_NORM": "0"}),
        (4, 1, {"SX_OPT_ROWS": "0"}),
        (4, 1, {"SX_OPT_ROWS_TABLE": "hc=0"}),
    ],
)
def test_combine_norm_keeps_generic_tile_when_hc_not_admitted(
    monkeypatch, n, fused_modules, env
):
    """Inside the decode graph, the tile only changes where HC rows run."""
    hc = C.hc_module()
    from vllm.models.qwen4_exp.nvidia.ops import hc as hcops

    kernel = MagicMock()
    monkeypatch.setattr(hcops, "_hc_combine_norm_kernel", kernel)
    monkeypatch.setattr(hcops.current_platform, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(hc, "_SX_HC_FUSED_MODULES", fused_modules)
    monkeypatch.setattr(hcops, "is_sm70_decode_graph_compiling", lambda: True)
    monkeypatch.setattr(hc, "_sx_decode_graph_active", lambda: True)
    overrides = {
        "SX_OPT_ROWS": None,
        "SX_OPT_ROWS_TABLE": None,
        "SX_OPT_ROWS_MAX_M": None,
        "SX_OPT_ROWS_HC_NORM": None,
        **env,
    }
    with C.sx_env(**overrides):
        hcops._hc_combine_norm(
            torch.empty(n, 10240, dtype=torch.float16),
            torch.empty(n, 2560, dtype=torch.float16),
            torch.empty(n, 4, dtype=torch.float16),
            torch.empty(2560, dtype=torch.float16),
            1e-6,
            4,
        )
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["BLOCK_SIZE"] == 512 and kwargs["PREFETCH_WEIGHT"] is False


def test_combine_norm_eager_keeps_generic_tile(monkeypatch):
    """Outside a FULL decode-graph capture the N>1 tile never changes."""
    hc = C.hc_module()
    from vllm.models.qwen4_exp.nvidia.ops import hc as hcops

    kernel = MagicMock()
    monkeypatch.setattr(hcops, "_hc_combine_norm_kernel", kernel)
    monkeypatch.setattr(hcops.current_platform, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(hc, "_SX_HC_FUSED_MODULES", 1)
    monkeypatch.setattr(hc, "_sx_decode_graph_active", lambda: True)
    hcops._hc_combine_norm(
        torch.empty(4, 10240, dtype=torch.float16),
        torch.empty(4, 2560, dtype=torch.float16),
        torch.empty(4, 4, dtype=torch.float16),
        torch.empty(2560, dtype=torch.float16),
        1e-6,
        4,
    )
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["BLOCK_SIZE"] == 512 and kwargs["PREFETCH_WEIGHT"] is False
