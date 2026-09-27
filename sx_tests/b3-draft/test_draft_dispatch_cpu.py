# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission tests for the batch-3 draft group (MTP-K1 / MTP-7 / MTP-8).

No GPU needed; runs anywhere the patched vllm imports (e.g. inside image
1.8.0-dev2 with the changed files bind-mounted):

  /opt/venv/bin/python -m pytest -q sx_tests/b3-draft/test_draft_dispatch_cpu.py

Checks
  * SX_OPT_MTP_DRAFT_TILES: the draft tile table is consulted only when the
    MTP drafter armed it; unarmed, "0", VLLM_SM70_MTP_MOE_TUNED_CONFIG=0 or the
    legacy-warmup context give exactly the previous selection (1Cat tile at
    M1/M5, None -> 0.0.3 tile elsewhere) for every width; M1 and M5 keep the
    1Cat tile bit-for-bit in the default table; other shapes are unaffected;
    inline / file overrides, precedence, invalid entries and the shipped JSON.
  * get_default_config / try_get_optimal_moe_config pick the table on SM70.
  * _sx_mtp_draft_contract: method mtp + SM70 + TP4 + FP16 + exact topology.
  * SX_OPT_MTP_DRAFT_GEMV: the draft install replaces exactly the role/shape
    planned projections (qkv / o / indexer / router / HC down), marks only
    combine HC modules, leaves the shared-expert gate and the MR9a fused-HC
    counter alone, and "0" / online QPN8 / a failed contract install nothing
    (the tile table stays armed with SX_OPT_MTP_DRAFT_GEMV=0).
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_draft_common as C  # noqa: E402
import torch  # noqa: E402

import vllm.envs as envs  # noqa: E402

ONECAT = {
    "BLOCK_SIZE_M": 2,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}
BM4 = {**ONECAT, "BLOCK_SIZE_M": 4}
# b3a validation: default table from the V100 sweep (M1-224 1Cat tile,
# M225-448 BM4/BN64/s2, M449-2048 BM16/BN64/s3).
SWEPT_BM4 = {**ONECAT, "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "num_stages": 2}
SWEPT_BM16 = {**ONECAT, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}
ALL_M = tuple(range(1, 260)) + (447, 448, 449, 450, 1024, 2048, 2049, 4096)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", True)
    for key in (
        "SX_OPT_MTP_DRAFT_TILES",
        "SX_OPT_MTP_DRAFT_TILES_TABLE",
        "SX_OPT_MTP_DRAFT_TILES_FILE",
        "SX_OPT_MTP_DRAFT_GEMV",
        "VLLM_BATCH_INVARIANT",
        "VLLM_SM70_QWEN38_FP16_GEMV",
        "VLLM_SM70_QWEN38_FUSED_HC_FP16",
    ):
        monkeypatch.delenv(key, raising=False)
    mod = C.fm()
    saved = mod._sx_mtp_draft_tiles_armed
    mod._sx_mtp_draft_tiles_armed = False
    mod.sx_mtp_draft_tiles_cache_clear()
    yield
    mod._sx_mtp_draft_tiles_armed = saved
    mod.sx_mtp_draft_tiles_cache_clear()


def _decode_config(m: int):
    return C.fm()._get_sm70_mtp_moe_decode_config(m, *C.SHAPE)


def _previous(m: int):
    return dict(ONECAT) if m in (1, 5) else None


# ---------------------------------------------------------------------------
# Tile table admission
# ---------------------------------------------------------------------------
def test_unarmed_is_previous_behaviour_for_every_width() -> None:
    for m in ALL_M:
        assert _decode_config(m) == _previous(m), m


def test_default_table_widths() -> None:
    with C.table_ctx():
        for m in ALL_M:
            got = _decode_config(m)
            if m <= 224:
                assert got == ONECAT, m
            elif m <= 448:
                assert got == SWEPT_BM4, m
            elif m <= 2048:
                assert got == SWEPT_BM16, m
            else:
                # Uncovered widths keep the previous selection.
                assert got == _previous(m), m
        # M1 and M5 (C1 k=4 draft) keep 1Cat's tile exactly.
        assert _decode_config(1) == ONECAT
        assert _decode_config(5) == ONECAT


@pytest.mark.parametrize("value", ["0", " 0 "])
def test_switch_off_restores_previous(value: str) -> None:
    with C.table_ctx(SX_OPT_MTP_DRAFT_TILES=value):
        for m in ALL_M:
            assert _decode_config(m) == _previous(m), m


def test_1cat_rollback_and_legacy_warmup_context_win(monkeypatch) -> None:
    with C.table_ctx():
        with C.fm().force_sm70_mtp_moe_legacy_config():
            assert all(_decode_config(m) is None for m in ALL_M)
        monkeypatch.setattr(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", False)
        assert all(_decode_config(m) is None for m in ALL_M)


@pytest.mark.parametrize(
    "shape",
    [
        (17, 256, 128, 2048, 8),
        (2, 128, 128, 2048, 8),
        (2, 256, 160, 2560, 10),
        (5, 512, 128, 2560, 10),
        (5, 512, 160, 2048, 10),
        (5, 512, 160, 2560, 8),
        (40, 512, 320, 2560, 10),
    ],
)
def test_other_shapes_unaffected_when_armed(shape) -> None:
    mod = C.fm()
    before = mod._get_sm70_mtp_moe_decode_config(*shape)
    with C.table_ctx():
        assert mod._get_sm70_mtp_moe_decode_config(*shape) == before


def test_qwen36_tile_unchanged_when_armed() -> None:
    mod = C.fm()
    with C.table_ctx():
        for m in range(2, 17):
            cfg = mod._get_sm70_mtp_moe_decode_config(m, 256, 128, 2048, 8)
            assert cfg is not None and cfg["BLOCK_SIZE_M"] == 8


def test_returned_config_is_a_copy() -> None:
    with C.table_ctx():
        first = _decode_config(20)
        first["BLOCK_SIZE_M"] = 99
        assert _decode_config(20) == ONECAT


def test_inline_table_precedence_and_exact_entries() -> None:
    table = "1-40:4x64x64x8x2,5:2x128x64x4x3,41-64:8x128x32x4x4x2;65-65:16x32x64x4x3"
    with C.table_ctx(SX_OPT_MTP_DRAFT_TILES_TABLE=table):
        source, parsed = C.fm().sx_mtp_draft_tile_table()
        assert source == "SX_OPT_MTP_DRAFT_TILES_TABLE"
        assert len(parsed) == 4
        # Exact M5 entry beats the covering range.
        assert _decode_config(5) == ONECAT
        assert _decode_config(4) == {
            **ONECAT,
            "BLOCK_SIZE_M": 4,
            "BLOCK_SIZE_N": 64,
            "num_warps": 8,
            "num_stages": 2,
        }
        assert _decode_config(50) == {
            **ONECAT,
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 2,
            "num_stages": 4,
        }
        assert _decode_config(65)["BLOCK_SIZE_M"] == 16
        # Not covered -> previous behaviour.
        assert _decode_config(66) is None


@pytest.mark.parametrize(
    "table",
    [
        "garbage",
        "1-32:3x128x64x4x3",  # BM not a power of two
        "1-32:2x128x64x4",  # too few fields
        "0-4:2x128x64x4x3",  # M must be >= 1
        "9-4:2x128x64x4x3",  # empty range
        "1-4:2x128x64x3x3",  # warps 3
    ],
)
def test_invalid_inline_table_falls_back(table: str) -> None:
    with C.table_ctx(SX_OPT_MTP_DRAFT_TILES_TABLE=table):
        source, _ = C.fm().sx_mtp_draft_tile_table()
        assert source != "SX_OPT_MTP_DRAFT_TILES_TABLE"
        assert _decode_config(20) == ONECAT
        assert _decode_config(40) == ONECAT
        assert _decode_config(300) == SWEPT_BM4


def test_file_table_list_and_mapping_formats(tmp_path) -> None:
    list_file = tmp_path / "list.json"
    list_file.write_text(
        json.dumps(
            {
                "shape": [512, 160, 2560, 10],
                "table": [
                    {"m": [1, 16], "config": dict(ONECAT)},
                    {"m": 17, "config": {**ONECAT, "BLOCK_SIZE_N": 64}},
                    {"m": "18-100", "config": {**ONECAT, "BLOCK_SIZE_M": 8}},
                    {"m": [3, 1], "config": dict(ONECAT)},  # invalid range
                    {"m": 101, "config": {**ONECAT, "SPLIT_K": 2}},  # invalid
                ],
            }
        )
    )
    with C.table_ctx(SX_OPT_MTP_DRAFT_TILES_FILE=str(list_file)):
        source, parsed = C.fm().sx_mtp_draft_tile_table()
        assert source == str(list_file) and len(parsed) == 3
        assert _decode_config(17)["BLOCK_SIZE_N"] == 64
        assert _decode_config(60)["BLOCK_SIZE_M"] == 8
        assert _decode_config(101) is None
        assert _decode_config(5) == ONECAT

    map_file = tmp_path / "map.json"
    map_file.write_text(
        json.dumps({"triton_version": "x", "1-8": dict(ONECAT), "9": dict(BM4)})
    )
    with C.table_ctx(SX_OPT_MTP_DRAFT_TILES_FILE=str(map_file)):
        assert _decode_config(9) == BM4
        assert _decode_config(10) is None
        assert _decode_config(5) == ONECAT


def test_file_with_wrong_shape_or_missing_falls_back(tmp_path) -> None:
    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps(
            {"shape": [256, 128, 2048, 8], "table": [{"m": [1, 999], "config": BM4}]}
        )
    )
    for path in (str(wrong), str(tmp_path / "missing.json")):
        with C.table_ctx(SX_OPT_MTP_DRAFT_TILES_FILE=path):
            source, _ = C.fm().sx_mtp_draft_tile_table()
            assert source != path
            assert _decode_config(20) == ONECAT


def test_inline_beats_file(tmp_path) -> None:
    file = tmp_path / "t.json"
    file.write_text(json.dumps({"table": [{"m": [1, 999], "config": BM4}]}))
    with C.table_ctx(
        SX_OPT_MTP_DRAFT_TILES_FILE=str(file),
        SX_OPT_MTP_DRAFT_TILES_TABLE="1-999:8x64x64x4x3",
    ):
        assert _decode_config(3)["BLOCK_SIZE_M"] == 8


def test_shipped_json_matches_builtin_table() -> None:
    mod = C.fm()
    path = mod._sx_default_mtp_draft_table_path()
    assert os.path.exists(path), path
    parsed = mod._sx_load_mtp_draft_table_file(path)
    builtin = tuple(sorted(mod._SX_MTP_DRAFT_BUILTIN_TABLE, key=lambda e: (e[1] - e[0], e[0])))
    assert parsed == builtin
    with C.table_ctx():
        assert mod.sx_mtp_draft_tile_table()[0] == path


def test_builtin_table_keeps_bk64_and_split_k1() -> None:
    for lo, hi, tile in C.fm()._SX_MTP_DRAFT_BUILTIN_TABLE:
        assert tile["BLOCK_SIZE_K"] == 64 and tile["SPLIT_K"] == 1, (lo, hi)
    lo_hi = sorted((lo, hi) for lo, hi, _ in C.fm()._SX_MTP_DRAFT_BUILTIN_TABLE)
    assert lo_hi[0][0] == 1 and lo_hi[-1][1] >= 120  # covers B*(k+1) <= 24*5
    for (_, hi), (lo, _) in zip(lo_hi, lo_hi[1:]):
        assert lo == hi + 1  # contiguous


def test_builtin_fallback_without_json_is_narrowest_first(monkeypatch, tmp_path) -> None:
    """Missing shipped JSON -> built-in table, with the same narrowest-first
    precedence as file / inline tables (a wide-first built-in edit must not
    shadow an exact entry)."""
    mod = C.fm()
    monkeypatch.setattr(
        mod, "_sx_default_mtp_draft_table_path", lambda: str(tmp_path / "none.json")
    )
    with C.table_ctx():
        source, table = mod.sx_mtp_draft_tile_table()
        assert source == "built-in"
        assert [(lo, hi) for lo, hi, _ in table] == [(1, 224), (225, 448), (449, 2048)]
        assert _decode_config(5) == ONECAT and _decode_config(40) == ONECAT
        assert _decode_config(300) == SWEPT_BM4 and _decode_config(600) == SWEPT_BM16
    monkeypatch.setattr(
        mod,
        "_SX_MTP_DRAFT_BUILTIN_TABLE",
        ((1, 128, dict(BM4)), (5, 5, dict(ONECAT))),
    )
    with C.table_ctx():
        assert _decode_config(5) == ONECAT
        assert _decode_config(6) == BM4


def test_arming_logs_and_returns_state() -> None:
    mod = C.fm()
    assert mod.arm_sm70_mtp_draft_moe_tiles(True) is True
    assert mod._sx_mtp_draft_tiles_armed
    assert mod.arm_sm70_mtp_draft_moe_tiles(False) is False
    assert _decode_config(20) is None


# ---------------------------------------------------------------------------
# get_default_config / try_get_optimal_moe_config on a fake SM70 platform
# ---------------------------------------------------------------------------
class _FakeSM70Platform:
    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def is_rocm():
        return False

    @staticmethod
    def has_device_capability(capability):
        return capability == 70


def test_get_default_config_selects_table_on_sm70(monkeypatch) -> None:
    mod = C.fm()
    monkeypatch.setattr(mod, "current_platform", _FakeSM70Platform())
    monkeypatch.setattr(envs, "VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG", True)
    legacy = mod.get_default_config(20, *C.SHAPE[:3], C.TOPK, None)
    assert legacy["BLOCK_SIZE_M"] == 16 and legacy["BLOCK_SIZE_N"] == 32
    assert "num_warps" not in legacy
    with C.table_ctx():
        assert mod.get_default_config(20, *C.SHAPE[:3], C.TOPK, None) == ONECAT
        assert mod.get_default_config(80, *C.SHAPE[:3], C.TOPK, None) == ONECAT
        assert mod.get_default_config(600, *C.SHAPE[:3], C.TOPK, None) == SWEPT_BM16
        big = mod.get_default_config(4096, *C.SHAPE[:3], C.TOPK, None)
        assert big["BLOCK_SIZE_M"] == 64  # M > 2048 prefill tile unchanged


def test_try_get_optimal_moe_config_path(monkeypatch) -> None:
    mod = C.fm()
    monkeypatch.setattr(mod, "current_platform", _FakeSM70Platform())
    monkeypatch.setattr(mod, "get_moe_configs", lambda *a, **k: None)
    with C.table_ctx():
        cfg = mod.try_get_optimal_moe_config(
            (C.E, 2 * C.I_LOCAL, C.HIDDEN), (C.E, C.HIDDEN, C.I_LOCAL), C.TOPK, None, 40
        )
        assert cfg == ONECAT
    cfg = mod.try_get_optimal_moe_config(
        (C.E, 2 * C.I_LOCAL, C.HIDDEN), (C.E, C.HIDDEN, C.I_LOCAL), C.TOPK, None, 40
    )
    assert cfg["BLOCK_SIZE_M"] == 16


# ---------------------------------------------------------------------------
# Draft contract / install (mtp.py)
# ---------------------------------------------------------------------------
def _mtp():
    from vllm.models.qwen4_exp.nvidia import mtp

    return mtp


def _text_config(**overrides):
    fields = dict(
        hidden_size=2560,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        num_attention_heads=24,
        num_key_value_heads=2,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    fields.update(overrides)
    return NS(**fields)


def _vllm_config(method="mtp", tp=4, pp=1, ep=False, dtype=torch.float16, **text):
    return NS(
        speculative_config=None if method is None else NS(method=method),
        model_config=NS(dtype=dtype, hf_text_config=_text_config(**text)),
        parallel_config=NS(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
        ),
    )


class _FakePlatform:
    def __init__(self, sm70: bool = True) -> None:
        self.sm70 = sm70

    def is_cuda(self):
        return True

    def is_device_capability(self, capability):
        return self.sm70 and capability in ((7, 0), 70)


@pytest.fixture
def sm70(monkeypatch):
    monkeypatch.setattr(_mtp(), "current_platform", _FakePlatform(True))


def test_contract_accepts_exact_mtp_lane(sm70) -> None:
    assert _mtp()._sx_mtp_draft_contract(_vllm_config())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method=None),
        dict(method="eagle"),
        dict(method="dflash"),
        dict(tp=2),
        dict(pp=2),
        dict(ep=True),
        dict(dtype=torch.bfloat16),
        dict(num_experts=256),
        dict(num_hidden_layers=40),
        dict(hidden_size=4096),
        dict(moe_intermediate_size=512),
    ],
)
def test_contract_rejects(sm70, kwargs) -> None:
    assert not _mtp()._sx_mtp_draft_contract(_vllm_config(**kwargs))


def test_contract_rejects_non_sm70(monkeypatch) -> None:
    monkeypatch.setattr(_mtp(), "current_platform", _FakePlatform(False))
    assert not _mtp()._sx_mtp_draft_contract(_vllm_config())


def _linear(prefix: str, out_features: int, in_features: int):
    # Validation fix (b3a): the weight parameters query the TP group even with
    # disable_tp=True, so stub the TP helpers (no torch.distributed on CPU).
    from unittest import mock

    import vllm.model_executor.layers.linear as linear_mod
    import vllm.model_executor.parameter as param_mod
    from vllm.model_executor.layers.linear import ReplicatedLinear

    patches = [
        mock.patch.object(mod, name, fn)
        for mod in (param_mod, linear_mod)
        for name, fn in (
            ("get_tensor_model_parallel_rank", lambda: 0),
            ("get_tensor_model_parallel_world_size", lambda: 1),
        )
        if hasattr(mod, name)
    ]
    for p in patches:
        p.start()
    try:
        return ReplicatedLinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
            prefix=prefix,
            disable_tp=True,
        )
    finally:
        for p in patches:
            p.stop()


class _HC(torch.nn.Module):
    def __init__(self, prefix: str, use_combine: bool = True) -> None:
        super().__init__()
        self.use_combine = use_combine
        self.lora_rank, self.hc_count, self.hidden_size = 320, 4, 2560
        if use_combine:
            self.input_mix_weight_down_block_inject = _linear(
                f"{prefix}.input_mix_weight_down_block_inject", 336, 10240
            )
        else:
            self.input_mix_weight_down = _linear(
                f"{prefix}.input_mix_weight_down", 320, 10240
            )
        self.input_mix_weight_up = _linear(f"{prefix}.input_mix_weight_up", 10240, 320)


class _SharedExpert(torch.nn.Module):
    def __init__(self, exact: bool) -> None:
        super().__init__()
        self._sm70_exact_shared_expert_gate = exact
        self._sx_shared_gate_rows = False


def _fake_draft():
    p = "mtp.layers.48"
    root = torch.nn.Module()
    root.fc_embedding = _linear("mtp.fc_embedding", 640, 2560)  # no plan
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = _linear(f"{p}.self_attn.qkv_proj", 3584, 2560)
    layer.self_attn.o_proj = _linear(f"{p}.self_attn.o_proj", 2560, 1536)
    layer.self_attn.indexer = torch.nn.Module()
    layer.self_attn.indexer.index_qk_proj = _linear(
        f"{p}.self_attn.indexer.index_qk_proj", 640, 2560
    )
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = _linear(f"{p}.mlp.gate", 512, 2560)
    layer.mlp.shared_expert_gate = _linear(f"{p}.mlp.shared_expert_gate", 1, 2560)
    layer.mlp.shared_expert = _SharedExpert(exact=True)
    layer.bad_shape = _linear(f"{p}.bad.self_attn.o_proj", 2560, 1024)
    layer.attn_hyper_connection = _HC(f"{p}.attn_hyper_connection")
    layer.mlp_hyper_connection = _HC(f"{p}.mlp_hyper_connection")
    root.layers = torch.nn.ModuleList([layer])
    root.hyper_connection_mixer = _HC("mtp.hyper_connection_mixer", use_combine=False)
    return root, layer


def test_install_replaces_exact_roles_and_marks_hc() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
        Qwen38SM70FP16LinearMethod,
    )

    counter = sm70_fp16_hc._SX_HC_FUSED_MODULES
    root, layer = _fake_draft()
    counts = _mtp()._sx_install_mtp_draft_fp16_routes(root)
    # qkv / o / indexer / router + the two HC down projections (the HC down
    # role has a GEMV plan too; the target installs it the same way).
    assert counts == {"gemv": 6, "hc": 2}
    for module in (
        layer.self_attn.qkv_proj,
        layer.self_attn.o_proj,
        layer.self_attn.indexer.index_qk_proj,
        layer.mlp.gate,
        layer.attn_hyper_connection.input_mix_weight_down_block_inject,
        layer.mlp_hyper_connection.input_mix_weight_down_block_inject,
    ):
        assert type(module.quant_method) is Qwen38SM70FP16LinearMethod
    for module in (
        root.fc_embedding,
        layer.mlp.shared_expert_gate,
        layer.bad_shape,
        layer.attn_hyper_connection.input_mix_weight_up,
        root.hyper_connection_mixer.input_mix_weight_down,
        root.hyper_connection_mixer.input_mix_weight_up,
    ):
        assert type(module.quant_method) is UnquantizedLinearMethod
    assert layer.attn_hyper_connection._sm70_qwen38_fp16_fused_hc
    assert layer.mlp_hyper_connection._sm70_qwen38_fp16_fused_hc
    assert not getattr(root.hyper_connection_mixer, "_sm70_qwen38_fp16_fused_hc", False)
    # The shared-expert gate is qwen2_moe's (SX_OPT_MTP_MOE_ROUTES).
    assert not layer.mlp.shared_expert._sx_shared_gate_rows
    # The target's MR9a counter is not touched by the draft.
    assert sm70_fp16_hc._SX_HC_FUSED_MODULES == counter


def test_install_skips_non_fp16_weights() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    root, layer = _fake_draft()
    with torch.no_grad():
        layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
    counts = _mtp()._sx_install_mtp_draft_fp16_routes(root)
    assert counts["gemv"] == 5
    assert type(layer.mlp.gate.quant_method) is UnquantizedLinearMethod


def _recorder(calls: list, result=None):
    def install(model, **kwargs):
        calls.append((model, kwargs))
        return {} if result is None else result

    return install


def test_prepare_arms_tiles_and_installs(monkeypatch, sm70) -> None:
    mtp = _mtp()
    calls = []
    monkeypatch.setattr(
        mtp, "_sx_install_mtp_draft_fp16_routes", _recorder(calls, {"x": 1})
    )
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", False)
    root = torch.nn.Module()
    assert mtp._sx_prepare_mtp_draft_sm70(root, _vllm_config()) == {"x": 1}
    assert calls == [(root, {"gemv": True, "hc": True})]
    assert C.fm()._sx_mtp_draft_tiles_armed


def test_prepare_gemv_switch_off_still_arms_tiles(monkeypatch, sm70) -> None:
    mtp = _mtp()
    calls = []
    monkeypatch.setattr(mtp, "_sx_install_mtp_draft_fp16_routes", _recorder(calls))
    monkeypatch.setenv("SX_OPT_MTP_DRAFT_GEMV", "0")
    assert mtp._sx_prepare_mtp_draft_sm70(torch.nn.Module(), _vllm_config()) == {}
    assert calls == []
    assert C.fm()._sx_mtp_draft_tiles_armed


def test_prepare_skips_install_under_online_qpn8(monkeypatch, sm70) -> None:
    mtp = _mtp()
    calls = []
    monkeypatch.setattr(mtp, "_sx_install_mtp_draft_fp16_routes", _recorder(calls))
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", True)
    assert mtp._sx_prepare_mtp_draft_sm70(torch.nn.Module(), _vllm_config()) == {}
    assert calls == []


def test_prepare_skips_install_under_batch_invariant(monkeypatch, sm70) -> None:
    """The opaque GEMV / fused-HC ops bypass linear_batch_invariant, so the
    drafter keeps UnquantizedLinearMethod (the tile table is bypassed by
    get_default_config itself in that mode)."""
    mtp = _mtp()
    calls = []
    monkeypatch.setattr(mtp, "_sx_install_mtp_draft_fp16_routes", _recorder(calls))
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert envs.VLLM_BATCH_INVARIANT
    assert mtp._sx_prepare_mtp_draft_sm70(torch.nn.Module(), _vllm_config()) == {}
    assert calls == []
    assert C.fm()._sx_mtp_draft_tiles_armed


@pytest.mark.parametrize("gemv_env", [None, "1", "0"])
@pytest.mark.parametrize("hc_env", [None, "1", "0"])
def test_prepare_respects_explicit_kernel_kill_switches(
    monkeypatch, sm70, gemv_env, hc_env
) -> None:
    """Unset = draft default (on); an explicit "0" of the global kernel-family
    switch keeps the drafter off that family too."""
    mtp = _mtp()
    calls = []
    monkeypatch.setattr(mtp, "_sx_install_mtp_draft_fp16_routes", _recorder(calls))
    monkeypatch.setattr(envs, "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", False)
    for name, value in (
        ("VLLM_SM70_QWEN38_FP16_GEMV", gemv_env),
        ("VLLM_SM70_QWEN38_FUSED_HC_FP16", hc_env),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    root = torch.nn.Module()
    mtp._sx_prepare_mtp_draft_sm70(root, _vllm_config())
    want_gemv, want_hc = gemv_env != "0", hc_env != "0"
    if not want_gemv and not want_hc:
        assert calls == []
    else:
        assert calls == [(root, {"gemv": want_gemv, "hc": want_hc})]
    assert C.fm()._sx_mtp_draft_tiles_armed


def test_install_route_families_are_independent() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
        Qwen38SM70FP16LinearMethod,
    )

    root, layer = _fake_draft()
    assert _mtp()._sx_install_mtp_draft_fp16_routes(root, gemv=False) == {
        "gemv": 0,
        "hc": 2,
    }
    assert type(layer.self_attn.qkv_proj.quant_method) is UnquantizedLinearMethod
    assert layer.attn_hyper_connection._sm70_qwen38_fp16_fused_hc

    root, layer = _fake_draft()
    assert _mtp()._sx_install_mtp_draft_fp16_routes(root, hc=False) == {
        "gemv": 6,
        "hc": 0,
    }
    assert type(layer.self_attn.qkv_proj.quant_method) is Qwen38SM70FP16LinearMethod
    assert not getattr(layer.attn_hyper_connection, "_sm70_qwen38_fp16_fused_hc", False)


@pytest.mark.parametrize("kwargs", [dict(method=None), dict(tp=2), dict(method="eagle")])
def test_prepare_outside_contract_does_nothing(monkeypatch, sm70, kwargs) -> None:
    mtp = _mtp()
    calls = []
    monkeypatch.setattr(mtp, "_sx_install_mtp_draft_fp16_routes", _recorder(calls))
    assert mtp._sx_prepare_mtp_draft_sm70(torch.nn.Module(), _vllm_config(**kwargs)) == {}
    assert calls == []
    assert not C.fm()._sx_mtp_draft_tiles_armed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
