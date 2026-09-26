# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission tests for SX_OPT_SHARED_GATE_ROWS and SX_OPT_QSA_TOPK_ROWS.

No GPU needed (the native ops are mocked); runs anywhere the patched vllm
imports (e.g. inside the image):

  /opt/venv/bin/python -m pytest -q sx_tests/b2-small-native/test_dispatch_cpu.py

Checks that the switches only widen the admitted row range, that "0" / an
old extension / instances built without the new attribute restore the
previous behaviour exactly, and that the QSA rows op is taken from the same
source (validation sidecar or wheel) as the existing selector.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b2_common  # noqa: E402,F401  (keeps ``import vllm`` on the installed package)
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Shared-expert gate (qwen2_moe.py)
# ---------------------------------------------------------------------------
def _gate_layer(rows_attr):
    gate = Mock(side_effect=lambda x: (torch.ones(x.shape[0], 1, dtype=x.dtype), None))
    gate.weight = torch.zeros(1, 2560, dtype=torch.float16)
    fields = dict(
        layer_idx=0,
        _sm70_exact_shared_expert_gate=True,
        gate_up_proj=NS(forward_fused_silu_and_mul=lambda x: x),
        down_proj=lambda x: (x.clone(), None),
        expert_gate=gate,
    )
    if rows_attr is not None:
        fields["_sx_shared_gate_rows"] = rows_attr
    return NS(**fields), gate


@pytest.mark.parametrize("rows", (1, 2, 4, 16, 17, 24, 32, 33, 64))
@pytest.mark.parametrize("rows_attr", (None, False, True))
def test_shared_gate_admission(monkeypatch, rows: int, rows_attr) -> None:
    from vllm import _sm70_ops as ops
    from vllm.model_executor.models import qwen2_moe

    monkeypatch.setattr(qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda l, i, x: x)
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: True)
    exact = Mock()
    monkeypatch.setattr(ops, "qwen38_shared_gate_exact_out", exact)
    layer, gate = _gate_layer(rows_attr)
    x = torch.ones(rows, 2560, dtype=torch.float16)
    result = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    use_fused = rows == 1 or (bool(rows_attr) and rows <= 32)
    assert exact.call_count == int(use_fused)
    assert gate.call_count == int(not use_fused)
    if use_fused:
        out_arg, x_arg, w_arg = exact.call_args.args
        assert out_arg.shape == (rows, 2560) and x_arg is x
        assert w_arg is gate.weight
    else:
        torch.testing.assert_close(result, torch.sigmoid(torch.ones_like(x)))


def test_shared_gate_rows_need_contiguous_rows(monkeypatch) -> None:
    from vllm import _sm70_ops as ops
    from vllm.model_executor.models import qwen2_moe

    monkeypatch.setattr(qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda l, i, x: x)
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: True)
    exact = Mock()
    monkeypatch.setattr(ops, "qwen38_shared_gate_exact_out", exact)
    layer, gate = _gate_layer(True)
    wide = torch.ones(8, 2 * 2560, dtype=torch.float16)
    x = wide[:, :2560]  # (8, 2560) rows with a 5120 stride
    assert not x.is_contiguous()
    layer.down_proj = lambda t: (torch.ones(8, 2560, dtype=torch.float16), None)
    qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    assert exact.call_count == 0 and gate.call_count == 1


def test_shared_gate_shape_helper_default_unchanged() -> None:
    from vllm.model_executor.models import qwen2_moe

    helper = qwen2_moe._sm70_fused_shared_expert_gate_shape_supported
    x17 = torch.empty(17, 2560, dtype=torch.float16)
    assert not helper(x17, torch.empty_like(x17))  # old default bound 16
    assert helper(x17, torch.empty_like(x17), max_tokens=32)
    x33 = torch.empty(33, 2560, dtype=torch.float16)
    assert not helper(x33, torch.empty_like(x33), max_tokens=32)


def test_shared_gate_capability_is_cached(monkeypatch) -> None:
    from vllm.model_executor.models import qwen2_moe

    probe = Mock(return_value=True)
    monkeypatch.setattr(qwen2_moe, "_sx_probe_shared_gate_rows", probe)
    monkeypatch.setattr(qwen2_moe, "_SX_SHARED_GATE_ROWS_CAPABLE", None)
    assert qwen2_moe._sx_shared_gate_rows_capable() is True
    assert qwen2_moe._sx_shared_gate_rows_capable() is True
    assert probe.call_count == 1


def test_shared_gate_probe_without_op(monkeypatch) -> None:
    from vllm import _sm70_ops as ops
    from vllm.model_executor.models import qwen2_moe

    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: False)
    assert qwen2_moe._sx_probe_shared_gate_rows() is False


def _construct_shared_expert(monkeypatch, qwen2_moe, *, spec, switch=True):
    """Build a Qwen2MoeMLP shared expert on CPU with the layers stubbed."""
    monkeypatch.setattr(
        qwen2_moe, "envs", NS(VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION=True)
    )
    monkeypatch.setattr(qwen2_moe, "MergedColumnParallelLinear", lambda *a, **k: NS())
    monkeypatch.setattr(qwen2_moe, "RowParallelLinear", lambda *a, **k: NS())
    monkeypatch.setattr(qwen2_moe, "SiluAndMul", lambda **k: NS())
    monkeypatch.setattr(
        qwen2_moe, "_sm70_force_shared_expert_silu_custom_op", lambda prefix: True
    )
    monkeypatch.setattr(
        qwen2_moe,
        "_sm70_fused_shared_expert_gate_module_supported",
        lambda gate_up, down: True,
    )
    monkeypatch.setattr(
        qwen2_moe,
        "get_current_vllm_config_or_none",
        lambda: NS(speculative_config=spec),
    )
    monkeypatch.setattr(qwen2_moe, "_SX_OPT_SHARED_GATE_ROWS", switch)
    capable = Mock(return_value=True)
    monkeypatch.setattr(qwen2_moe, "_sx_shared_gate_rows_capable", capable)
    layer = qwen2_moe.Qwen2MoeMLP(
        hidden_size=2560,
        intermediate_size=160,
        hidden_act="silu",
        reduce_results=False,
        expert_gate=NS(weight=torch.zeros(1, 2560, dtype=torch.float16)),
        prefix="model.layers.3.mlp.shared_expert",
    )
    return layer, capable


@pytest.mark.parametrize("switch", (True, False))
@pytest.mark.parametrize("speculative", (False, True))
def test_shared_gate_rows_construction_gate(monkeypatch, switch, speculative) -> None:
    """Rows gate only without speculative decoding (MTP etc. keep M == 1)."""
    from vllm.model_executor.models import qwen2_moe

    spec = object() if speculative else None
    layer, capable = _construct_shared_expert(
        monkeypatch, qwen2_moe, spec=spec, switch=switch
    )
    assert layer._sm70_exact_shared_expert_gate is True
    assert layer._sx_shared_gate_rows is (switch and not speculative)
    # The probe (one launch + host sync) never runs when rows cannot be used.
    assert capable.call_count == int(switch and not speculative)


def test_speculative_decoding_helper(monkeypatch) -> None:
    from vllm.model_executor.models import qwen2_moe

    monkeypatch.setattr(qwen2_moe, "get_current_vllm_config_or_none", lambda: None)
    assert qwen2_moe._sx_speculative_decoding_configured() is False
    monkeypatch.setattr(
        qwen2_moe,
        "get_current_vllm_config_or_none",
        lambda: NS(speculative_config=None),
    )
    assert qwen2_moe._sx_speculative_decoding_configured() is False
    monkeypatch.setattr(
        qwen2_moe,
        "get_current_vllm_config_or_none",
        lambda: NS(speculative_config=object()),
    )
    assert qwen2_moe._sx_speculative_decoding_configured() is True


# ---------------------------------------------------------------------------
# QSA top-k rows op selection (ops/qsa.py)
# ---------------------------------------------------------------------------
def _qsa():
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    return qsa


@pytest.mark.parametrize("rows", (0, 1, 2, 17, 24, 32, 33, 512))
@pytest.mark.parametrize("enabled", (True, False))
def test_qsa_topk_rows_admission(monkeypatch, rows: int, enabled: bool) -> None:
    qsa = _qsa()
    wheel_rows = object()
    monkeypatch.setattr(
        qsa.torch,
        "ops",
        NS(
            _C_qsa_sm70=NS(),
            _C=NS(
                qsa_lexicographic_topk=object(),
                qsa_lexicographic_topk_decode_rows=wheel_rows,
            ),
        ),
    )
    monkeypatch.setattr(qsa, "_SX_OPT_QSA_TOPK_ROWS", enabled)
    expected = wheel_rows if enabled and 2 <= rows <= 32 else None
    assert qsa._sx_qsa_topk_rows_op(rows) is expected


def test_qsa_topk_rows_op_source(monkeypatch) -> None:
    qsa = _qsa()
    wheel_rows, sidecar_rows = object(), object()
    # Old wheel (no rows op), no sidecar -> old launch.
    monkeypatch.setattr(
        qsa.torch,
        "ops",
        NS(_C_qsa_sm70=NS(), _C=NS(qsa_lexicographic_topk=object())),
    )
    assert qsa._sm70_qsa_lexicographic_topk_rows_op() is None
    # New wheel, no sidecar -> wheel rows op.
    qsa.torch.ops._C.qsa_lexicographic_topk_decode_rows = wheel_rows
    assert qsa._sm70_qsa_lexicographic_topk_rows_op() is wheel_rows
    # Older sidecar (selector only) -> never mix in the wheel rows op.
    qsa.torch.ops._C_qsa_sm70 = NS(qsa_lexicographic_topk=object())
    assert qsa._sm70_qsa_lexicographic_topk_rows_op() is None
    # Patched sidecar -> its own rows op.
    qsa.torch.ops._C_qsa_sm70.qsa_lexicographic_topk_decode_rows = sidecar_rows
    assert qsa._sm70_qsa_lexicographic_topk_rows_op() is sidecar_rows
