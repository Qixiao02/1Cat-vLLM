# SPDX-License-Identifier: Apache-2.0
"""Mixed-QKV fused GDN target verification: request and dispatch (CPU).

Run (CPU, no vLLM install needed; see ``_port_boot.py``):

    python -m pytest -q sx_tests/mtp-port-gdn/test_mtp_gdn_mixed_qkv_dispatch_cpu.py

The real ``QwenGatedDeltaNetAttention._forward_core`` and
``rearrange_mixed_qkv`` are cut out of qwen_gdn_linear_attn.py and run on CPU
tensors (a tensor subclass reports ``is_cuda``) with recording stubs for the
conv/recurrent kernels. The arithmetic of both recurrent loaders is the same
Triton kernel and is checked bit for bit on a V100 by
``tests/kernels/test_sm70_mtp_gdn_mixed_qkv.py``.

Asserted:
* the MTP-lane request (``enable_sx_mtp_mixed_qkv_verify``) or the upstream
  request (``enable_sm70_fused_sigmoid_mixed_qkv``) routes pure verify batches
  of 2..16 rows to ``fused_sigmoid_gating_delta_rule_update_mixed_qkv`` with
  the conv output, the spec query offsets, the state rows and the state-slot
  selectors, and skips the Q/K/V split;
* everything outside upstream's gate keeps the split-copy route: 20 rows,
  no request, not SM70, other head geometry, FP16 state, DDTree parents,
  mixed spec/decode batches, the DFlash2 packed verifier;
* the MTP-lane request never changes non-spec decode routing (packed
  recurrent decode and the plain decode recurrence stay as they were);
* ``_sx_mtp_mixed_qkv_verify_lane``: lane contract + Volta, switch, fail-closed.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _port_boot as boot  # noqa: E402

PREFIX = "model.layers.0.linear_attn"
QKV = 2560  # 4 x 128 (q) + 4 x 128 (k) + 12 x 128 (v) per TP4 rank


class CudaLike(torch.Tensor):
    """CPU tensor that passes the ``is_cuda`` checks of the request."""

    @property
    def is_cuda(self):  # type: ignore[override]
        return True


def cuda_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.Tensor._make_subclass(CudaLike, tensor)


def _gdn_metadata_cls():
    namespace = {"dataclass": dataclass, "torch": torch}
    boot.cut(boot.GDN_ATTN, ("GDN_SPEC_METADATA_TENSORS", "GDNAttentionMetadata"), namespace)
    return namespace["GDNAttentionMetadata"]


GDNAttentionMetadata = _gdn_metadata_cls()


class Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.messages: list[str] = []

    def names(self):
        return [name for name, _ in self.calls]


def _make_core(rec: Recorder, sm70: dict):
    def conv_update(x, conv_state, weight, bias, activation, **kwargs):
        rec.calls.append(("conv_update", dict(x=x, **kwargs)))
        return x.clone()

    def recurrent(**kwargs):
        rec.calls.append(("split_copy", kwargs))
        value = kwargs["v"]
        return torch.zeros(1, *value.shape[1:]), kwargs["initial_state"]

    def recurrent_mixed(**kwargs):
        rec.calls.append(("mixed_qkv", kwargs))
        tokens = kwargs["mixed_qkv"].shape[0]
        out = torch.zeros(1, tokens, kwargs["num_v_heads"], kwargs["head_v_dim"])
        return out, kwargs["initial_state"]

    namespace = {
        "torch": torch,
        "GDNAttentionMetadata": GDNAttentionMetadata,
        "_ddtree_parent_ids_require_branch": lambda *args: False,
        "_encode_layer_name": lambda prefix: prefix,
        "_log_runtime_route_once": lambda message, *a: rec.messages.append(message),
        "_sm70_gdn_graph_buffer_copy": lambda *a, **k: None,
        "_sm70_gdn_graph_buffer_copy_state_slice": lambda *a, **k: None,
        "_sm70_gdn_prefill_profile_start": lambda: 0.0,
        "_sm70_gdn_prefill_profile_end": lambda *a, **k: None,
        "_sm70_flashqla_original_prefill_enabled": lambda: False,
        "causal_conv1d_update": conv_update,
        "current_platform": SimpleNamespace(
            is_device_capability=lambda capability: sm70["on"] and capability == 70
        ),
        "fused_sigmoid_gating_delta_rule_update": recurrent,
        "fused_sigmoid_gating_delta_rule_update_mixed_qkv": recurrent_mixed,
        "is_conv_state_dim_first": lambda: True,
        "logger": SimpleNamespace(warning=lambda *a, **k: None),
    }
    forward_core = boot.cut_method(
        boot.GDN_LAYER, "QwenGatedDeltaNetAttention", "_forward_core", namespace
    )
    rearrange = boot.cut_method(
        boot.GDN_LAYER, "QwenGatedDeltaNetAttention", "rearrange_mixed_qkv", namespace
    )
    return forward_core, rearrange, namespace


def _layer(rearrange, rec, *, sx=False, upstream=False, **overrides):
    values = dict(
        prefix=PREFIX,
        tp_size=4,
        num_k_heads=16,
        num_v_heads=48,
        key_dim=2048,
        value_dim=6144,
        head_k_dim=128,
        head_v_dim=128,
        conv1d=SimpleNamespace(weight=torch.zeros(QKV, 1, 4), bias=None),
        activation="silu",
        A_log=torch.zeros(12),
        dt_bias=torch.zeros(12),
        enable_packed_recurrent_decode=False,
        enable_sm70_dflash2_fused_qkv_pack=False,
        enable_sm70_fused_sigmoid_mixed_qkv=upstream,
        enable_sx_mtp_mixed_qkv_verify=sx,
        _can_use_dflash2_packed_gdn_verify=lambda **kwargs: False,
    )
    values.update(overrides)
    layer = SimpleNamespace(**values)
    layer.rearrange_mixed_qkv = types.MethodType(rearrange, layer)
    layer._forward_core_decode_non_spec = lambda **kwargs: rec.calls.append(
        ("packed_decode", kwargs)
    )
    layer._forward_dflash2_packed_gdn_verify = lambda **kwargs: (
        rec.calls.append(("dflash2_packed", kwargs))
    )
    return layer


def _spec_metadata(batch: int, width: int, *, ssm_rows: int = 64, **overrides):
    tokens = batch * width
    values = dict(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=batch,
        num_spec_decode_tokens=tokens,
        num_actual_tokens=tokens,
        spec_query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * width,
        spec_state_indices_tensor=torch.arange(batch * width, dtype=torch.int32)
        .reshape(batch, width)
        .add(1),
        spec_sequence_masks=torch.ones(batch, dtype=torch.bool),
        spec_token_indx=torch.arange(tokens, dtype=torch.int32),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32),
        num_accepted_tokens=torch.ones(batch, dtype=torch.int32),
        spec_state_slot_selectors=torch.arange(batch, dtype=torch.int32) % width + 1,
    )
    values.update(overrides)
    return GDNAttentionMetadata(**values)


def _run(forward_core, namespace, layer, metadata, *, ssm_dtype=torch.float32):
    tokens = metadata.num_actual_tokens
    namespace["get_forward_context"] = lambda: SimpleNamespace(
        attn_metadata={PREFIX: metadata}
    )
    width = (2 * layer.key_dim + layer.value_dim) // layer.tp_size
    heads = layer.num_v_heads // layer.tp_size
    mixed_qkv = cuda_like(torch.randn(tokens, width).half())
    b = torch.randn(tokens, heads).half()
    a = torch.randn(tokens, heads).half()
    core_attn_out = torch.full((tokens, heads, 128), 7.0)
    conv_state = torch.zeros(64, width, 7).half()
    ssm_state = torch.zeros(64, heads, 128, 128, dtype=ssm_dtype)
    forward_core(layer, mixed_qkv, b, a, core_attn_out, (conv_state, ssm_state))
    return core_attn_out


@pytest.fixture
def core():
    rec = Recorder()
    sm70 = {"on": True}
    forward_core, rearrange, namespace = _make_core(rec, sm70)
    return SimpleNamespace(
        rec=rec, sm70=sm70, forward=forward_core, rearrange=rearrange, ns=namespace
    )


@pytest.mark.parametrize(
    "batch,width", [(1, 2), (1, 5), (2, 5), (3, 5), (4, 4), (2, 8), (1, 8)]
)
@pytest.mark.parametrize("request_kind", ["sx", "upstream"])
def test_pure_verify_up_to_16_rows_takes_mixed_loader(core, batch, width, request_kind):
    layer = _layer(core.rearrange, core.rec, **{request_kind: True})
    metadata = _spec_metadata(batch, width)
    out = _run(core.forward, core.ns, layer, metadata)
    assert core.rec.names() == ["conv_update", "mixed_qkv"]
    conv_x = core.rec.calls[0][1]["x"]
    kwargs = core.rec.calls[1][1]
    # The conv output itself, not a re-packed copy.
    assert torch.equal(kwargs["mixed_qkv"], conv_x)
    assert kwargs["mixed_qkv"].shape == (batch * width, QKV)
    assert (kwargs["num_q_heads"], kwargs["num_v_heads"]) == (4, 12)
    assert (kwargs["head_k_dim"], kwargs["head_v_dim"]) == (128, 128)
    assert torch.equal(kwargs["cu_seqlens"], metadata.spec_query_start_loc)
    assert kwargs["ssm_state_indices"] is metadata.spec_state_indices_tensor
    assert kwargs["num_accepted_tokens"] is metadata.spec_state_slot_selectors
    assert kwargs["inplace_final_state"] and kwargs["use_qk_l2norm_in_kernel"]
    assert torch.count_nonzero(out) == 0  # the route's output was written back
    assert "SM70 mixed-QKV fused GDN target-verification route hit." in (
        core.rec.messages
    )


@pytest.mark.parametrize(
    "label, layer_overrides, metadata_args, sm70, ssm_dtype",
    [
        ("no request", dict(), (1, 5), True, torch.float32),
        ("20 rows", dict(sx=True), (4, 5), True, torch.float32),
        ("24 rows", dict(sx=True), (3, 8), True, torch.float32),
        ("not sm70", dict(sx=True), (1, 5), False, torch.float32),
        ("fp16 state", dict(sx=True), (1, 5), True, torch.float16),
        ("8 kv heads", dict(sx=True, num_k_heads=32), (1, 5), True, torch.float32),
    ],
)
def test_outside_upstream_gate_keeps_split_copy(
    core, label, layer_overrides, metadata_args, sm70, ssm_dtype
):
    core.sm70["on"] = sm70
    layer = _layer(core.rearrange, core.rec, **layer_overrides)
    layer.key_dim = layer.num_k_heads * layer.head_k_dim
    tokens = metadata_args[0] * metadata_args[1]
    _run(core.forward, core.ns, layer, _spec_metadata(*metadata_args), ssm_dtype=ssm_dtype)
    assert core.rec.names() == ["conv_update", "split_copy"], label
    assert core.rec.calls[1][1]["q"].shape[1] == tokens


def test_ddtree_parents_keep_split_copy(core):
    layer = _layer(core.rearrange, core.rec, sx=True)
    metadata = _spec_metadata(1, 5, ddtree_parent_ids=torch.zeros(1, 5, dtype=torch.int32))
    metadata.ddtree_num_tree_tokens_cpu = torch.tensor([5])
    _run(core.forward, core.ns, layer, metadata)
    assert core.rec.names() == ["conv_update", "split_copy"]


def test_dflash2_packed_verifier_keeps_priority(core):
    layer = _layer(
        core.rearrange,
        core.rec,
        sx=True,
        _can_use_dflash2_packed_gdn_verify=lambda **kwargs: True,
    )
    _run(core.forward, core.ns, layer, _spec_metadata(1, 5))
    assert core.rec.names() == ["conv_update", "dflash2_packed"]


def test_mixed_spec_and_decode_batch_keeps_split_copy(core):
    # Legacy mixed routing: one verify request (5 rows) and one plain decode.
    layer = _layer(core.rearrange, core.rec, sx=True)
    metadata = _spec_metadata(
        1,
        5,
        num_decodes=1,
        num_decode_tokens=1,
        num_actual_tokens=6,
        spec_sequence_masks=torch.tensor([True, False]),
        # int64 here: CPU index_copy_ rejects the int32 indices CUDA accepts.
        spec_token_indx=torch.arange(5),
        non_spec_token_indx=torch.tensor([5]),
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([9], dtype=torch.int32),
    )
    _run(core.forward, core.ns, layer, metadata)
    assert core.rec.names() == ["conv_update", "conv_update", "split_copy", "split_copy"]


@pytest.mark.parametrize("packed", [False, True])
def test_lane_request_never_reroutes_non_spec_decode(core, packed):
    layer = _layer(core.rearrange, core.rec, sx=True, enable_packed_recurrent_decode=packed)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([3, 4], dtype=torch.int32),
    )
    _run(core.forward, core.ns, layer, metadata)
    if packed:
        assert core.rec.names() == ["packed_decode"]
    else:
        assert core.rec.names() == ["conv_update", "split_copy"]
    assert "mixed_qkv" not in core.rec.names()


@pytest.fixture
def lane_ns(monkeypatch):
    state = {"contract": True, "volta": True, "raise": False}

    def contract(model_config, speculative_config, parallel_config):
        if state["raise"]:
            raise RuntimeError("partial config")
        state["args"] = (model_config, speculative_config, parallel_config)
        return state["contract"]

    fake = types.ModuleType("vllm.config.vllm")
    fake._is_sm70_qwen38_mtp_lane_contract = contract
    for name in ("vllm", "vllm.config"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", fake)
    namespace = {
        "_SX_OPT_MTP_GDN_MIXED_QKV": True,
        "_sm70_current_device_is_volta": lambda: state["volta"],
    }
    boot.cut(boot.GDN_LAYER, ("_sx_mtp_mixed_qkv_verify_lane",), namespace)
    namespace["state"] = state
    return namespace


def test_lane_helper(lane_ns):
    helper = lane_ns["_sx_mtp_mixed_qkv_verify_lane"]
    state = lane_ns["state"]
    cfg = SimpleNamespace(model_config="m", speculative_config="s", parallel_config="p")
    assert helper(cfg) is True
    assert state["args"] == ("m", "s", "p")
    state["volta"] = False
    assert helper(cfg) is False
    state["volta"] = True
    state["contract"] = False
    assert helper(cfg) is False
    state["contract"] = True
    state["raise"] = True
    assert helper(cfg) is False
    state["raise"] = False
    lane_ns["_SX_OPT_MTP_GDN_MIXED_QKV"] = False
    assert helper(cfg) is False


def test_switch_is_read_once_with_default_on():
    source = boot.read(boot.GDN_LAYER)
    assert (
        '_SX_OPT_MTP_GDN_MIXED_QKV = os.environ.get("SX_OPT_MTP_GDN_MIXED_QKV", "1")'
        ' != "0"' in source
    )
    # The layer arms the flag from the lane helper at construction.
    assert "self.enable_sx_mtp_mixed_qkv_verify = _sx_mtp_mixed_qkv_verify_lane(" in (
        source
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
