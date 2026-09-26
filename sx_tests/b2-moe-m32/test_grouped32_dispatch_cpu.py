# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the SX_OPT_MOE_GROUPED32 / SX_OPT_MOE_GROUPED_MASK gates.

Runs anywhere vllm imports (no GPU):
  /opt/venv/bin/python -m pytest -q sx_tests/b2-moe-m32/test_grouped32_dispatch_cpu.py

Asserts:
  * widths: layers without the load-time attribute keep exactly B8/B16
    (tests/quantization/test_sm70_nvfp4_grouped_decode_dispatch.py semantics);
    grouped_max_tokens=32 admits 8, 16 and 17..32 only; split 4 (B8), 8
    (B16) and the layer's B17+ split.
  * load-time contract: Qwen3.8 TP4 only, not with DBO, not with speculative
    decoding; split env parsing.
  * live-row discovery fails closed (None) for missing/ambiguous/mismatched
    query_start_loc views and returns the shared tail view otherwise; the
    result is cached per forward context.
  * v2 op lookup: absent ops -> None (cached); an explicit bad
    SX_OPT_MOE_GROUPED32_LIBRARY raises.
  * masking invariant (design_1 [C5] risk): the V2 runner pads the persistent
    query_start_loc buffer past the live requests with the unpadded token
    count every step (source tripwire) and at capture (InputBatch.make_dummy
    on CPU buffers). A stale tail smaller than the live count would zero live
    MoE rows, so a runner change that breaks this must fail here.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe


def _ctx(monkeypatch, metadata):
    ctx = NS(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(moe, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe, "get_forward_context", lambda: ctx)
    return ctx


def _decode_ctx(monkeypatch):
    return _ctx(monkeypatch, {"attn": NS(max_query_len=1)})


@pytest.mark.parametrize("tokens", [1, 2, 4, 7, 8, 9, 15, 16, 17, 20, 24, 31, 32, 33, 64])
def test_widths_legacy_layer_unchanged(monkeypatch, tokens):
    _decode_ctx(monkeypatch)
    layer = NS(sm70_nvfp4_grouped_decode=True)  # no batch-2 attributes
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    assert moe._use_grouped_decode(layer, x, ids) == (tokens in (8, 16))
    split = moe._grouped_decode_split(layer, x, ids)
    assert split == {8: 4, 16: 8}.get(tokens)


@pytest.mark.parametrize("split32", [8, 4])
@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 15, 16, 17, 18, 23, 24, 25, 32, 33, 48])
def test_widths_grouped32_layer(monkeypatch, tokens, split32):
    _decode_ctx(monkeypatch)
    layer = NS(
        sm70_nvfp4_grouped_decode=True,
        sm70_nvfp4_grouped_max_tokens=32,
        sm70_nvfp4_grouped32_split=split32,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    admitted = tokens in (8, 16) or 17 <= tokens <= 32
    assert moe._use_grouped_decode(layer, x, ids) == admitted
    expected = {8: 4, 16: 8}.get(tokens, split32 if admitted else None)
    assert moe._grouped_decode_split(layer, x, ids) == expected
    # Prefill / mixed metadata never admits any width.
    _ctx(monkeypatch, {"attn": NS(max_query_len=2)})
    assert moe._grouped_decode_split(layer, x, ids) is None


def test_grouped_switch_off_keeps_16(monkeypatch):
    _decode_ctx(monkeypatch)
    layer = NS(sm70_nvfp4_grouped_decode=True, sm70_nvfp4_grouped_max_tokens=16)
    for tokens in (17, 24, 32):
        x = torch.empty(tokens, 2560, dtype=torch.float16)
        ids = torch.empty(tokens, 10, dtype=torch.int32)
        assert moe._grouped_decode_split(layer, x, ids) is None


def _contract_layer(tp=4, experts=512, hidden=2560, inter=160, top_k=10):
    return NS(
        moe_config=NS(tp_size=tp),
        sm70_nvfp4_num_experts=experts,
        sm70_nvfp4_hidden_size=hidden,
        sm70_nvfp4_intermediate_size=inter,
        sm70_nvfp4_top_k=top_k,
    )


def _config(ubatching=False, spec=None):
    return NS(
        scheduler_config=NS(max_num_batched_tokens=8192),
        parallel_config=NS(use_ubatching=ubatching),
        speculative_config=spec,
    )


def test_contract_gate(monkeypatch):
    monkeypatch.setattr(moe, "get_current_vllm_config_or_none", lambda: _config())
    assert moe._grouped_v2_contract(_contract_layer())
    for layer in (
        _contract_layer(tp=2, inter=320),
        _contract_layer(tp=1, inter=640),
        _contract_layer(experts=256, hidden=2048, inter=128, top_k=8),
        _contract_layer(tp=8, experts=288, hidden=4096, inter=256, top_k=8),
    ):
        assert not moe._grouped_v2_contract(layer)
    monkeypatch.setattr(
        moe, "get_current_vllm_config_or_none", lambda: _config(ubatching=True)
    )
    assert not moe._grouped_v2_contract(_contract_layer())
    monkeypatch.setattr(
        moe,
        "get_current_vllm_config_or_none",
        lambda: _config(spec=NS(method="mtp")),
    )
    assert not moe._grouped_v2_contract(_contract_layer())


@pytest.mark.parametrize(
    "value,expected", [("8", 8), ("4", 4), (" 4 ", 4), ("5", 8), ("x", 8), ("", 8)]
)
def test_split_env(monkeypatch, value, expected):
    monkeypatch.setenv("SX_OPT_MOE_GROUPED32_SPLIT", value)
    assert moe._grouped32_split_from_env() == expected
    monkeypatch.delenv("SX_OPT_MOE_GROUPED32_SPLIT")
    assert moe._grouped32_split_from_env() == 8


# ----------------------------------------------------------------------------
# live-row discovery (CPU tensors stand in for the device buffer)
CPU = torch.device("cpu")


def test_live_rows_view_success():
    buf = torch.arange(34, dtype=torch.int32)  # persistent query_start_loc
    m = 24
    a = NS(query_start_loc=buf[: m + 1])  # e.g. QSA metadata
    b = NS(query_start_loc=buf[: m + 1])  # a different view object, same data
    gdn = NS(non_spec_query_start_loc=torch.zeros(3, dtype=torch.int32))
    view = moe._find_live_rows_view({"a": a, "b": b, "a2": a, "g": gdn}, m, CPU)
    assert view is not None and view.numel() == 1
    assert view.data_ptr() == buf[m:].data_ptr()
    buf[m] = 17
    assert int(view[0]) == 17  # a view, not a copy: replays see new values


@pytest.mark.parametrize(
    "case",
    ["none", "list", "empty", "no_qsl", "dtype", "numel", "ndim", "device",
     "conflict", "not_tensor"],
)
def test_live_rows_view_fails_closed(case):
    m = 16
    buf = torch.arange(34, dtype=torch.int32)
    good = NS(query_start_loc=buf[: m + 1])
    metadata = {
        "none": None,
        "list": [{"a": good}],
        "empty": {},
        "no_qsl": {"g": NS(num_decodes=16)},
        "dtype": {"a": good, "b": NS(query_start_loc=buf[: m + 1].long())},
        "numel": {"a": good, "b": NS(query_start_loc=buf[: m + 2])},
        "ndim": {"a": NS(query_start_loc=buf[: m + 1].view(1, -1))},
        "device": {"a": NS(query_start_loc=buf[: m + 1].to("meta"))},
        "conflict": {"a": good, "b": NS(query_start_loc=buf[: m + 1].clone())},
        "not_tensor": {"a": NS(query_start_loc=list(range(m + 1)))},
    }[case]
    assert moe._find_live_rows_view(metadata, m, CPU) is None


def test_live_rows_cached_per_forward(monkeypatch):
    buf = torch.arange(34, dtype=torch.int32)
    ctx = _ctx(monkeypatch, {"a": NS(max_query_len=1, query_start_loc=buf[:25])})
    calls = []
    real = moe._find_live_rows_view
    monkeypatch.setattr(
        moe, "_find_live_rows_view", lambda *a: calls.append(a) or real(*a)
    )
    x = torch.empty(24, 2560, dtype=torch.float16)
    first = moe._grouped_decode_live_rows(x)
    second = moe._grouped_decode_live_rows(x)
    assert first is second and len(calls) == 1
    assert ctx.additional_kwargs[moe._GROUPED_LIVE_ROWS_KEY][0] == 24
    # A different width in the same context re-discovers (numel check).
    assert moe._grouped_decode_live_rows(torch.empty(16, 2560)) is None
    assert len(calls) == 2


# ----------------------------------------------------------------------------
def test_v2_ops_absent(monkeypatch):
    monkeypatch.setattr(moe, "_grouped_v2_state", {})
    monkeypatch.setattr(moe, "_find_grouped_v2_namespace", lambda: None)
    monkeypatch.setattr(moe, "_grouped_v2_library_path", lambda: (None, False))
    assert moe._load_grouped_v2_ops() is None
    assert moe._grouped_v2_state["ops"] is None  # cached, no retry per layer


def test_v2_ops_bad_explicit_library(monkeypatch, tmp_path):
    bad = tmp_path / "missing.so"
    monkeypatch.setattr(moe, "_grouped_v2_state", {})
    monkeypatch.setattr(moe, "_find_grouped_v2_namespace", lambda: None)
    monkeypatch.setenv("SX_OPT_MOE_GROUPED32_LIBRARY", str(bad))
    with pytest.raises(RuntimeError):
        moe._load_grouped_v2_ops()


def test_v2_ops_bad_bundled_library_warns(monkeypatch, tmp_path):
    bad = tmp_path / "_sx_nvfp4_grouped32_C.so"
    bad.write_bytes(b"not an ELF")
    monkeypatch.setattr(moe, "_grouped_v2_state", {})
    monkeypatch.setattr(moe, "_find_grouped_v2_namespace", lambda: None)
    monkeypatch.setattr(moe, "_grouped_v2_library_path", lambda: (str(bad), False))
    assert moe._load_grouped_v2_ops() is None


# ----------------------------------------------------------------------------
# SX_OPT_MOE_GROUPED_MASK invariant: query_start_loc tail == live token count.
def _v2_runner_source() -> str:
    """Text of the installed V2 model runner (not executed); repo copy as a
    fallback when the vllm package cannot be imported (stubbed CPU runs)."""
    path = None
    try:
        spec = importlib.util.find_spec("vllm.v1.worker.gpu.model_runner")
        if spec is not None and spec.origin:
            path = Path(spec.origin)
    except (ImportError, ValueError):
        path = None
    if path is None or not path.exists():
        path = Path(__file__).resolve().parents[2] / (
            "vllm/v1/worker/gpu/model_runner.py"
        )
    return path.read_text(encoding="utf-8")


def test_runner_pads_query_start_loc_tail_every_step():
    src = re.sub(r"\s+", "", _v2_runner_source())
    # num_tokens is the unpadded scheduled count (the live decode rows) ...
    assert "num_tokens=scheduler_output.total_num_scheduled_tokens" in src
    # ... written into every entry past the live requests ...
    assert "query_start_loc_np=np.empty(self.max_num_reqs+1,dtype=np.int32)" in src
    assert "query_start_loc_np[num_reqs+1:]=num_tokens" in src
    # ... and the whole persistent buffer is refreshed every step ...
    assert (
        "async_copy_to_gpu(query_start_loc_np,out=self.input_buffers.query_start_loc)"
        in src
    )
    # ... while the attention metadata holds a view of that buffer, so the
    # FULL-graph tail pointer [num_reqs_padded] reads this step's live count.
    assert (
        "query_start_loc=self.input_buffers.query_start_loc[:num_reqs_padded+1]"
        in src
    )


def test_capture_dummy_batch_pads_tail():
    """FULL-graph capture/dummy batches write tail == width (no masking) and
    refresh the whole tail after a wider batch (capture order is widest first)."""
    try:
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    except Exception as exc:  # noqa: BLE001 - stubbed/partial vllm
        pytest.skip(f"V2 runner InputBatch unavailable: {exc}")
    buffers = InputBuffers(max_num_reqs=24, max_num_tokens=64, device=CPU)
    buffers.query_start_loc.fill_(-1)
    for width in (24, 16, 8, 17):
        InputBatch.make_dummy(width, width, buffers)
        qsl = buffers.query_start_loc
        assert qsl[: width + 1].tolist() == list(range(width + 1)), width
        assert (qsl[width:] == width).all(), (width, qsl.tolist())
        view = moe._find_live_rows_view(
            {"a": NS(query_start_loc=qsl[: width + 1])}, width, CPU
        )
        assert view is not None and int(view[0]) == width
