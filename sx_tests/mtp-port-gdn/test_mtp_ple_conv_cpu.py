# SPDX-License-Identifier: Apache-2.0
"""Fused MTP4 PLE rollback/conv (upstream 763189d9a): gate and semantics on CPU.

Run (CPU, no vLLM install needed; see ``_port_boot.py``):

    python -m pytest -q sx_tests/mtp-port-gdn/test_mtp_ple_conv_cpu.py

``Qwen4ExpPLELayer._short_conv_dilated_spec_batched`` and the op probe are
cut out of the real ple_layer.py. ``_C::qwen38_ple_spec_sm70_out`` is
registered on CPU with the schema string taken from
csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu and implemented as a
line-by-line Python emulation of that CUDA kernel (rollback, 9 + 5 history,
four FMA taps in FP32, FP16 conv boundary, FP32 SiLU, FP16 output, state
commit). The real kernel's output/state bits are checked against the
generic path on a V100 by ``tests/kernels/test_sm70_mtp_ple_conv.py``.

Asserted:
* the gate admits exactly upstream's case (VLLM_SM70_MTP_PLE_CONV, one
  verify request, 5 query rows, M5/M10, H10240, FP16 input/weight, FP16/FP32
  [states, 10240, 13] cache, int32 contiguous metadata, SM70, op present)
  and every deviation keeps the generic path;
* over 30 successive rollback/commit rounds (null and live state IDs,
  accepted 0/1/3/5/8, query lengths 0/1/3/5, both cache layouts, both cache
  dtypes, M5 and M10 padding) the kernel semantics keep every state bit of
  the generic path, zero the padded rows and agree with the generic outputs
  to FP16 rounding (CPU conv1d differs from the kernel's FMA order);
* the Python call matches the registered schema; CMakeLists builds the file.
"""

from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _port_boot as boot  # noqa: E402

HIDDEN = 10240
OP = "qwen38_ple_spec_sm70_out"
CALLS = {"op": 0}


class CudaLike(torch.Tensor):
    @property
    def is_cuda(self):  # type: ignore[override]
        return True


def cuda_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.Tensor._make_subclass(CudaLike, tensor)


def emulate_kernel(output, state, x, weight, ids, starts, accepted):
    """Python twin of ple_conv<State> (one CTA column = all channels here)."""
    CALLS["op"] += 1
    x = x.as_subclass(torch.Tensor)
    state = state.as_subclass(torch.Tensor)
    output = output.as_subclass(torch.Tensor)
    rows = x.shape[0]
    sid = int(ids[0])
    count = int(starts[1]) - int(starts[0])
    rollback = max(0, min(4, int(accepted[0]) - 1)) if sid else 0
    zero = torch.zeros(HIDDEN, dtype=torch.float16)
    h = []
    for t in range(9):
        h.append(state[sid, :, rollback + t].to(torch.float16) if sid else zero)
    for t in range(5):
        h.append(x[t] if t < count else zero)
    w = weight.as_subclass(torch.Tensor)
    for r in range(5):
        v = torch.zeros(HIDDEN, dtype=torch.float32)
        for k in range(4):
            # __fmaf_rn: exact product plus accumulator, one FP32 rounding.
            v = (h[r + k * 3].double() * w[:, k].double() + v.double()).float()
        v = v.half().float()
        silu = (v / (1.0 + torch.exp(-v))).half()
        output[r] = silu if r < count else zero
    output[5:rows] = 0
    if sid:
        for t in range(13):
            if t < 8 + count:
                state[sid, :, t] = h[t + 1].to(state.dtype)


def _schema() -> str:
    source = boot.read(boot.PLE_CU)
    match = re.search(r'm\.def\(\s*"(qwen38_ple_spec_sm70_out\(.*?)"\s*\);', re.sub(
        r'"\s*\n\s*"', "", source
    ), re.S)
    assert match, "schema not found in qwen38_ple_spec_sm70.cu"
    return match.group(1)


_LIB = None


def _register_op():
    global _LIB
    if hasattr(torch.ops._C, OP):
        return
    _LIB = torch.library.Library("_C", "FRAGMENT")
    _LIB.define(_schema())
    _LIB.impl(OP, emulate_kernel, "CPU")


_register_op()


@pytest.fixture
def ple():
    state = {"env": True, "sm70": True}
    utils_src = boot.read(boot.BACKEND_UTILS)
    null_block = int(re.search(r"(?m)^NULL_BLOCK_ID = (\d+)$", utils_src)[1])
    namespace = {
        "torch": torch,
        "F": F,
        "NULL_BLOCK_ID": null_block,
        "envs": None,
        "current_platform": SimpleNamespace(
            is_device_capability=lambda cap: state["sm70"] and tuple(cap) == (7, 0)
        ),
        "logger": SimpleNamespace(
            info_once=lambda *a, **k: None, warning_once=lambda *a, **k: None
        ),
    }

    class Envs:
        @property
        def VLLM_SM70_MTP_PLE_CONV(self):  # noqa: N802 - mirrors vllm.envs
            return state["env"]

    namespace["envs"] = Envs()
    boot.cut(boot.PLE, ("_SM70_PLE_SPEC_CONV_OP", "_sm70_ple_spec_conv_op_available"), namespace)
    fn = boot.cut_method(
        boot.PLE, "Qwen4ExpPLELayer", "_short_conv_dilated_spec_batched", namespace
    )
    layer = SimpleNamespace(conv_state_len=9, short_conv_dilation=3)
    CALLS["op"] = 0
    return SimpleNamespace(fn=fn, layer=layer, state=state, ns=namespace)


def _inputs(rows=5, state_dtype=torch.float16, time_major=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    shape = (3, 13, HIDDEN) if time_major else (3, HIDDEN, 13)
    initial = torch.randn(shape, generator=gen).to(state_dtype)
    x = torch.randn(rows, HIDDEN, generator=gen).half()
    weight = (torch.randn(HIDDEN, 4, generator=gen) * 0.1).half()
    return initial, x, weight


def _state_view(storage: torch.Tensor, time_major: bool) -> torch.Tensor:
    return storage.transpose(1, 2) if time_major else storage


def _call(ple, x, state, weight, ids, starts, accepted, *, query_len=5, wrap=True):
    args = (x, state, weight, ids, starts, accepted)
    if wrap:
        args = tuple(cuda_like(t) for t in args)
    out = ple.fn(ple.layer, *args, query_len)
    return out.as_subclass(torch.Tensor)


def _meta(sid, count, acc):
    ids = torch.tensor([sid], dtype=torch.int32)
    starts = torch.tensor([0, count], dtype=torch.int32)
    accepted = torch.tensor([acc], dtype=torch.int32)
    return ids, starts, accepted


@pytest.mark.parametrize("rows", [5, 10])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("time_major", [False, True])
def test_successive_rounds_match_generic(ple, rows, state_dtype, time_major):
    initial, x, weight = _inputs(rows, state_dtype, time_major)
    fused_storage, generic_storage = initial.clone(), initial.clone()
    fused_state = _state_view(fused_storage, time_major)
    generic_state = _state_view(generic_storage, time_major)
    gen = torch.Generator().manual_seed(20260927)
    for step in range(30):
        count = (0, 1, 3, 5)[step % 4]
        sid = 0 if step % 7 == 0 else 1 + step % 2
        acc = (0, 1, 3, 5, 8)[step % 5]
        scale = (0.001, 0.03, 0.1, 1.0, 3.0, 30.0)[step % 6]
        x = (torch.randn(rows, HIDDEN, generator=gen) * scale).half()
        before = CALLS["op"]
        ple.state["env"] = True
        fused = _call(ple, x, fused_state, weight, *_meta(sid, count, acc))
        assert CALLS["op"] == before + 1
        ple.state["env"] = False
        generic = _call(ple, x, generic_state, weight, *_meta(sid, count, acc))
        assert CALLS["op"] == before + 1
        bits = torch.int16 if state_dtype == torch.float16 else torch.int32
        assert torch.equal(
            fused_storage.view(bits), generic_storage.view(bits)
        ), f"state differs at step {step}"
        assert fused.shape == generic.shape == x.shape
        assert torch.count_nonzero(fused[count:]) == 0
        assert torch.count_nonzero(generic[count:]) == 0
        torch.testing.assert_close(
            fused[:count].float(), generic[:count].float(), rtol=4e-3, atol=4e-3
        )


def _base_case():
    initial, x, weight = _inputs(5)
    return initial, x, weight, *_meta(1, 5, 3)


@pytest.mark.parametrize(
    "label",
    [
        "env off",
        "not sm70",
        "op missing",
        "two requests",
        "k3 query len",
        "15 rows",
        "fp32 input",
        "int64 ids",
        "non-contiguous x",
        "bf16 state",
        "state width 14",
        "not cuda",
        "dilation 2",
    ],
)
def test_gate_rejections_keep_generic(ple, label):
    state, x, weight, ids, starts, accepted = _base_case()
    query_len, wrap = 5, True
    if label == "env off":
        ple.state["env"] = False
    elif label == "not sm70":
        ple.state["sm70"] = False
    elif label == "op missing":
        ple.ns["_SM70_PLE_SPEC_CONV_OP"] = False
    elif label == "two requests":
        ids = torch.tensor([1, 2], dtype=torch.int32)
        starts = torch.tensor([0, 5, 10], dtype=torch.int32)
        accepted = torch.tensor([1, 2], dtype=torch.int32)
        x = torch.cat((x, x))
    elif label == "k3 query len":
        query_len = 4
        x = x[:4].contiguous()
        starts = torch.tensor([0, 4], dtype=torch.int32)
    elif label == "15 rows":
        x = torch.cat((x, x, x))
    elif label == "fp32 input":
        x = x.float()
    elif label == "int64 ids":
        ids = ids.long()
    elif label == "non-contiguous x":
        x = torch.cat((x, x), dim=1)[:, ::2]
    elif label == "bf16 state":
        state = state.bfloat16()
    elif label == "state width 14":
        state = torch.cat((state, state[..., :1]), dim=-1)
    elif label == "not cuda":
        wrap = False
    elif label == "dilation 2":
        ple.layer.short_conv_dilation = 2
    try:
        _call(ple, x, state, weight.float() if label == "fp32 input" else weight,
              ids, starts, accepted, query_len=query_len, wrap=wrap)
    except RuntimeError:
        # The generic path may reject a geometry the gate refused (e.g. the
        # 14-wide cache); what matters is that the fused op was not called.
        pass
    assert CALLS["op"] == 0, label


def test_admitted_case_calls_op_once(ple):
    state, x, weight, ids, starts, accepted = _base_case()
    _call(ple, x, state, weight, ids, starts, accepted)
    assert CALLS["op"] == 1


def test_schema_and_build_entry():
    schema = _schema()
    assert schema.startswith(
        "qwen38_ple_spec_sm70_out(Tensor(a!) output, Tensor(b!) state, Tensor x, "
        "Tensor weight, Tensor ids, Tensor starts, Tensor accepted) -> ()"
    )
    cu = boot.read(boot.PLE_CU)
    assert 'm.impl("qwen38_ple_spec_sm70_out", &ple_spec_conv);' in cu
    assert "TORCH_LIBRARY_FRAGMENT(_C, m)" in cu
    assert '"${VLLM_SM70_TURBOMIND_ROOT}/ops/qwen38_ple_spec_sm70.cu"' in boot.read(
        boot.CMAKE
    )
    source = boot.read(boot.PLE)
    call = re.search(r"torch\.ops\._C\.qwen38_ple_spec_sm70_out\((.*?)\)", source, re.S)
    args = [a.strip() for a in call.group(1).split(",") if a.strip()]
    assert args == [
        "output",
        "conv_state",
        "x_spec",
        "conv_weights",
        "spec_state_indices_tensor",
        "spec_query_start_loc",
        "num_accepted_tokens",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
