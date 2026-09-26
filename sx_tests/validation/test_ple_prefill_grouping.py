# SPDX-License-Identifier: Apache-2.0
"""opt180dev1 validation fix: bounded PLE short-conv prefill packing.

SX_OPT_PLE_PREFILL_MAX_PACKED_ROWS (default 8192) splits the zero-padded
[num_prefills, max_len, 10240] packing of the dilated PLE short-conv prefill
into consecutive request groups when num_prefills * max_len exceeds the bound
(multi-block align chunks put ~7.8K-token chunks next to other prefills and
OOMed the engine at gmu 0.90).

Asserted (1x V100):
  * grouped == unsplit (bound 0 = old path): output and conv-state write-back,
    compared bitwise (int16 view) for realistic step layouts; if cuDNN picks a
    batch-dependent algorithm the test reports the max abs diff and requires
    it to be 0 (the depthwise conv sums 4 taps; any change would show here);
  * layouts under the bound take the untouched path (no grouping);
  * peak memory of the grouped path is printed next to the unsplit one.

Run:  /opt/venv/bin/python -m pytest -q -s sx_tests/validation/test_ple_prefill_grouping.py
"""

from __future__ import annotations

import types

import pytest
import torch

ple = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ple_layer")

HIDDEN = 10240
KERNEL = 4
DILATION = 3
STATE_LEN = (KERNEL - 1) * DILATION

LAYOUTS = [
    (3, [7056, 352, 432, 352]),  # the step that OOMed (C4 x 8K prompts)
    (0, [7840, 352]),
    (5, [352, 7056, 400, 384]),
    (2, [4000, 4000, 192]),
    (0, [784] * 10),  # 7840 padded rows: under the bound, untouched path
    (7, [1] + [300] * 5 + [6000]),
    (0, [5000]),  # single prefill: untouched
]


def _fake_layer():
    fake = types.SimpleNamespace(conv_state_len=STATE_LEN, short_conv_dilation=DILATION)
    fake._short_conv_dilated_prefill_packed = types.MethodType(
        ple.Qwen4ExpPLELayer._short_conv_dilated_prefill_packed, fake
    )
    return fake


def _case(decodes, lens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = len(lens)
    tokens = sum(lens)
    x_p = torch.randn(tokens, HIDDEN, generator=gen, device="cuda").half()
    slots = n + 8
    conv_state = torch.randn(slots, HIDDEN, STATE_LEN, generator=gen, device="cuda").half()
    weights = (torch.randn(HIDDEN, KERNEL, generator=gen, device="cuda") * 0.3).half()
    perm = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[:n]
    state_idx = perm.to(device="cuda", dtype=torch.int32)
    has_init = torch.rand(n, generator=gen, device="cuda") < 0.6
    starts = [0]
    for _ in range(decodes):
        starts.append(starts[-1] + 1)
    for length in lens:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int32, device="cuda")
    meta = types.SimpleNamespace(
        non_spec_query_start_loc=qsl,
        has_initial_states_p=has_init,
        max_prefill_query_len=max(lens),
        prefill_query_lens_cpu=tuple(lens),
    )
    return x_p, meta, conv_state, weights, state_idx


def _run(bound, decodes, lens, seed):
    x_p, meta, conv_state, weights, state_idx = _case(decodes, lens, seed)
    old_bound = ple._SX_PLE_PREFILL_MAX_PACKED_ROWS
    calls = []
    fake = _fake_layer()
    inner = fake._short_conv_dilated_prefill_packed

    def spy(*args):
        calls.append(args[6])  # num_prefills of this packed call
        return inner(*args)

    fake._short_conv_dilated_prefill_packed = spy
    ple._SX_PLE_PREFILL_MAX_PACKED_ROWS = bound
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = ple.Qwen4ExpPLELayer._short_conv_dilated_prefill_batched(
            fake, x_p, meta, conv_state, weights, state_idx,
            len(lens), decodes, sum(lens),
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
    finally:
        ple._SX_PLE_PREFILL_MAX_PACKED_ROWS = old_bound
    return out, conv_state, calls, peak


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("layout", range(len(LAYOUTS)))
def test_grouped_prefill_equals_unsplit(layout):
    decodes, lens = LAYOUTS[layout]
    out_old, state_old, calls_old, peak_old = _run(0, decodes, lens, seed=layout)
    out_new, state_new, calls_new, peak_new = _run(8192, decodes, lens, seed=layout)
    expect_split = len(lens) > 1 and len(lens) * max(lens) > 8192
    assert calls_old == [len(lens)]
    if expect_split:
        assert len(calls_new) > 1 and sum(calls_new) == len(lens), calls_new
        for first, last in ple._sx_ple_prefill_groups(lens, 8192):
            group = lens[first:last]
            assert len(group) == 1 or len(group) * max(group) <= 8192
    else:
        assert calls_new == [len(lens)], calls_new
    diff_out = (out_new.float() - out_old.float()).abs().max().item()
    diff_state = (state_new.float() - state_old.float()).abs().max().item()
    print(f"[sx-ple] decodes={decodes} lens={lens} groups={calls_new} "
          f"peak_MiB unsplit={peak_old / 2**20:.0f} grouped={peak_new / 2**20:.0f} "
          f"max_abs_diff out={diff_out:.3e} state={diff_state:.3e}", flush=True)
    assert torch.equal(_bits(out_new), _bits(out_old))
    assert torch.equal(_bits(state_new), _bits(state_old))


def test_group_planner():
    g = ple._sx_ple_prefill_groups
    assert g([7056, 352, 432, 352], 8192) == [(0, 1), (1, 4)]
    assert g([784] * 10, 8192) == [(0, 10)]
    assert g([8000, 9000], 8192) == [(0, 1), (1, 2)]
    assert g([5], 8192) == [(0, 1)]
