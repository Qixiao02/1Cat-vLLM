# SPDX-License-Identifier: Apache-2.0
"""Bounded packing of the PLE short-conv prefill (CPU).

Run (CPU only, torch is the only dependency):

    python sx_tests/align-multiblock/test_ple_prefill_groups.py
    python -m pytest -q sx_tests/align-multiblock/test_ple_prefill_groups.py

A multi-block chunk puts one request of several thousand tokens next to the
other prefills of a step. The dilated PLE short-conv packs all prefill
requests of a step into a zero-padded ``[num_prefills, max_len, hidden]``
buffer, which then grows to ``num_prefills`` times the long chunk.
``SX_OPT_PLE_PREFILL_MAX_PACKED_ROWS`` (default 8192) packs consecutive
requests in groups whose padded rows stay within the bound.

``ple_layer.py`` cannot be imported without the serving stack, so the group
planner and the two prefill methods are cut out of the source file with
``ast`` and run on a stand-in layer. They are plain torch code.

Asserted:
* the planner: consecutive groups covering every request, each within the
  bound unless it is a single request, and greedy (a group could not have
  taken the next request);
* grouped packing equals the unsplit packing on the step layouts that
  overflowed in production (scaled hidden size) and on random ones: the conv
  state written back bit for bit, the output to 1e-6 (the CPU convolution
  and SiLU round the last bit differently for differently shaped batches);
* a step under the bound, a single prefill, a bound of 0 and metadata
  without usable lengths all take the single unsplit call;
* the switch: default 8192, an unparsable value falls back to it.
Expected: all pass. The half-precision GPU comparison on the real layer is
``sx_tests/validation/test_ple_prefill_grouping.py`` of the SX fork.
"""

from __future__ import annotations

import ast
import os
import random
import sys
import types
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402

PLE_LAYER = "vllm/models/qwen4_exp/nvidia/ple_layer.py"
BOUND = "_SX_PLE_PREFILL_MAX_PACKED_ROWS"
PLANNER = "_sx_ple_prefill_groups"
HIDDEN = 24
KERNEL = 4
DILATION = 3
STATE_LEN = (KERNEL - 1) * DILATION
# Outputs are O(1); a request convolved with the wrong state or rows is off
# by about that much.
OUTPUT_TOLERANCE = 1e-6


def _assert_same(out_new, state_new, out_old, state_old) -> None:
    assert torch.equal(state_new, state_old)
    assert out_new.shape == out_old.shape
    assert (out_new - out_old).abs().max().item() <= OUTPUT_TOLERANCE


def _load(env: dict[str, str] | None = None) -> types.SimpleNamespace:
    """The switch, the planner and the two prefill methods of the real file."""
    tree = ast.parse(align_boot.read_source(PLE_LAYER))
    picked: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Try) and BOUND in ast.unparse(node):
            picked.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == PLANNER:
            picked.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpPLELayer":
            picked.extend(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and item.name
                in (
                    "_short_conv_dilated_prefill_batched",
                    "_short_conv_dilated_prefill_packed",
                )
            )
    assert len(picked) == 4, [getattr(node, "name", "switch") for node in picked]
    namespace = dict(
        os=types.SimpleNamespace(environ=dict(env or {})),
        torch=torch,
        F=F,
        Sequence=Sequence,
        NULL_BLOCK_ID=0,
        PleShortConvAttentionMetadata=object,
    )
    exec(  # noqa: S102 - source tree code, as `import` would run it
        compile(ast.Module(body=picked, type_ignores=[]), PLE_LAYER, "exec"), namespace
    )
    return types.SimpleNamespace(**namespace)


@pytest.fixture(scope="module")
def ple() -> types.SimpleNamespace:
    return _load()


def _case(decodes: int, lens: list[int], seed: int, *, lens_cpu="exact"):
    gen = torch.Generator().manual_seed(seed)
    n = len(lens)
    x_p = torch.randn(sum(lens), HIDDEN, generator=gen)
    slots = n + 8
    conv_state = torch.randn(slots, HIDDEN, STATE_LEN, generator=gen)
    weights = torch.randn(HIDDEN, KERNEL, generator=gen) * 0.3
    # Slot 0 is the null block; a request pointing at it has no state.
    state_idx = torch.randperm(slots, generator=gen)[:n].to(torch.int32)
    has_init = torch.rand(n, generator=gen) < 0.6
    starts = list(range(decodes + 1))
    for length in lens:
        starts.append(starts[-1] + length)
    meta = types.SimpleNamespace(
        non_spec_query_start_loc=torch.tensor(starts, dtype=torch.int32),
        has_initial_states_p=has_init,
        max_prefill_query_len=max(lens),
    )
    if lens_cpu == "exact":
        meta.prefill_query_lens_cpu = tuple(lens)
    elif lens_cpu != "missing":
        meta.prefill_query_lens_cpu = lens_cpu
    return x_p, meta, conv_state, weights, state_idx


def _stand_in_layer() -> types.SimpleNamespace:
    return types.SimpleNamespace(conv_state_len=STATE_LEN, short_conv_dilation=DILATION)


def _run(ple, bound: int, decodes: int, lens: list[int], seed: int, **case):
    x_p, meta, conv_state, weights, state_idx = _case(decodes, lens, seed, **case)
    calls: list[int] = []
    layer = _stand_in_layer()

    def packed(*args):
        calls.append(args[6])  # num_prefills of this packed call
        return ple._short_conv_dilated_prefill_packed(layer, *args)

    layer._short_conv_dilated_prefill_packed = packed
    batched = ple._short_conv_dilated_prefill_batched
    saved = batched.__globals__[BOUND]
    batched.__globals__[BOUND] = bound
    try:
        out = batched(
            layer, x_p, meta, conv_state, weights, state_idx, len(lens), decodes,
            sum(lens),
        )
    finally:
        batched.__globals__[BOUND] = saved
    return out, conv_state, calls


def _assert_plan(ple, lens: list[int], max_rows: int) -> list[tuple[int, int]]:
    groups = ple._sx_ple_prefill_groups(lens, max_rows)
    assert groups[0][0] == 0 and groups[-1][1] == len(lens)
    for (first, last), following in zip(groups, groups[1:] + [None]):
        assert first < last
        group = lens[first:last]
        assert len(group) == 1 or len(group) * max(group) <= max_rows
        if following is not None:
            assert following[0] == last
            # Greedy: the group could not have taken the next request.
            wider = lens[first : last + 1]
            assert len(wider) * max(wider) > max_rows
    return groups


# (decodes, prefill lengths, bound): step layouts from the SX fork's GPU test,
# and the same shapes at a bound small enough for random CPU cases.
LAYOUTS = [
    (3, [7056, 352, 432, 352], 8192),  # four 8K prompts: the step that overflowed
    (0, [7840, 352], 8192),
    (5, [352, 7056, 400, 384], 8192),
    (2, [4000, 4000, 192], 8192),
    (7, [1] + [300] * 5 + [6000], 8192),
    (0, [784] * 10, 8192),  # 7840 padded rows: under the bound
    (0, [5000], 8192),  # a single prefill
    (1, [90, 3, 40, 40, 41, 7, 120], 128),
]


def test_group_planner(ple) -> None:
    plan = ple._sx_ple_prefill_groups
    assert plan([7056, 352, 432, 352], 8192) == [(0, 1), (1, 4)]
    assert plan([784] * 10, 8192) == [(0, 10)]
    assert plan([8000, 9000], 8192) == [(0, 1), (1, 2)]
    assert plan([5], 8192) == [(0, 1)]
    assert plan([9000], 8192) == [(0, 1)]
    rng = random.Random(0)
    for _ in range(3000):
        lens = [
            rng.choice([1, rng.randrange(1, 60), rng.randrange(1, 9000)])
            for _ in range(rng.randrange(1, 12))
        ]
        _assert_plan(ple, lens, rng.choice([1, 64, 1000, 8192, 1 << 30]))


@pytest.mark.parametrize("layout", range(len(LAYOUTS)))
def test_grouped_prefill_equals_unsplit(ple, layout: int) -> None:
    decodes, lens, bound = LAYOUTS[layout]
    out_old, state_old, calls_old = _run(ple, 0, decodes, lens, seed=layout)
    out_new, state_new, calls_new = _run(ple, bound, decodes, lens, seed=layout)
    assert calls_old == [len(lens)]
    groups = _assert_plan(ple, lens, bound)
    if len(lens) > 1 and len(lens) * max(lens) > bound:
        assert len(groups) > 1
        assert calls_new == [last - first for first, last in groups]
    else:
        assert calls_new == [len(lens)]
    _assert_same(out_new, state_new, out_old, state_old)


def test_random_layouts_equal_unsplit(ple) -> None:
    rng = random.Random(1)
    split = 0
    for seed in range(60):
        lens = [
            rng.choice([1, rng.randrange(1, 40), rng.randrange(1, 400)])
            for _ in range(rng.randrange(1, 9))
        ]
        decodes = rng.randrange(0, 6)
        bound = rng.choice([64, 256, 512])
        out_old, state_old, _ = _run(ple, 0, decodes, lens, seed=seed)
        out_new, state_new, calls = _run(ple, bound, decodes, lens, seed=seed)
        split += len(calls) > 1
        _assert_same(out_new, state_new, out_old, state_old)
    assert split > 20


@pytest.mark.parametrize(
    "lens_cpu",
    ["missing", None, (7056, 352, 432), (7056, 352, 432, 351)],
    ids=["no-attribute", "none", "wrong-count", "wrong-sum"],
)
def test_unusable_lengths_keep_the_unsplit_packing(ple, lens_cpu) -> None:
    lens = [7056, 352, 432, 352]
    out_old, state_old, _ = _run(ple, 0, 3, lens, seed=9)
    out_new, state_new, calls = _run(ple, 8192, 3, lens, seed=9, lens_cpu=lens_cpu)
    assert calls == [4]
    assert torch.equal(out_new, out_old) and torch.equal(state_new, state_old)


def test_the_comparison_sees_a_wrong_state(ple) -> None:
    """Swapping two requests' state slots is far outside the tolerance."""
    lens = [7056, 352, 432, 352]
    out_old, state_old, _ = _run(ple, 0, 3, lens, seed=9)
    x_p, meta, conv_state, weights, state_idx = _case(3, lens, 9)
    meta.has_initial_states_p[:] = True
    layer = _stand_in_layer()
    layer._short_conv_dilated_prefill_packed = types.MethodType(
        ple._short_conv_dilated_prefill_packed, layer
    )
    out_new = ple._short_conv_dilated_prefill_batched(
        layer, x_p, meta, conv_state, weights, state_idx.flip(0), 4, 3, sum(lens)
    )
    assert (out_new - out_old).abs().max().item() > 1000 * OUTPUT_TOLERANCE


@pytest.mark.parametrize(
    "env,expected",
    [({}, 8192), ({"SX_OPT_PLE_PREFILL_MAX_PACKED_ROWS": "0"}, 0),
     ({"SX_OPT_PLE_PREFILL_MAX_PACKED_ROWS": "4096"}, 4096),
     ({"SX_OPT_PLE_PREFILL_MAX_PACKED_ROWS": "many"}, 8192)],
)
def test_switch(env, expected) -> None:
    assert getattr(_load(env), BOUND) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
