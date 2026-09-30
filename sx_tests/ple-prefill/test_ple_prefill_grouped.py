# SPDX-License-Identifier: Apache-2.0
"""PLE short-conv prefill: fewer live buffers, same bits, with and without groups.

Run (CPU is enough, no vLLM install needed; see ``ple_boot.py``):

    python sx_tests/ple-prefill/test_ple_prefill_grouped.py
    python -m pytest -q sx_tests/ple-prefill/test_ple_prefill_grouped.py

The oracle is the padded-copy implementation the prefill path had before, as
kept in ``tests/models/qwen4_exp/test_ple_short_conv_prefill.py``.

Asserted:
* the cases of that upstream test file pass against this tree (one pack per
  step, bitwise);
* with ``prefill_query_lens_cpu`` set and the packed-rows bound low enough to
  split the step into several groups, the output and the written conv state
  are bitwise equal to the oracle applied group by group, on random request
  mixes (zero-length rows, rows without a state slot, rows without an initial
  state, an empty cache). Against the oracle run as one pack the conv state
  is still bitwise equal and the output agrees to one float32 rounding step:
  a CPU convolution rounds the last bit differently for a different padded
  width, with the old packing code just the same;
* the grouping really happens in those cases, and a step under the bound or
  without the lengths still runs as one pack;
* on CUDA only: the peak extra memory of one pack stays within 2.25 packs.
Expected: all pass (the CUDA case is skipped on a CPU-only machine).
"""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ple_boot  # noqa: E402

ple = ple_boot.install()
up = ple_boot.upstream_test_module()
NULL_BLOCK_ID = sys.modules["vllm.v1.attention.backends.utils"].NULL_BLOCK_ID


@pytest.mark.parametrize("name", sorted(up.CASES))
def test_upstream_cases_pass(name: str) -> None:
    up.test_prefill_short_conv_matches_padded_copy_bitwise(name, "cpu", torch.float32)


def _oracle_by_group(module, x_p, metadata, conv_state, rest, lengths, groups):
    """The oracle, run on each group the way the prefill wrapper splits a step."""
    conv_weights, state_indices, _, _, _ = rest
    output = torch.empty_like(x_p)
    token_start = 0
    for first, last in groups:
        group_lens = lengths[first:last]
        tokens = sum(group_lens)
        if tokens > 0:
            starts = [0]
            for length in group_lens:
                starts.append(starts[-1] + length)
            group_meta = SimpleNamespace(
                non_spec_query_start_loc=torch.tensor(starts, dtype=torch.int32),
                has_initial_states_p=metadata.has_initial_states_p[first:last],
                max_prefill_query_len=max(group_lens),
            )
            output[token_start : token_start + tokens] = up._reference_prefill_batched(
                module, x_p[token_start : token_start + tokens], group_meta, conv_state,
                conv_weights, state_indices[first:last], last - first, 0, tokens,
            )
        token_start += tokens
    return output


def _random_case(seed: int):
    rng = random.Random(seed)
    num = rng.randint(2, 6)
    lengths = [rng.choice([0, 1, 2, 3, 5, 8, 13, 21, 34]) for _ in range(num)]
    if sum(lengths) == 0:
        lengths[0] = 7
    slots = 8
    picks = rng.sample(range(1, slots), num)          # distinct, non-null slots
    state_indices = [NULL_BLOCK_ID if rng.random() < 0.15 else p for p in picks]
    return dict(
        lengths=lengths,
        decode_tokens=rng.randint(0, 3),
        state_indices=state_indices,
        has_initial=[rng.random() < 0.6 for _ in range(num)],
        kernel_size=rng.choice([1, 2, 4]),
        dilation=rng.choice([1, 3]),
        hidden_size=rng.choice([4, 16]),
        state_slots=slots,
        empty_cache=rng.random() < 0.1,
        seed=seed,
    )


@pytest.mark.parametrize("seed", range(200))
def test_grouped_prefill_matches_oracle_bitwise(seed: int, monkeypatch) -> None:
    spec = _random_case(seed)
    module, x_p, args = up._case("cpu", torch.float32, **spec)
    metadata, conv_state, *rest = args
    lengths = spec["lengths"]
    metadata.prefill_query_lens_cpu = tuple(lengths)
    bound = random.Random(seed).choice([1, 8, 21, 40, 10_000])
    monkeypatch.setattr(ple, "_SX_PLE_PREFILL_MAX_PACKED_ROWS", bound)

    packs = []
    packed = module._short_conv_dilated_prefill_packed

    def counting(*a, **k):
        packs.append(a[6])                      # num_prefills of this pack
        return packed(*a, **k)

    monkeypatch.setattr(module, "_short_conv_dilated_prefill_packed", counting)

    groups = ple._sx_ple_prefill_groups(lengths, bound)
    split = len(lengths) * max(lengths) > bound and len(groups) > 1
    one_pack_state = conv_state.clone()
    one_pack = up._reference_prefill_batched(module, x_p, metadata, one_pack_state, *rest)
    grouped_state = conv_state.clone()
    expected = (
        _oracle_by_group(module, x_p, metadata, grouped_state, rest, lengths, groups)
        if split
        else up._reference_prefill_batched(module, x_p, metadata, grouped_state, *rest)
    )
    actual = module._short_conv_dilated_prefill_batched(x_p, metadata, conv_state, *rest)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert torch.equal(conv_state, grouped_state)
    assert torch.equal(conv_state, one_pack_state)
    assert torch.allclose(actual, one_pack, rtol=1e-6, atol=1e-7)

    if split:
        # Zero-token groups are skipped, every other group is one pack.
        expect = [last - first for first, last in groups if sum(lengths[first:last]) > 0]
        assert packs == expect
    else:
        assert packs == [len(lengths)]


def test_one_pack_without_lengths_or_under_the_bound(monkeypatch) -> None:
    spec = dict(lengths=[9, 4, 6], decode_tokens=1, state_indices=[3, 1, 2],
                has_initial=[True, False, True])
    for lens, bound in ((None, 1), ((9, 4, 6), 27), ((9, 4, 6), 0)):
        module, x_p, args = up._case("cpu", torch.float32, **spec)
        metadata, conv_state, *rest = args
        metadata.prefill_query_lens_cpu = lens
        monkeypatch.setattr(ple, "_SX_PLE_PREFILL_MAX_PACKED_ROWS", bound)
        packs = []
        packed = module._short_conv_dilated_prefill_packed
        monkeypatch.setattr(
            module, "_short_conv_dilated_prefill_packed",
            lambda *a, _p=packed, **k: (packs.append(a[6]), _p(*a, **k))[1],
        )
        reference_state = conv_state.clone()
        expected = up._reference_prefill_batched(
            module, x_p, metadata, reference_state, *rest
        )
        actual = module._short_conv_dilated_prefill_batched(
            x_p, metadata, conv_state, *rest
        )
        assert packs == [3]
        assert torch.equal(actual, expected)
        assert torch.equal(conv_state, reference_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_one_pack_holds_two_pack_sized_buffers() -> None:
    lengths = [2048]
    module, x_p, args = up._case(
        "cuda", torch.float16, lengths=lengths, decode_tokens=0, state_indices=[1],
        has_initial=[True], hidden_size=1024,
    )
    metadata, conv_state, *rest = args
    pack_bytes = x_p.numel() * x_p.element_size()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = module._short_conv_dilated_prefill_batched(x_p, metadata, conv_state, *rest)
    torch.cuda.synchronize()
    grown = torch.cuda.max_memory_allocated() - base
    del out
    assert grown <= 2.25 * pack_bytes, (grown, pack_bytes)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
