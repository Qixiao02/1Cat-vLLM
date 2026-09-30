# SPDX-License-Identifier: Apache-2.0
"""``_mamba_block_aligned_split``: off means upstream, on stops where needed.

Run (CPU only, no vLLM install needed; see ``align_boot.py``):

    python sx_tests/align-multiblock/test_split_equivalence.py
    python -m pytest -q sx_tests/align-multiblock/test_split_equivalence.py

The real ``Scheduler._mamba_block_aligned_split`` of the working tree is
called unbound on small stand-in objects, the way upstream's
``tests/v1/core/test_mamba_align_chunk_split.py`` does.

Asserted:
* ``UPSTREAM_SPLIT`` below is the d30469863 function (compared with the git
  blob when the base commit is readable);
* with the switch off, and on an object without any ``_sx_*`` attribute, the
  split equals upstream on 200,000 random cases (resumed requests, Eagle
  tail rule, shared-prefix boundaries, sub-block budgets);
* with the switch on, a chunk that starts mid-block equals upstream
  (200,000 random cases, all starting mid-block);
* with the switch on, a chunk that starts on a boundary is never shorter
  than upstream's, ends on a boundary unless it reaches the end of the
  request or the budget is below one block, and never crosses a boundary
  that the real ``MambaManager.reachable_block_mask`` retains
  (200,000 random cases; retention 0 and positive, Eagle and non-Eagle);
* chunk plans of a lone request for the production geometries;
* ``SX_OPT_ALIGN_MAX_CHUNK_BLOCKS`` caps a chunk.
Expected: all pass.
"""

from __future__ import annotations

import ast
import os
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402
from align_harness import chunk_ends, split_shim  # noqa: E402

BUDGET = 8192
NUM_RANDOM_CASES = 200_000


# Body copied verbatim from d30469863 vllm/v1/core/sched/scheduler.py:414-478.
def UPSTREAM_SPLIT(  # noqa: N802
    self,
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    start = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    # `request.num_tokens - 1` extends prefill handling to resumed requests
    # replaying output tokens while leaving ordinary decode untouched.
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens

    block_size = (
        self.mamba_state_block_size
        if self.mamba_state_block_size is not None
        else self.cache_config.block_size
    )
    # Prefix lookup is capped at n - 1. Flooring from n would register an
    # unreachable state whenever the request length is block aligned.
    last_token = request.num_tokens - 1
    last_cache_position = last_token - last_token % block_size
    if self.use_eagle:
        last_cache_position = max(last_cache_position - block_size, 0)

    end = start + num_new_tokens
    if end < prefill_end:
        aligned_end = end // block_size * block_size
        # Only take the aligned end when it advances past `start`. Otherwise
        # this returns 0, the caller treats that as "cannot schedule", and
        # the request is skipped on every step while it holds its KV blocks
        # and encoder-cache entries. The condition recurs whenever the chunk
        # available this step is shorter than one state block -- e.g. when
        # the encoder-cache gate caps it just before an image placeholder --
        # so the request starves indefinitely. Scheduling the shorter,
        # unaligned chunk only skips this block's Mamba state checkpoint, and
        # it cannot straddle two state blocks because `aligned_end <= start`
        # implies `end < next_block_boundary`.
        if aligned_end > start:
            end = aligned_end

    # The align allocator materializes one recurrent-state column per
    # scheduler step. A step spanning multiple state blocks leaves the
    # interior slots null, so every crossed boundary must end a chunk.
    next_block_boundary = (start // block_size + 1) * block_size
    end = min(
        (
            stop
            for stop in (
                next_block_boundary,
                last_cache_position,
                getattr(request, "shared_prefix_boundary", 0)
                // block_size
                * block_size,
            )
            if start < stop < end
        ),
        default=end,
    )
    return max(end - start, 0)


@pytest.fixture(scope="module")
def universe() -> align_boot.Universe:
    return align_boot.load()


@pytest.fixture(scope="module")
def split(universe):
    return universe.scheduler.Scheduler._mamba_block_aligned_split


def _random_request(rng: random.Random, block: int, *, aligned: bool | None):
    """A request somewhere in its prefill and the tokens offered to it."""
    prompt = rng.choice(
        [
            rng.randrange(1, 5 * block),
            rng.randrange(1, 60 * block),
            rng.randrange(1, 60) * block + rng.choice([-1, 0, 1]),
        ]
    )
    outputs = rng.choice([0, 0, 0, rng.randrange(1, 3 * block)])
    num_tokens = prompt + outputs
    start = rng.randrange(0, num_tokens)
    if aligned is True or (aligned is None and rng.random() < 0.5):
        start = start // block * block
    elif aligned is False and start % block == 0:
        start += rng.randrange(1, block)
        if start >= num_tokens:
            return None
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_prompt_tokens=prompt,
        num_tokens=num_tokens,
    )
    if rng.random() < 0.3:
        request.shared_prefix_boundary = rng.randrange(0, num_tokens + 1)
    remaining = num_tokens - start
    offered = rng.choice(
        [
            min(remaining, BUDGET),
            min(remaining, rng.randrange(1, BUDGET + 1)),
            min(remaining, rng.randrange(1, block + 1)),
        ]
    )
    return request, offered


def test_vendored_upstream_split_is_the_base_commit() -> None:
    if not align_boot.have_base():
        pytest.skip(f"base commit {align_boot.BASE_REV} is not readable")

    def body(source: str, name: str) -> str:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.dump(ast.Module(body=node.body, type_ignores=[]))
        raise AssertionError(name)

    rel_path = align_boot.REAL_MODULES[align_boot.SCHEDULER]
    upstream = body(
        align_boot.read_source(rel_path, align_boot.BASE_REV),
        "_mamba_block_aligned_split",
    )
    with open(__file__, encoding="utf-8") as f:
        vendored = body(f.read(), "UPSTREAM_SPLIT")
    assert vendored == upstream


def test_switch_off_equals_upstream(universe, split) -> None:
    references = [UPSTREAM_SPLIT]
    if align_boot.have_base():
        base = align_boot.load(upstream=align_boot.PATCHED_MODULES)
        references.append(base.scheduler.Scheduler._mamba_block_aligned_split)
    rng = random.Random(0)
    checked = 0
    while checked < NUM_RANDOM_CASES:
        block = rng.choice([784, 816, 3296, 4096, 16])
        eagle = rng.random() < 0.3
        case = _random_request(rng, block, aligned=None)
        if case is None:
            continue
        request, offered = case
        # Upstream's own unit test passes an object without `_sx_*` attributes.
        bare = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16),
            mamba_state_block_size=rng.choice([block, block, None]),
            use_eagle=eagle,
        )
        if bare.mamba_state_block_size is None:
            bare.cache_config.block_size = block
        off = split_shim(universe, block, eagle=eagle, multiblock=False)
        want = UPSTREAM_SPLIT(bare, request, offered)
        for reference in references:
            assert reference(bare, request, offered) == want
        assert split(bare, request, offered) == want, (block, eagle, vars(request))
        assert split(off, request, offered) == want, (block, eagle, vars(request))
        checked += 1


def test_mid_block_start_equals_upstream(universe, split) -> None:
    rng = random.Random(1)
    checked = 0
    while checked < NUM_RANDOM_CASES:
        block = rng.choice([784, 816, 3296, 4096])
        eagle = rng.random() < 0.3
        case = _random_request(rng, block, aligned=False)
        if case is None:
            continue
        request, offered = case
        shim = split_shim(
            universe,
            block,
            eagle=eagle,
            retention=rng.choice([0, 0, 4 * block, 10 * block]),
            max_blocks=rng.choice([0, 0, 3]),
        )
        assert request.num_computed_tokens % block
        assert split(shim, request, offered) == UPSTREAM_SPLIT(shim, request, offered)
        checked += 1


def _crossed_retained_blocks(universe, shim, request, end: int) -> int:
    """Blocks inside the chunk [computed, end) that upstream would cache.

    Asks the real ``MambaManager.reachable_block_mask`` with the inputs
    ``MambaManager.cache_blocks`` gives it. Only the block holding the state
    at the end of the chunk is a real block, so any other retained block is a
    state that per-block chunking caches and this chunk loses.
    """
    import torch

    block = shim.mamba_state_block_size
    coordinator = shim.kv_cache_manager.coordinator
    boundaries = list(coordinator.get_replay_boundaries(request, block))
    if getattr(request, "shared_prefix_boundary", 0):
        boundaries.append(request.shared_prefix_boundary)
    spec = universe.kv_cache_interface.MambaSpec(
        block_size=block,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    mask = universe.allocator.MambaManager.reachable_block_mask(
        request.num_computed_tokens // block,
        end // block,
        block,
        spec,
        shim._sx_align_retention,
        boundaries,
    )
    assert mask is not None
    return sum(mask[:-1])


def _aligned_cases(universe, seed: int, count: int):
    rng = random.Random(seed)
    produced = 0
    while produced < count:
        block = rng.choice([784, 816, 3296, 4096])
        eagle = rng.random() < 0.4
        retention = rng.choice([0, 0, 2 * block, 4 * block, 10 * block])
        case = _random_request(rng, block, aligned=True)
        if case is None:
            continue
        request, offered = case
        start = request.num_computed_tokens
        if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
            continue
        produced += 1
        yield split_shim(universe, block, eagle=eagle, retention=retention), request, (
            offered
        )


def test_aligned_start_stops_at_every_retained_boundary(universe, split) -> None:
    """No chunk crosses a state that ``MambaManager.cache_blocks`` would cache."""
    multiblock = 0
    for shim, request, offered in _aligned_cases(universe, 2, NUM_RANDOM_CASES):
        block = shim.mamba_state_block_size
        start = request.num_computed_tokens
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        got = split(shim, request, offered)
        assert UPSTREAM_SPLIT(shim, request, offered) <= got <= offered
        end = start + got
        # The chunk ends on the grid, at the end of what was offered when that
        # reaches the end of the prefill, or inside the first block when less
        # than one block was offered.
        assert end % block == 0 or end == start + offered
        if end % block:
            assert start + offered >= prefill_end or offered < block
        assert _crossed_retained_blocks(universe, shim, request, end) == 0, (
            vars(shim),
            vars(request),
            offered,
        )
        multiblock += (end - start) > block
    assert multiblock > NUM_RANDOM_CASES // 10


def test_the_retained_boundary_check_sees_a_missing_stop(universe) -> None:
    """Chunks that simply run to the budget do cross retained states."""
    crossed = 0
    for shim, request, offered in _aligned_cases(universe, 3, 20_000):
        block = shim.mamba_state_block_size
        start = request.num_computed_tokens
        end = start + offered
        if end < max(request.num_prompt_tokens, request.num_tokens - 1):
            end = max(end // block * block, start + 1)
        crossed += _crossed_retained_blocks(universe, shim, request, end) > 0
    assert crossed > 1_000


# (state block, Eagle tail rule) -> prompt length -> chunk ends, budget 8192.
PLANS = {
    (784, False): {
        450: [450],
        784: [784],
        785: [784, 785],
        3000: [2352, 3000],
        8192: [7840, 8192],
        32768: [7840, 15680, 23520, 31360, 32144, 32768],
        65536: [7840, 15680, 23520, 31360, 39200, 47040, 54880, 62720, 65072, 65536],
    },
    (816, True): {
        450: [450],
        816: [816],
        2048: [816, 2048],
        # Block aligned: both replay boundaries, 1632 and 2448, end a chunk.
        3264: [1632, 2448, 3264],
        8192: [7344, 8192],
        32768: [8160, 16320, 24480, 31824, 32768],
    },
}
# (prompt, steps per-block, steps multi-block, resend hit): the design's table.
FLASH_NEXT_STEPS = [
    (3_000, 4, 2, 2_352),
    (8_192, 11, 2, 7_840),
    (32_768, 42, 6, 32_144),
    (65_536, 84, 10, 65_072),
    (110_000, 141, 15, 109_760),
]


@pytest.mark.parametrize("block,eagle", sorted(PLANS))
def test_chunk_plans(universe, split, block: int, eagle: bool) -> None:
    on = split_shim(universe, block, eagle=eagle)
    off = split_shim(universe, block, eagle=eagle, multiblock=False)
    for prompt_len, expected in PLANS[(block, eagle)].items():
        assert chunk_ends(split, on, prompt_len) == expected, prompt_len
        per_block = chunk_ends(split, off, prompt_len)
        assert per_block == chunk_ends(UPSTREAM_SPLIT, off, prompt_len)
        # Multi-block only ever drops chunk ends.
        assert set(expected) <= set(per_block), prompt_len


@pytest.mark.parametrize("prompt_len,per_block,multi_block,hit", FLASH_NEXT_STEPS)
def test_flash_next_step_counts(universe, split, prompt_len, per_block,
                                multi_block, hit) -> None:
    on = split_shim(universe, 784)
    off = split_shim(universe, 784, multiblock=False)
    ends = chunk_ends(split, on, prompt_len)
    assert len(chunk_ends(split, off, prompt_len)) == per_block
    assert len(ends) == multi_block
    # The resend checkpoint is a chunk end in both plans.
    assert hit == (prompt_len - 1) // 784 * 784
    assert hit in ends


@pytest.mark.parametrize(
    "block,budget,threshold,prompt_len,expected",
    [
        # 27B, grid 4096, threshold 8192 under a 16384 budget: 8192 chunks.
        (4096, 16384, 8192, 110_000, 14),
        # Same grid with the budget as the only limit: chunks leave the window.
        (4096, 16384, 0, 110_000, 8),
        # Today's 3296 grid with threshold 8000: 6592 chunks.
        (3296, 16384, 8000, 110_000, 18),
    ],
)
def test_27b_step_counts(universe, split, block, budget, threshold, prompt_len,
                         expected) -> None:
    on = split_shim(universe, block)
    ends = chunk_ends(split, on, prompt_len, budget=budget, threshold=threshold)
    assert len(ends) == expected
    assert (prompt_len - 1) // block * block in ends


def test_retention_interval_and_cap(universe, split) -> None:
    periodic = split_shim(universe, 784, retention=4 * 784)
    assert chunk_ends(split, periodic, 8192) == [3136, 6272, 7840, 8192]
    # A chunk starting between two retained boundaries stops at the next one.
    assert chunk_ends(split, periodic, 8192, start=784) == [3136, 6272, 7840, 8192]
    capped = split_shim(universe, 784, max_blocks=3)
    assert chunk_ends(split, capped, 8192) == [2352, 4704, 7056, 7840, 8192]
    # The 7840 interval coincides with a full-budget chunk: no extra step.
    fork_like = split_shim(universe, 784, retention=7840)
    plain = split_shim(universe, 784)
    for prompt_len in (8192, 32768, 110_000):
        assert chunk_ends(split, fork_like, prompt_len) == chunk_ends(
            split, plain, prompt_len
        )


def test_shared_prefix_boundary_ends_a_chunk(universe, split) -> None:
    on = split_shim(universe, 784)
    ends = chunk_ends(split, on, 8192, shared_prefix_boundary=2000)
    assert ends == [1568, 7840, 8192]


def test_sub_block_budget_then_per_block(universe, split) -> None:
    """Less than one block of budget: no boundary is crossed from mid-block."""
    on = split_shim(universe, 784)
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=8192,
                              num_tokens=8192)
    assert split(on, request, 352) == 352
    request.num_computed_tokens = 352
    # Mid-block: stop at the next boundary whatever the budget.
    assert split(on, request, 7840) == 784 - 352
    request.num_computed_tokens = 784
    assert split(on, request, 7408) == 7056


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
