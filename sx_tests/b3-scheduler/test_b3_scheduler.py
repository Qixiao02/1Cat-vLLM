# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the batch-3a scheduler group (no GPU needed).

* SX_OPT_ALIGN_MULTIBLOCK_SPEC (design_1 [MTP-5] / design_2 [PF1] /
  design_3 [C7]): align multi-block prefill chunks in the SM70 Qwen3.8
  native-MTP lane, with the Eagle/MTP tail rule and an Eagle-aware
  shared-prefix checkpoint.
* SX_OPT_ALIGN_TAIL_MIN_TOKENS (design_2 [PF4], default 3136): no extra step
  only to take the tail checkpoint of a short remaining prefill.
* SX_OPT_PREFILL_CAP_WITH_DECODES (design_2 [PF3], default off): decode-aware
  prefill chunk cap.

Run inside the deployed image (1.8.0-dev2) with the overlay bind-mounted, from
a directory where ``import vllm`` resolves to the installed package:

    cd /tmp && /opt/venv/bin/python -m pytest -q \
        <repo>/sx_tests/b3-scheduler/test_b3_scheduler.py

What is asserted:
* chunk plans (budget 8192) of the real ``_mamba_block_aligned_split`` for
  prompt lengths {450, 784, 785, 1000, 1568, 2048, 3136, 8192, 32768} in the
  no-MTP lane (block 784) and the MTP lane (Eagle tail rule; blocks 784 and
  816), with and without PF4, after prefix hits, and under the PF3 cap;
* with SX_OPT_ALIGN_TAIL_MIN_TOKENS=0 and the PF3 cap off, the split equals a
  verbatim copy of the 1.8.0-dev2 function (300k randomized cases); with PF4
  it differs only where PF4 applies;
* the same plans from the REAL ``Scheduler.schedule()`` (sync and async
  scheduler, native-MTP k=2/4 verify steps with rejection), a recurrent-state
  oracle mirroring the MRV2 align pre-copy / forward / post-process with the
  1 + k speculative state columns, and identical-resend bookkeeping: the
  resend hits the deepest reachable checkpoint and computes exactly the
  remaining chunks of the first run (prefix-cache miss == hit);
* shared system prompts with short distinct suffixes (the shared checkpoint
  is also each request's tail position): with PF4 the second sharer takes
  the shared-prefix stop and the third one on hits it, in both lanes and
  both attention page layouts (review fix; the PF4 resend rule used to drop
  every such stop, so no sharer ever hit); the PF4 resend memo keeps
  identical resends on the first run's chunks and is bounded (LRU);
* the PF3 per-step token budget is raised to one state block (a smaller
  budget held long prompts, and FCFS everything behind them, while any
  decode ran);
* randomized mixed workloads with shared prefixes, identical resends,
  natural preemption (small KV pools), async scheduling, k in {1, 2, 3, 4},
  PF3/PF4 on and off: no oracle violation, no allocator assert, outputs
  equal the reference;
* policy gating in ``Scheduler.__init__`` (real init) for the MTP contract.
Expected result: all tests pass.
"""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sched_harness import (  # noqa: E402
    OracleError,
    SchedHarness,
    make_prompt,
)

BUDGET = 8192
LENGTHS = [450, 784, 785, 1000, 1568, 2048, 3136, 8192, 32768]
# lane -> (state block, Eagle/MTP tail rule)
LANES = {"nomtp": (784, False), "mtp784": (784, True), "mtp816": (816, True)}


# --------------------------------------------------------------------------
# Verbatim 1.8.0-dev2 split (git b53c180) for equivalence checks
# --------------------------------------------------------------------------
def _dev2_mamba_block_aligned_split(
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
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens

    block_size = (
        self.mamba_state_block_size
        if self.mamba_state_block_size is not None
        else self.cache_config.block_size
    )
    last_token = request.num_tokens - 1
    last_cache_position = last_token - last_token % block_size
    if self.use_eagle:
        last_cache_position = max(last_cache_position - block_size, 0)

    end = start + num_new_tokens
    if end < prefill_end:
        aligned_end = end // block_size * block_size
        if aligned_end > start:
            end = aligned_end

    if getattr(self, "_sx_align_multiblock", False) and start % block_size == 0:
        stops = []
        if getattr(self, "_sx_align_tail", True):
            stops.append(last_cache_position)
        max_blocks = getattr(self, "_sx_align_max_blocks", 0)
        if max_blocks > 0:
            stops.append(start + max_blocks * block_size)
        shared_stop = getattr(request, "_sx_align_shared_stop", 0)
        if shared_stop:
            stops.append(shared_stop)
        end = min((stop for stop in stops if start < stop < end), default=end)
        return max(end - start, 0)

    next_block_boundary = (start // block_size + 1) * block_size
    end = min(
        (
            stop
            for stop in (next_block_boundary, last_cache_position)
            if start < stop < end
        ),
        default=end,
    )
    return max(end - start, 0)


def _split_fn():
    from vllm.v1.core.sched.scheduler import Scheduler

    return Scheduler._mamba_block_aligned_split


def _shim(block: int, eagle: bool, tail_min: int, *, multiblock: bool = True,
          tail: bool = True, max_blocks: int = 0, step_blocks=None,
          prefill_left=None) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        mamba_state_block_size=block,
        use_eagle=eagle,
        _sx_align_multiblock=multiblock,
        _sx_align_tail=tail,
        _sx_align_max_blocks=max_blocks,
        _sx_align_tail_min=tail_min,
        _sx_step_max_blocks=step_blocks if step_blocks is not None else max_blocks,
        _sx_step_prefill_left=prefill_left,
    )


def _chunk_ends(split, shim, prompt_len: int, start: int = 0,
                budget: int = BUDGET) -> list[int]:
    request = SimpleNamespace(
        num_computed_tokens=start, num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
    )
    ends = []
    while request.num_computed_tokens < prompt_len:
        n = min(prompt_len - request.num_computed_tokens, budget)
        n = split(shim, request, n)
        assert n > 0
        request.num_computed_tokens += n
        ends.append(request.num_computed_tokens)
    return ends


# (lane, tail_min) -> prompt length -> chunk ends; idle engine, budget 8192.
# tail_min 0 is the 1.8.0-dev2 multi-block policy (plus the Eagle tail rule
# for the MTP lanes); 3136 is the PF4 default.
PLANS = {
    ("nomtp", 0): {
        450: [450], 784: [784], 785: [784, 785], 1000: [784, 1000],
        1568: [784, 1568], 2048: [1568, 2048], 3136: [2352, 3136],
        8192: [7840, 8192],
        32768: [7840, 15680, 23520, 31360, 32144, 32768],
    },
    ("nomtp", 3136): {
        450: [450], 784: [784], 785: [785], 1000: [1000], 1568: [1568],
        2048: [2048], 3136: [3136], 8192: [7840, 8192],
        32768: [7840, 15680, 23520, 31360, 32768],
    },
    ("mtp784", 0): {
        450: [450], 784: [784], 785: [785], 1000: [1000], 1568: [1568],
        2048: [784, 2048], 3136: [1568, 3136], 8192: [7056, 8192],
        32768: [7840, 15680, 23520, 31360, 32768],
    },
    ("mtp784", 3136): {
        450: [450], 784: [784], 785: [785], 1000: [1000], 1568: [1568],
        2048: [2048], 3136: [3136], 8192: [7056, 8192],
        32768: [7840, 15680, 23520, 31360, 32768],
    },
    ("mtp816", 0): {
        450: [450], 784: [784], 785: [785], 1000: [1000], 1568: [1568],
        2048: [816, 2048], 3136: [1632, 3136], 8192: [7344, 8192],
        32768: [8160, 16320, 24480, 31824, 32768],
    },
    ("mtp816", 3136): {
        450: [450], 784: [784], 785: [785], 1000: [1000], 1568: [1568],
        2048: [2048], 3136: [3136], 8192: [7344, 8192],
        32768: [8160, 16320, 24480, 31824, 32768],
    },
}

# Per-block plans (multi-block off): what the MTP lane ran before this batch.
def _per_block_ends(prompt_len: int, block: int, eagle: bool,
                    start: int = 0) -> list[int]:
    return _chunk_ends(_split_fn(), _shim(block, eagle, 0, multiblock=False),
                       prompt_len, start)


@pytest.mark.parametrize("tail_min", [0, 3136])
@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize("prompt_len", LENGTHS)
def test_split_chunk_plans(lane: str, tail_min: int, prompt_len: int) -> None:
    block, eagle = LANES[lane]
    split = _split_fn()
    expected = PLANS[(lane, tail_min)][prompt_len]
    assert _chunk_ends(split, _shim(block, eagle, tail_min), prompt_len) == expected
    if tail_min == 0:
        assert _chunk_ends(_dev2_mamba_block_aligned_split,
                           _shim(block, eagle, 0), prompt_len) == expected
    # Per-block (dev2 MTP lane / multi-block off) is unchanged.
    per_block = _per_block_ends(prompt_len, block, eagle)
    assert per_block == _chunk_ends(
        _dev2_mamba_block_aligned_split,
        _shim(block, eagle, 0, multiblock=False), prompt_len,
    )
    assert len(expected) <= len(per_block)


# (lane, tail_min, prompt, hit) -> chunk ends from the hit
HIT_PLANS = [
    ("nomtp", 0, 1000, 784, [1000]),
    ("nomtp", 0, 2048, 784, [1568, 2048]),
    ("nomtp", 3136, 2048, 784, [2048]),
    ("nomtp", 0, 3136, 1568, [2352, 3136]),
    ("nomtp", 3136, 3136, 1568, [3136]),
    ("nomtp", 0, 8192, 7056, [7840, 8192]),
    ("nomtp", 3136, 8192, 7056, [8192]),
    ("nomtp", 3136, 8192, 1568, [7840, 8192]),
    ("nomtp", 3136, 8192, 7840, [8192]),
    ("nomtp", 0, 32768, 784, [8624, 16464, 24304, 32144, 32768]),
    ("nomtp", 3136, 32768, 1568, [9408, 17248, 25088, 32144, 32768]),
    ("nomtp", 0, 32768, 31360, [32144, 32768]),
    ("nomtp", 3136, 32768, 31360, [32768]),
    ("mtp816", 0, 2048, 816, [2048]),
    ("mtp816", 0, 3136, 816, [1632, 3136]),
    ("mtp816", 3136, 3136, 816, [3136]),
    ("mtp816", 3136, 8192, 816, [7344, 8192]),
    ("mtp816", 3136, 8192, 7344, [8192]),
    ("mtp816", 3136, 32768, 1632, [9792, 17952, 26112, 31824, 32768]),
    ("mtp816", 3136, 32768, 31824, [32768]),
    ("mtp784", 0, 8192, 784, [7056, 8192]),
    ("mtp784", 3136, 8192, 7056, [8192]),
]


@pytest.mark.parametrize("lane,tail_min,prompt_len,hit,expected", HIT_PLANS)
def test_split_chunk_plans_after_prefix_hit(lane, tail_min, prompt_len, hit,
                                            expected) -> None:
    block, eagle = LANES[lane]
    split = _split_fn()
    assert _chunk_ends(split, _shim(block, eagle, tail_min), prompt_len,
                       start=hit) == expected


# PF3: decode-aware cap of 3 blocks while decodes run (idle plans = PLANS).
PF3_PLANS = {
    ("nomtp", 0): {
        450: [450], 785: [784, 785], 2048: [1568, 2048], 3136: [2352, 3136],
        8192: [2352, 4704, 7056, 7840, 8192],
    },
    ("nomtp", 3136): {
        450: [450], 785: [785], 2048: [2048], 3136: [2352, 3136],
        8192: [2352, 4704, 7056, 8192],
        32768: [2352, 4704, 7056, 9408, 11760, 14112, 16464, 18816, 21168,
                23520, 25872, 28224, 30576, 32768],
    },
    ("mtp816", 3136): {
        450: [450], 2048: [2048], 3136: [2448, 3136],
        8192: [2448, 4896, 7344, 8192],
        32768: [2448, 4896, 7344, 9792, 12240, 14688, 17136, 19584, 22032,
                24480, 26928, 29376, 31824, 32768],
    },
}


@pytest.mark.parametrize("lane,tail_min", sorted(PF3_PLANS))
def test_split_decode_aware_cap_plans(lane, tail_min) -> None:
    block, eagle = LANES[lane]
    split = _split_fn()
    for prompt_len, expected in PF3_PLANS[(lane, tail_min)].items():
        shim = _shim(block, eagle, tail_min, step_blocks=3)
        assert _chunk_ends(split, shim, prompt_len) == expected, prompt_len
        # Every non-final chunk is a block multiple (a valid checkpoint).
        assert all(e % block == 0 for e in expected[:-1])


def test_split_prefill_token_budget_holds_and_caps() -> None:
    split = _split_fn()
    B = 784

    def call(prompt_len, start, left, n=BUDGET):
        shim = _shim(B, False, 3136, prefill_left=left)
        req = SimpleNamespace(num_computed_tokens=start,
                              num_prompt_tokens=prompt_len, num_tokens=prompt_len)
        got = split(shim, req, min(n, prompt_len - start))
        return got, shim._sx_step_prefill_left

    # Budget below one block, more than one block to go: held (no sub-block
    # chunk that would force per-block chunks later).
    assert call(3000, 0, 500) == (0, 500)
    # A request with at most one block to go is never held.
    assert call(450, 0, 100) == (450, 0)
    assert call(8192, 7840, 100) == (352, 0)
    # Otherwise capped at a block multiple of the budget.
    assert call(8192, 0, 2000) == (1568, 432)
    assert call(8192, 0, 4096) == (3920, 176)
    # Enough budget: unchanged plan, budget consumed.
    assert call(2048, 0, 4096) == (2048, 2048)


def test_split_equals_dev2_with_new_switches_off() -> None:
    """tail_min 0 and no per-step cap: bitwise the 1.8.0-dev2 split; with the
    PF4 default only the skipped tail/shared-at-tail stop can differ."""
    split = _split_fn()
    rng = random.Random(0)
    checked = changed = 0
    for _ in range(300_000):
        block = rng.choice([784, 816])
        eagle = rng.random() < 0.3
        prompt = rng.choice([rng.randrange(1, 40_000), rng.randrange(1, 5_000)])
        outputs = rng.choice([0, 0, 0, rng.randrange(1, 3000)])
        start = rng.randrange(0, prompt + outputs)
        if rng.random() < 0.6:
            start = start // block * block
        request = SimpleNamespace(
            num_computed_tokens=start,
            num_prompt_tokens=prompt,
            num_tokens=prompt + outputs,
        )
        if rng.random() < 0.2:
            request._sx_align_shared_stop = rng.randrange(0, prompt) // block * block
        remaining = request.num_tokens - start
        if remaining <= 0:
            continue
        n = rng.randrange(1, min(remaining, BUDGET) + 1)
        if rng.random() < 0.5:
            n = min(remaining, BUDGET)
        multiblock = rng.random() < 0.8
        tail = rng.random() < 0.8
        max_blocks = rng.choice([0, 0, 2, 4])
        old = _dev2_mamba_block_aligned_split(
            _shim(block, eagle, 0, multiblock=multiblock, tail=tail,
                  max_blocks=max_blocks), request, n)
        same = split(_shim(block, eagle, 0, multiblock=multiblock, tail=tail,
                           max_blocks=max_blocks), request, n)
        assert same == old, (block, eagle, prompt, outputs, start, n)
        new = split(_shim(block, eagle, 3136, multiblock=multiblock, tail=tail,
                          max_blocks=max_blocks), request, n)
        prefill_end = max(prompt, request.num_tokens - 1)
        applies = (
            multiblock and start % block == 0 and start < prefill_end
            and start + n >= prefill_end and prefill_end - start <= 3136
        )
        if not applies:
            assert new == old
        else:
            assert old <= new <= n
            changed += new != old
        checked += 1
    assert checked > 250_000 and changed > 1_000


# --------------------------------------------------------------------------
# Real Scheduler: single-request plans, identical resend (miss == hit)
# --------------------------------------------------------------------------
HARNESS_LANES = {
    # name: (lane key, num_spec, async, attention block (None = state block))
    "nomtp": ("nomtp", 0, False, None),
    "mtp816_k4": ("mtp816", 4, False, None),
    "mtp816_k4_async": ("mtp816", 4, True, None),
    "mtp816_k4_attn16": ("mtp816", 4, True, 16),
    "mtp784_k2": ("mtp784", 2, False, None),
}


def _env(tail_min: int, **extra) -> dict[str, str]:
    env = {"SX_OPT_ALIGN_TAIL_MIN_TOKENS": str(tail_min)}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _expected_hit(first: list[int], prompt_len: int, block: int,
                  eagle: bool) -> int:
    reachable = (prompt_len - 1) // block * block - (block if eagle else 0)
    return max([0] + [e for e in first if e % block == 0 and e <= reachable])


@pytest.mark.parametrize("tail_min", [0, 3136])
@pytest.mark.parametrize("name", sorted(HARNESS_LANES))
@pytest.mark.parametrize("prompt_len", LENGTHS)
def test_scheduler_plans_and_identical_resend(name, tail_min, prompt_len) -> None:
    lane, k, async_sched, attn_block = HARNESS_LANES[name]
    block, eagle = LANES[lane]
    h = SchedHarness(block_size=block, hash_block_size=16, num_spec=k,
                     async_sched=async_sched, env=_env(tail_min),
                     seed=prompt_len, attn_block_size=attn_block,
                     num_blocks=4096 if attn_block is None else 12_000)
    assert h.sched._sx_align_multiblock
    assert h.sched._sx_align_lane == ("mtp" if k else "no-mtp")
    prompt = make_prompt(prompt_len, prompt_len)
    h.add_request("first", prompt, max_tokens=24)
    h.run_until_idle()
    first = h.prefill_chunk_ends("first")
    assert first == PLANS[(lane, tail_min)][prompt_len]
    if k:
        assert h.verify_rows > 0
    h.add_request("resend", prompt, max_tokens=24)
    h.run_until_idle()
    hit = _expected_hit(first, prompt_len, block, eagle)
    assert h.hits["resend"] == hit
    # The resend computes exactly the remaining chunks of the first run, so
    # the output on the prefix-cache hit equals the output on the miss.
    assert h.prefill_chunk_ends("resend") == [e for e in first if e > hit]


@pytest.mark.parametrize("async_sched", [False, True])
def test_mtp_lane_concurrent_long_prefills(async_sched) -> None:
    """4 x 8K plus short prompts in the MTP lane: multi-block chunks share the
    budget, every chunk starting on a boundary spans several blocks, and the
    decode rows keep verifying while prefills run."""
    h = SchedHarness(block_size=816, hash_block_size=16, num_spec=4,
                     async_sched=async_sched, seed=3)
    for i in range(4):
        h.add_request(f"L{i}", make_prompt(100 + i, 8192), max_tokens=64)
    for i in range(6):
        h.add_request(f"s{i}", make_prompt(200 + i, 450 + 97 * i), max_tokens=96)
    h.run_until_idle()
    assert h.spec_prefill_multiblock_chunks >= 4
    # The first long prompt is not split per block any more (per-block: 11).
    assert len(h.prefill_chunk_ends("L0")) <= 3
    for i in range(4):
        assert h.prefill_chunk_ends(f"L{i}")[-1] == 8192
    mixed = [s for s in h.step_log if s["prefill"] and s["decode"]]
    assert mixed, "expected mixed prefill + verify steps"


@pytest.mark.parametrize(
    "attn_block,hit,s1_ends,s2_ends",
    [
        # Attention block == state block (unified 816-token pages): the
        # Eagle-pruned attention hit is 816 (the raw 1632 is never reachable).
        (None, 816, [816, 2448, 4000], [2448, 4000]),
        # 16-token attention blocks: attention matches 1984, Eagle drops one
        # 16-token block, the state lookup reaches 1632.
        (16, 1632, [1632, 4000], [4000]),
    ],
)
def test_mtp_shared_system_prompt_eagle_checkpoint(attn_block, hit, s1_ends,
                                                   s2_ends) -> None:
    """The shared-prefix checkpoint follows what a later request's Eagle/MTP
    lookup can restore, so requests with the same 2000-token system prompt
    hit it in the MTP lane."""
    system = make_prompt(99, 2000)
    h = SchedHarness(block_size=816, hash_block_size=16, num_spec=4, seed=1,
                     attn_block_size=attn_block, num_blocks=12_000)
    for i in range(4):
        h.add_request(f"s{i}", system + make_prompt(100 + i, 2000), max_tokens=8)
        h.run_until_idle()
    assert [h.hits[f"s{i}"] for i in range(4)] == [0, 0, hit, hit]
    assert h.prefill_chunk_ends("s0") == [2448, 4000]
    assert h.prefill_chunk_ends("s1") == s1_ends
    assert h.prefill_chunk_ends("s2") == s2_ends
    assert h.prefill_chunk_ends("s3") == s2_ends


def test_nomtp_shared_system_prompt_with_tail_min() -> None:
    """No-MTP shared prefix with PF4: 3000-token prompts fit one step; the
    shared stop still checkpoints the 784 boundary of the system prompt."""
    system = make_prompt(98, 1000)
    h = SchedHarness(block_size=784, hash_block_size=16, seed=2)
    for i in range(4):
        h.add_request(f"s{i}", system + make_prompt(100 + i, 2000), max_tokens=8)
        h.run_until_idle()
    assert [h.hits[f"s{i}"] for i in range(4)] == [0, 0, 784, 784]
    assert h.prefill_chunk_ends("s0") == [3000]
    assert h.prefill_chunk_ends("s1") == [784, 3000]
    assert h.prefill_chunk_ends("s2") == [3000]


# Shared system prompt + short distinct suffixes whose shared checkpoint is
# also the requests' own tail position (the common "fixed instructions + short
# input" shape). PF4 skips the first request's tail stop; the second sharer
# must still take the shared-prefix stop so the third one on hits (review fix:
# the PF4 resend rule used to drop every shared stop at the tail, so no sharer
# ever restored the state). tail_min 0 is the dev2 policy.
# (block, k, attention block, system, suffix) -> tail_min -> (hits, chunk ends)
SHORT_SUFFIX_SHARED = {
    (784, 0, None, 1000, 300): {
        0: ([0, 784, 784, 784], [[784, 1300], [1300], [1300], [1300]]),
        3136: ([0, 0, 784, 784], [[1300], [784, 1300], [1300], [1300]]),
    },
    (784, 0, None, 1600, 400): {
        0: ([0, 1568, 1568, 1568], [[1568, 2000], [2000], [2000], [2000]]),
        3136: ([0, 0, 1568, 1568], [[2000], [1568, 2000], [2000], [2000]]),
    },
    (784, 0, 16, 1600, 400): {
        0: ([0, 1568, 1568, 1568], [[1568, 2000], [2000], [2000], [2000]]),
        3136: ([0, 0, 1568, 1568], [[2000], [1568, 2000], [2000], [2000]]),
    },
    (816, 4, None, 2000, 500): {
        0: ([0, 0, 816, 816],
            [[1632, 2500], [816, 1632, 2500], [1632, 2500], [1632, 2500]]),
        3136: ([0, 0, 816, 816], [[2500], [816, 2500], [2500], [2500]]),
    },
    (816, 4, 16, 2000, 500): {
        0: ([0, 1632, 1632, 1632], [[1632, 2500], [2500], [2500], [2500]]),
        3136: ([0, 0, 1632, 1632], [[2500], [1632, 2500], [2500], [2500]]),
    },
}


@pytest.mark.parametrize("async_sched", [False, True])
@pytest.mark.parametrize("tail_min", [0, 3136])
@pytest.mark.parametrize("case", sorted(SHORT_SUFFIX_SHARED, key=str))
def test_shared_system_prompt_short_suffix(case, tail_min, async_sched) -> None:
    block, k, attn_block, system_len, suffix_len = case
    want_hits, want_ends = SHORT_SUFFIX_SHARED[case][tail_min]
    system = make_prompt(98, system_len)
    h = SchedHarness(block_size=block, hash_block_size=16, num_spec=k, seed=2,
                     env=_env(tail_min), attn_block_size=attn_block,
                     num_blocks=12_000, async_sched=async_sched)
    for i in range(4):
        h.add_request(f"s{i}", system + make_prompt(100 + i, suffix_len),
                      max_tokens=8)
        h.run_until_idle()
    assert [h.hits[f"s{i}"] for i in range(4)] == want_hits
    assert [h.prefill_chunk_ends(f"s{i}") for i in range(4)] == want_ends


@pytest.mark.parametrize("attn_block", [None, 16])
@pytest.mark.parametrize("tail_min", [0, 3136])
@pytest.mark.parametrize("lane", ["nomtp", "mtp"])
def test_e2e_shared_expectation_matches_scheduler(lane, tail_min,
                                                  attn_block) -> None:
    """The GPU e2e --shared expectation (default geometry) equals what the
    real scheduler does in both attention page layouts."""
    from e2e_b3_scheduler import shared_expectation

    block, k, system_len = (816, 4, 2000) if lane == "mtp" else (784, 0, 1600)
    want = shared_expectation(lane, block, tail_min, system_len, 300)
    assert want is not None and want[-1] > 0
    system = make_prompt(97, system_len)
    h = SchedHarness(block_size=block, hash_block_size=16, num_spec=k, seed=3,
                     env=_env(tail_min), attn_block_size=attn_block,
                     num_blocks=12_000, async_sched=bool(k))
    for i in range(4):
        h.add_request(f"s{i}", system + make_prompt(300 + i, 300), max_tokens=8)
        h.run_until_idle()
    assert [h.hits[f"s{i}"] for i in range(4)] == want


def test_identical_resend_memo_repeats_and_is_bounded(monkeypatch) -> None:
    """PF4 resend memo: every identical resend of a prompt whose tail PF4
    skipped replays the first run's single chunk (miss == hit, no extra
    step). Once the bounded memo forgot the prompt, a resend is treated like
    any sharer: it takes the shared stop at the tail (one extra step) and
    the next resend hits that checkpoint."""
    from vllm.v1.core.sched import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "_SX_TAIL_SKIP_MEMO_MAX", 2)
    h = SchedHarness(block_size=784, hash_block_size=16, seed=4)
    assert h.sched._sx_tail_skipped is not None
    a = make_prompt(1, 2048)
    for rid in ("a0", "a1", "a2"):
        h.add_request(rid, a, max_tokens=4)
        h.run_until_idle()
        assert (h.hits[rid], h.prefill_chunk_ends(rid)) == (0, [2048]), rid
    # Two other skipped-tail prompts evict ``a`` (LRU, capacity 2).
    for i in range(2):
        h.add_request(f"o{i}", make_prompt(10 + i, 2048), max_tokens=4)
        h.run_until_idle()
    assert len(h.sched._sx_tail_skipped) == 2
    h.add_request("a3", a, max_tokens=4)
    h.run_until_idle()
    assert (h.hits["a3"], h.prefill_chunk_ends("a3")) == (0, [1568, 2048])
    h.add_request("a4", a, max_tokens=4)
    h.run_until_idle()
    assert (h.hits["a4"], h.prefill_chunk_ends("a4")) == (1568, [2048])


@pytest.mark.parametrize("env", [
    {"SX_OPT_ALIGN_TAIL_MIN_TOKENS": "0"},
    {"SX_OPT_ALIGN_TAIL_CHECKPOINT": "0"},
    {"SX_OPT_ALIGN_SHARED_CHECKPOINT": "0"},
])
def test_resend_memo_off_without_pf4_tail_skip(env) -> None:
    """The memo only exists when PF4 can skip a tail that a shared stop could
    re-request; with tail_min 0, no tail checkpoint (Option A) or no shared
    checkpoints the split is the dev2 multi-block split."""
    h = SchedHarness(block_size=784, hash_block_size=16, seed=4, env=env)
    assert h.sched._sx_tail_skipped is None


@pytest.mark.parametrize("lane,k", [("nomtp", 0), ("mtp816", 4)])
@pytest.mark.parametrize("tail_min", [0, 3136])
def test_decode_aware_cap_with_running_decode(lane, k, tail_min) -> None:
    block, _ = LANES[lane]
    for cap in (False, True):
        env = _env(tail_min)
        if cap:
            env["SX_OPT_PREFILL_CAP_WITH_DECODES"] = "1"
        h = SchedHarness(block_size=block, hash_block_size=16, num_spec=k,
                         env=env, seed=5)
        assert h.sched._sx_prefill_cap == cap
        h.add_request("d", make_prompt(1, 300), max_tokens=400)
        while h.requests["d"].num_output_tokens < 3:
            h.step()
        h.add_request("long", make_prompt(2, 8192), max_tokens=4)
        h.run_until_idle()
        got = h.prefill_chunk_ends("long")
        if not cap:
            assert got == PLANS[(lane, tail_min)][8192]
        else:
            key = (lane, tail_min)
            want = PF3_PLANS[key][8192] if key in PF3_PLANS else None
            if want is None:  # mtp816 without PF4
                want = [2448, 4896, 7344, 8192]
            assert got == want
        # Idle engine: the cap never applies.
        h2 = SchedHarness(block_size=block, hash_block_size=16, num_spec=k,
                          env=env, seed=6)
        h2.add_request("long", make_prompt(2, 8192), max_tokens=4)
        h2.run_until_idle()
        assert h2.prefill_chunk_ends("long") == PLANS[(lane, tail_min)][8192]


def test_decode_aware_token_budget_per_step() -> None:
    env = _env(3136, SX_OPT_PREFILL_CAP_WITH_DECODES=1,
               SX_OPT_PREFILL_CAP_BLOCKS=3, SX_OPT_PREFILL_CAP_TOKENS=4096)
    h = SchedHarness(block_size=784, hash_block_size=16, env=env, seed=7)
    h.add_request("d", make_prompt(1, 300), max_tokens=600)
    while h.requests["d"].num_output_tokens < 3:
        h.step()
    for i in range(3):
        h.add_request(f"L{i}", make_prompt(10 + i, 8192), max_tokens=4)
    h.run_until_idle()
    for info in h.step_log:
        if info["decode"] and info["prefill"]:
            # Rows with at most one block to go are never held (they may
            # overshoot the budget); every other row fits what is left.
            capped = sum(
                e - s for s, e in info["prefill"].values()
                if not (e == 8192 and e - s <= 784)
            )
            assert capped <= 4096, info
    for i in range(3):
        ends = h.prefill_chunk_ends(f"L{i}")
        assert ends[-1] == 8192 and all(e % 784 == 0 for e in ends[:-1])


@pytest.mark.parametrize("block,k", [(784, 0), (816, 4)])
@pytest.mark.parametrize("cap_tokens", [1, 500, 783])
def test_decode_aware_token_budget_below_one_block_is_raised(block, k,
                                                             cap_tokens) -> None:
    """A PF3 step budget below one state block would hold a long prompt (and,
    FCFS, every waiting request behind it) for as long as any decode runs;
    the policy raises it to one block, so the long prompt is admitted with a
    one-block chunk in the next step and the short prompt behind it too."""
    env = _env(3136, SX_OPT_PREFILL_CAP_WITH_DECODES=1,
               SX_OPT_PREFILL_CAP_TOKENS=cap_tokens)
    h = SchedHarness(block_size=block, hash_block_size=16, num_spec=k,
                     env=env, seed=5)
    assert h.sched._sx_prefill_cap_tokens == block
    h.add_request("d", make_prompt(1, 300), max_tokens=3000)
    while h.requests["d"].num_output_tokens < 3:
        h.step()
    h.add_request("long", make_prompt(2, 8192), max_tokens=4)
    h.add_request("short", make_prompt(3, 300), max_tokens=4)
    h.step()
    assert h.prefill_chunk_ends("long") == [block]
    assert h.prefill_chunk_ends("short") == [300]
    for _ in range(40):
        if h.requests["long"].num_computed_tokens >= 8192:
            break
        h.step()
    assert h.prefill_chunk_ends("long")[-1] == 8192
    assert not h.requests["d"].is_finished()  # the decode kept running


# --------------------------------------------------------------------------
# Randomized workloads with the recurrent-state oracle
# --------------------------------------------------------------------------
def _random_env(rng: random.Random) -> dict[str, str]:
    env = {
        "SX_OPT_ALIGN_TAIL_MIN_TOKENS": str(rng.choice([0, 3136, 3136, 1568])),
        "SX_OPT_ALIGN_MAX_CHUNK_BLOCKS": str(rng.choice([0, 0, 0, 2, 4])),
        "SX_OPT_ALIGN_SHARED_CHECKPOINT": rng.choice(["1", "1", "0"]),
        "SX_OPT_ALIGN_TAIL_CHECKPOINT": rng.choice(["1", "1", "1", "0"]),
    }
    if rng.random() < 0.4:
        env["SX_OPT_PREFILL_CAP_WITH_DECODES"] = "1"
        env["SX_OPT_PREFILL_CAP_BLOCKS"] = str(rng.choice([1, 2, 3]))
        env["SX_OPT_PREFILL_CAP_TOKENS"] = str(rng.choice([0, 0, 2048, 4096]))
    return env


def _run_random(seed: int, k: int, *, preempt: bool) -> SchedHarness:
    rng = random.Random(seed)
    block = rng.choice([784, 816]) if k else 784
    hash_block = rng.choice([16, block])
    attn_block = None if preempt or hash_block != 16 else rng.choice([None, 16])
    h = SchedHarness(
        block_size=block,
        hash_block_size=hash_block,
        attn_block_size=attn_block,
        budget=rng.choice([1000, 2048, 8192, 8192]),
        max_num_seqs=rng.choice([2, 8, 24]),
        num_blocks=(rng.choice([160, 220]) if preempt
                    else 4096 if attn_block is None else 40_000),
        num_spec=k,
        async_sched=rng.random() < 0.5,
        env=_random_env(rng),
        seed=seed,
    )
    bases = [make_prompt(1000 + j, 9000) for j in range(3)]
    # Identical resends and short-suffix sharers (PF4 resend memo) come from
    # a separate stream, so the rest of each seed's workload is unchanged.
    extra = random.Random(seed ^ 0x5EED)
    sent: list[list[int]] = []
    n_req = 0
    for _ in range(6):
        for _ in range(rng.randrange(1, 5)):
            share = rng.choice([0, 500, block, 2 * block, 2000, 5000])
            prompt = rng.choice(bases)[:share] + make_prompt(
                rng.randrange(1 << 30), rng.randrange(1, 5000 if preempt else 7000)
            )
            h.add_request(f"q{n_req}", prompt, max_tokens=rng.randrange(1, 300))
            n_req += 1
            sent.append(prompt)
            roll = extra.random()
            if roll < 0.25:
                resend = extra.choice(sent)
            elif roll < 0.45:
                resend = extra.choice(bases)[: extra.choice([block, 2 * block, 1600])]
                resend = resend + make_prompt(extra.randrange(1 << 30),
                                              extra.randrange(1, 400))
            else:
                continue
            h.add_request(f"q{n_req}", resend,
                          max_tokens=extra.randrange(1, 300))
            n_req += 1
            sent.append(resend)
        for _ in range(rng.randrange(1, 25)):
            if h.busy():
                h.step()
    h.run_until_idle()  # the oracle raises on any violation
    assert len(h.finished) == n_req
    return h


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(10))
def test_randomized_mtp_workloads(seed: int, k: int) -> None:
    h = _run_random(1000 * k + seed, k, preempt=False)
    assert h.verify_rows > 0


@pytest.mark.parametrize("k", [0, 2, 4])
@pytest.mark.parametrize("seed", range(8))
def test_randomized_workloads_with_preemption(seed: int, k: int) -> None:
    # Small KV pools preempt on most seeds; the oracle holds either way.
    _run_random(50_000 + 1000 * k + seed, k, preempt=True)


def test_randomized_workloads_preempt_at_least_sometimes() -> None:
    total = 0
    for seed in range(6):
        h = _run_random(50_000 + 4000 + seed, 4, preempt=True)
        total += sum(r.num_preemptions for r in h.requests.values())
    assert total > 0


# --------------------------------------------------------------------------
# Negative controls: the oracle catches the bugs it is meant to catch
# --------------------------------------------------------------------------
def test_oracle_detects_multiblock_from_unaligned_start() -> None:
    def unsafe_split(self, request, num_new_tokens, a=0, b=0):
        start = request.num_computed_tokens + a + b
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens
        B = self.mamba_state_block_size
        end = start + num_new_tokens
        if end < prefill_end and end // B * B > start:
            end = end // B * B
        return end - start

    h = SchedHarness(block_size=816, num_spec=4, seed=1)
    h.sched._mamba_block_aligned_split = unsafe_split.__get__(h.sched)
    # r2 first gets the 192-token budget remainder (unaligned start), then
    # the unsafe split lets it span several blocks from mid-block.
    h.add_request("r1", make_prompt(1, 8000), max_tokens=4)
    h.add_request("r2", make_prompt(2, 8192), max_tokens=4)
    with pytest.raises(OracleError):
        h.run_until_idle()


def test_oracle_detects_missing_decode_boundary_copy() -> None:
    """Drop the post-process copy that checkpoints a boundary crossed by
    accepted verify tokens: a registered Mamba block then holds a stale
    state, and the oracle must flag it. (Async scheduling registers the
    decode-crossing checkpoint right after the step's output, before the next
    allocation frees the column; the sync scheduler never registers it.)"""
    h = SchedHarness(block_size=816, num_spec=4, seed=1, async_sched=True,
                     worker_bug="skip_postcopy")
    h.add_request("r1", make_prompt(1, 700), max_tokens=400)
    with pytest.raises(OracleError):
        h.run_until_idle()


def test_mtp_follow_up_hits_decode_checkpoint_async() -> None:
    """Multi-turn in the MTP lane (async): the first turn prefills 2048
    tokens in one step (PF4) and crosses 2448 and 3264 while verifying; the
    follow-up restores the decode-crossing checkpoint at 2448 (the Eagle rule
    keeps the hit one block below the 3264 attention match)."""
    h = SchedHarness(block_size=816, hash_block_size=16, num_spec=4,
                     async_sched=True, seed=11)
    h.add_request("t1", make_prompt(7, 2048), max_tokens=1300)
    h.run_until_idle()
    assert h.prefill_chunk_ends("t1") == [2048]
    follow_up = list(h.requests["t1"]._all_token_ids) + make_prompt(8, 300)
    h.add_request("t2", follow_up, max_tokens=4)
    h.run_until_idle()
    assert h.hits["t2"] == 2448


# --------------------------------------------------------------------------
# Policy gating (real Scheduler.__init__)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs,env,expected_lane",
    [
        (dict(num_spec=0), {}, "no-mtp"),
        (dict(num_spec=4), {}, "mtp"),
        (dict(num_spec=1), {}, "mtp"),
        (dict(num_spec=5), {}, "mtp"),
        (dict(num_spec=7), {}, "mtp"),
        (dict(num_spec=8), {}, None),
        (dict(num_spec=4, parallel_drafting=True), {}, None),
        (dict(num_spec=4), {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "0"}, None),
        (dict(num_spec=4), {"SX_OPT_ALIGN_MULTIBLOCK": "0"}, None),
        (dict(num_spec=0), {"SX_OPT_ALIGN_MULTIBLOCK": "0"}, None),
        (dict(num_spec=4, method="eagle3"), {}, None),
        (dict(num_spec=4, method="eagle3"), {"SX_OPT_ALIGN_MULTIBLOCK": "force"},
         None),
        (dict(num_spec=4, native_mtp=False), {}, None),
        (dict(num_spec=4, sm70=False), {}, None),
        (dict(num_spec=0, sm70=False), {}, None),
        (dict(num_spec=4, tp=2), {}, None),
    ],
)
def test_policy_gating_real_init(kwargs, env, expected_lane) -> None:
    h = SchedHarness(block_size=816 if kwargs.get("num_spec") else 784,
                     env=env, **kwargs)
    s = h.sched
    assert s._sx_align_lane == expected_lane
    assert s._sx_align_multiblock == (expected_lane is not None)
    if expected_lane is None:
        assert (s._sx_align_tail_min, s._sx_prefill_cap) == (0, False)
    else:
        assert s._sx_align_tail_min == 3136
        assert s._sx_prefill_cap is False  # PF3 is opt-in
        assert s._sx_align_shared is True


def test_policy_env_values() -> None:
    env = {
        "SX_OPT_ALIGN_TAIL_MIN_TOKENS": "0",
        "SX_OPT_PREFILL_CAP_WITH_DECODES": "1",
        "SX_OPT_PREFILL_CAP_BLOCKS": "2",
        "SX_OPT_PREFILL_CAP_TOKENS": "4096",
        "SX_OPT_ALIGN_MAX_CHUNK_BLOCKS": "4",
    }
    s = SchedHarness(block_size=816, num_spec=4, env=env).sched
    assert (s._sx_align_tail_min, s._sx_prefill_cap, s._sx_prefill_cap_blocks,
            s._sx_prefill_cap_tokens, s._sx_align_max_blocks) == (0, True, 2,
                                                                 4096, 4)
    # With no decode running the step keeps the static policy.
    s._sx_begin_prefill_cap_step()
    assert (s._sx_step_max_blocks, s._sx_step_prefill_left) == (4, None)


# --------------------------------------------------------------------------
# The GPU e2e script's expectation model matches the real split
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tail_min", [0, 3136])
@pytest.mark.parametrize("lane", sorted(LANES))
def test_e2e_expectation_model_matches_split(lane: str, tail_min: int) -> None:
    from e2e_b3_scheduler import chunk_plan, expected_hit

    block, eagle = LANES[lane]
    for n in LENGTHS + [1200, 5000, 16384, 65536]:
        assert chunk_plan(n, block, eagle, tail_min) == _chunk_ends(
            _split_fn(), _shim(block, eagle, tail_min), n
        )
        assert chunk_plan(n, block, eagle, 0, per_block=True) == \
            _per_block_ends(n, block, eagle)
        first = chunk_plan(n, block, eagle, tail_min)
        assert expected_hit(n, block, eagle, tail_min) == _expected_hit(
            first, n, block, eagle
        )
