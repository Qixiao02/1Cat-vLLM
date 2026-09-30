# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache hits of multi-block chunking equal upstream per-block's.

Run (CPU only, no vLLM install needed; see ``align_boot.py``):

    python sx_tests/align-multiblock/test_hit_parity.py
    python -m pytest -q -s sx_tests/align-multiblock/test_hit_parity.py

Two real schedulers get the same requests one after the other (fresh
prompts, identical resends, next turns of finished conversations, siblings
sharing a prefix). The baseline is pristine upstream: ``scheduler.py`` and
``single_type_kv_cache_manager.py`` from the base commit (the working tree
with ``SX_OPT_ALIGN_MULTIBLOCK=0`` when that commit is not readable). The
other is the working tree with the lane admitted. The recurrent-state oracle
of ``align_harness`` runs in both.

Asserted per layout, retention setting and scheduler (synchronous and async
with one step in flight):
* every request's hit length is equal in both runs;
* the Mamba blocks left in the prefix cache cover the same states;
* no cached block was evicted in either run (hits would depend on it);
* multi-block never takes more prefill steps; under dense retention (no
  interval, or an interval of one block) it is off and the chunk plans are
  identical;
* the step counts and resend hits of the design's Flash-Next table, and the
  review's sibling-prompt example (a retention interval of one full-budget
  chunk keeps every chunk end reusable).
Each case prints its request count, mismatches and prefill steps (``-s``).
Expected: all pass, 0 mismatches.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402
from align_harness import SchedHarness, make_prompt, run_sequential  # noqa: E402

FLASH_NEXT = dict(layout="flashnext", block_size=784, num_blocks=8192)
MTP = dict(layout="flashnext", block_size=816, num_blocks=8192, num_spec=4,
           method="mtp")
DFLASH_4096 = dict(layout="27b", block_size=4096, num_blocks=4096, num_spec=7,
                   method="dflash", budget=16384, threshold=8192)
DFLASH_3296 = dict(layout="27b", block_size=3296, num_blocks=4096, num_spec=7,
                   method="dflash", budget=16384, threshold=8000)
SPEC_ENV = {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash,mtp"}

# name -> (harness arguments, trials, requests per trial, longest prompt,
#          longest output, dense retention)
CASES = {
    # Flash-Next, no speculative decoding, budget 8192.
    "flashnext/retention=0": (dict(FLASH_NEXT, retention=0), 10, 12, 40_000, 900,
                              False),
    "flashnext/retention=4blocks": (dict(FLASH_NEXT, retention=3136), 8, 12, 40_000,
                                    900, False),
    "flashnext/retention=7840": (dict(FLASH_NEXT, retention=7840), 8, 12, 40_000,
                                 900, False),
    "flashnext/retention=None": (dict(FLASH_NEXT, retention=None), 4, 12, 40_000,
                                 900, True),
    "flashnext/retention=1block": (dict(FLASH_NEXT, retention=784), 3, 12, 40_000,
                                   900, True),
    # 16-token attention pages under the 784-token state block: attention
    # hits are finer than state hits, which exercises shared-prefix stops.
    "flashnext/attn16/retention=0": (
        dict(FLASH_NEXT, retention=0, attn_block_size=16, num_blocks=40_000),
        8, 12, 9_000, 900, False,
    ),
    # Block hashes on the state block itself.
    "flashnext/hash784/retention=0": (
        dict(FLASH_NEXT, retention=0, hash_block_size=784), 6, 12, 40_000, 900, False,
    ),
    # MTP-shaped: Eagle tail rule, k = 4, 816-token state block.
    "mtp-eagle/retention=0": (dict(MTP, retention=0), 8, 12, 40_000, 900, False),
    "mtp-eagle/retention=4blocks": (dict(MTP, retention=3264), 6, 12, 40_000, 900,
                                    False),
    "mtp-eagle/retention=None": (dict(MTP, retention=None), 3, 12, 40_000, 900,
                                 True),
    "mtp-eagle/attn16/retention=0": (
        dict(MTP, retention=0, attn_block_size=16, num_blocks=40_000),
        6, 12, 9_000, 900, False,
    ),
    # 27B-shaped: DFlash2, non-Eagle, k = 7, draft sliding-window group.
    "27b/B=4096/retention=0": (dict(DFLASH_4096, retention=0), 8, 10, 70_000, 5_000,
                               False),
    "27b/B=4096/retention=8192": (dict(DFLASH_4096, retention=8192), 6, 10, 70_000,
                                  5_000, False),
    "27b/B=4096/budget=8192": (
        dict(DFLASH_4096, retention=0, budget=8192, threshold=0), 5, 10, 70_000,
        5_000, False,
    ),
    "27b/B=3296/retention=0": (dict(DFLASH_3296, retention=0), 8, 10, 60_000, 4_000,
                               False),
    "27b/B=4096/retention=None": (dict(DFLASH_4096, retention=None), 3, 10, 70_000,
                                  5_000, True),
}


def _baseline(**kwargs) -> SchedHarness:
    """Upstream per-block chunking."""
    if align_boot.have_base():
        upstream = align_boot.load(upstream=align_boot.PATCHED_MODULES)
        return SchedHarness(universe=upstream, **kwargs)
    return SchedHarness(env={"SX_OPT_ALIGN_MULTIBLOCK": "0"}, **kwargs)


def _multiblock(**kwargs) -> SchedHarness:
    return SchedHarness(env=SPEC_ENV, **kwargs)


@pytest.mark.parametrize("queue_depth", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_hits_and_cached_states_equal_upstream(case: str, queue_depth: int) -> None:
    kwargs, trials, num_requests, max_len, max_output, dense = CASES[case]
    requests = mismatches = steps_upstream = steps_multiblock = 0
    for trial in range(trials):
        workload = (1000 * queue_depth + trial, num_requests, max_len, max_output)
        a = run_sequential(_baseline(queue_depth=queue_depth, **kwargs), *workload)
        b = run_sequential(_multiblock(queue_depth=queue_depth, **kwargs), *workload)
        assert not a.multiblock
        assert b.multiblock != dense
        assert a.requests.keys() == b.requests.keys()
        assert a.evictions == b.evictions == 0
        requests += len(a.requests)
        mismatches += sum(a.hits[rid] != b.hits[rid] for rid in a.requests)
        assert a.hits == b.hits, (case, trial)
        assert a.all_hits == b.all_hits
        assert a.cached_state_boundaries() == b.cached_state_boundaries()
        steps_upstream += a.prefill_steps()
        steps_multiblock += b.prefill_steps()
        assert b.prefill_steps() <= a.prefill_steps()
        if dense:
            assert a.chunks == b.chunks
    assert mismatches == 0
    assert dense == (steps_multiblock == steps_upstream)
    print(
        f"\n[parity] {case} queue_depth={queue_depth}: requests={requests} "
        f"hit mismatches={mismatches} prefill steps upstream={steps_upstream} "
        f"multi-block={steps_multiblock}"
    )


# (prompt, steps upstream, steps multi-block, resend hit): the design's table.
FLASH_NEXT_TABLE = [
    (3_000, 4, 2, 2_352),
    (8_192, 11, 2, 7_840),
    (32_768, 42, 6, 32_144),
    (65_536, 84, 10, 65_072),
    (110_000, 141, 15, 109_760),
]


@pytest.mark.parametrize("queue_depth", [1, 2])
@pytest.mark.parametrize("prompt_len,per_block,multi_block,hit", FLASH_NEXT_TABLE)
def test_flash_next_steps_and_resend_hit(prompt_len, per_block, multi_block, hit,
                                         queue_depth) -> None:
    prompt = make_prompt(prompt_len, prompt_len)
    for build, steps in ((_baseline, per_block), (_multiblock, multi_block)):
        h = build(queue_depth=queue_depth, **FLASH_NEXT)
        h.add_request("first", prompt, max_tokens=8)
        h.run_until_idle()
        assert h.prefill_steps("first") == steps
        h.add_request("resend", prompt, max_tokens=8)
        h.run_until_idle()
        assert (h.hits["first"], h.hits["resend"]) == (0, hit)
        # The resend computes the chunks the first run had after the hit.
        assert h.prefill_chunk_ends("resend") == [
            end for end in h.prefill_chunk_ends("first") if end > hit
        ]


@pytest.mark.parametrize(
    "kwargs,per_block,multi_block,hit",
    [
        # Grid 4096 with the documented flags: 8192-token chunks.
        (DFLASH_4096, 27, 14, 106_496),
        # Today's 3296 grid with threshold 8000: 6592-token chunks.
        (DFLASH_3296, 34, 18, 108_768),
    ],
)
def test_27b_steps_and_resend_hit(kwargs, per_block, multi_block, hit) -> None:
    prompt = make_prompt(27, 110_000)
    for build, steps in ((_baseline, per_block), (_multiblock, multi_block)):
        h = build(queue_depth=2, **kwargs)
        h.add_request("first", prompt, max_tokens=8)
        h.run_until_idle()
        assert h.prefill_steps("first") == steps
        h.add_request("resend", prompt, max_tokens=8)
        h.run_until_idle()
        assert h.hits["resend"] == hit


def _sibling_hits(build, retention) -> tuple[list[int], list[int]]:
    """Three questions about one 100K document."""
    document = make_prompt(7, 100_000)
    h = build(queue_depth=2, **dict(FLASH_NEXT, retention=retention))
    for i in range(3):
        h.add_request(f"q{i}", document + make_prompt(100 + i, 2_000), max_tokens=4)
        h.run_until_idle()
    rids = [f"q{i}" for i in range(3)]
    return [h.hits[rid] for rid in rids], [h.prefill_steps(rid) for rid in rids]


def test_sibling_prompts_need_a_retention_interval() -> None:
    # Default retention keeps replay and detected shared-prefix boundaries
    # only: the second sibling finds nothing, in upstream as in multi-block.
    assert _sibling_hits(_baseline, 0)[0] == [0, 0, 99_568]
    assert _sibling_hits(_multiblock, 0)[0] == [0, 0, 99_568]
    # An interval of one full-budget chunk keeps every chunk end without
    # adding a step to a cold prefill.
    hits, steps = _sibling_hits(_multiblock, 7_840)
    assert hits == [0, 94_080, 99_568]
    assert steps[0] == _sibling_hits(_multiblock, 0)[1][0]
    assert _sibling_hits(_baseline, 7_840)[0] == hits


def test_multi_turn_hits_the_decode_checkpoint() -> None:
    """The next turn restores the last boundary the first turn decoded past."""
    prompt = make_prompt(3, 50_000)
    for build in (_baseline, _multiblock):
        h = build(queue_depth=2, **FLASH_NEXT)
        h.add_request("turn1", prompt, max_tokens=600)
        h.run_until_idle()
        conversation = list(h.requests["turn1"]._all_token_ids)
        assert len(conversation) == 50_600
        h.add_request("turn2", conversation + make_prompt(4, 200), max_tokens=4)
        h.run_until_idle()
        assert h.hits["turn2"] == 50_600 // 784 * 784


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
