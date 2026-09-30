# SPDX-License-Identifier: Apache-2.0
"""Mamba state blocks while steps are in flight (``MambaManager``).

Run (CPU only, no vLLM install needed; see ``align_boot.py``):

    python sx_tests/align-multiblock/test_inflight_state_blocks.py
    python -m pytest -q -s sx_tests/align-multiblock/test_inflight_state_blocks.py

With async scheduling a prefill chunk is scheduled while the previous one is
still running, so the state block the previous chunk copies from cannot be
freed yet. Upstream's per-block chunks leave those blocks side by side and a
backward scan frees them later. A multi-block chunk leaves null blocks in
between, the scan stops there, and without the change in
``MambaManager.remove_skipped_blocks`` every chunk leaves one state block per
Mamba group behind until the request ends.

Asserted, on the real ``AsyncScheduler`` with one and two steps in flight:
* during and after a 110K and a 262K (27B-shaped: 255K) prefill a request
  holds as many real state columns per Mamba group as under upstream's
  per-block chunks: ``1 + k`` once it decodes (1 without speculative
  decoding), ``1 + k + steps in flight`` at most before;
* the working-tree scheduler on upstream's ``MambaManager`` does pile them
  up (14 and 34 columns), so the check above fails without the change;
* with upstream's per-block chunks the change frees nothing by itself: block
  tables, free-queue order and prefix cache are identical to pristine
  upstream after every step of random overlapping workloads, with
  preemption, in every lane;
* a state block freed this way keeps its hash and serves later prefix hits;
* abort and preemption drop the bookkeeping; no block is freed twice; the
  list of blocks waiting to be freed stays as short as the pipeline is deep.
Each leak case prints its numbers (``-s``).
Expected: all pass. Comparisons with upstream need the base commit in git
and are skipped otherwise.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402
from align_harness import (  # noqa: E402
    SchedHarness,
    concurrent_workload,
    make_prompt,
)

LANES = {
    "flashnext": dict(layout="flashnext", block_size=784),
    "mtp-eagle": dict(
        layout="flashnext", block_size=816, num_spec=4, method="mtp",
        env={"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "mtp"},
    ),
    "27b-4096": dict(
        layout="27b", block_size=4096, num_spec=7, method="dflash", budget=16384,
        threshold=8192, env={"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash"},
    ),
    "27b-3296": dict(
        layout="27b", block_size=3296, num_spec=7, method="dflash", budget=16384,
        threshold=8000, env={"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash"},
    ),
}
SMALL_POOLS = {"flashnext": 190, "mtp-eagle": 190, "27b-4096": 110, "27b-3296": 110}
LONG_PROMPTS = {
    "flashnext": (110_000, 262_000),
    "mtp-eagle": (110_000, 262_000),
    "27b-4096": (110_000, 255_000),
    "27b-3296": (110_000, 255_000),
}
LEAK_CASES = [
    (lane, prompt_len) for lane in sorted(LANES) for prompt_len in LONG_PROMPTS[lane]
]


def _upstream() -> align_boot.Universe:
    if not align_boot.have_base():
        pytest.skip(f"base commit {align_boot.BASE_REV} is not readable")
    return align_boot.load(upstream=align_boot.PATCHED_MODULES)


def _mamba_managers(h: SchedHarness):
    managers = h.manager.coordinator.single_type_managers
    return [managers[gid] for gid in h.mamba_group_ids]


def _prefill_then_decode(h: SchedHarness, prompt_len: int):
    """State columns held 300 decode tokens in, and the most held before."""
    h.add_request("r", make_prompt(prompt_len, prompt_len), max_tokens=320)
    while h.requests["r"].num_output_tokens < 300:
        h.step()
    columns = h.state_columns("r")
    h.run_until_idle()
    return columns, h.peak_state_columns["r"], h.prefill_steps("r")


# --------------------------------------------------------------------------
# The leak
# --------------------------------------------------------------------------
@pytest.mark.parametrize("queue_depth", [1, 2, 3])
@pytest.mark.parametrize("lane,prompt_len", LEAK_CASES)
def test_state_columns_held_equal_upstream(lane, prompt_len, queue_depth) -> None:
    kwargs = dict(LANES[lane], queue_depth=queue_depth)
    k = kwargs.get("num_spec", 0)
    h = SchedHarness(**kwargs)
    assert h.multiblock
    columns, peak, steps = _prefill_then_decode(h, prompt_len)
    assert columns == [1 + k] * len(h.mamba_group_ids)
    # The running column, the speculative ones, and one per unfinished step.
    assert peak == 1 + k + queue_depth
    report = (
        f"\n[state columns] {lane} {prompt_len} tokens queue_depth={queue_depth}: "
        f"multi-block {steps} steps, {columns[0]} held while decoding, peak {peak}"
    )
    if align_boot.have_base():
        upstream = SchedHarness(**dict(kwargs, universe=_upstream(), env=None))
        up_columns, up_peak, up_steps = _prefill_then_decode(upstream, prompt_len)
        assert (columns, peak) == (up_columns, up_peak)
        assert steps < up_steps
        report += f"; upstream {up_steps} steps, {up_columns[0]} held, peak {up_peak}"
    print(report)


@pytest.mark.parametrize("queue_depth", [2, 3])
@pytest.mark.parametrize("lane,prompt_len", LEAK_CASES)
def test_upstream_allocator_piles_up_state_blocks(lane, prompt_len,
                                                  queue_depth) -> None:
    """The working-tree scheduler on upstream's ``MambaManager``: the leak."""
    _upstream()
    leaky = align_boot.load(upstream=[align_boot.ALLOCATOR])
    kwargs = dict(LANES[lane], queue_depth=queue_depth, universe=leaky)
    k = kwargs.get("num_spec", 0)
    h = SchedHarness(**kwargs)
    assert h.multiblock
    columns, peak, steps = _prefill_then_decode(h, prompt_len)
    assert min(columns) > 1 + k
    assert peak > 1 + k + queue_depth
    print(
        f"\n[state columns] {lane} {prompt_len} tokens queue_depth={queue_depth}: "
        f"without the allocator change {columns[0]} held while decoding "
        f"({steps} prefill steps)"
    )
    if lane == "flashnext" and queue_depth == 2:
        # The review's measurement.
        assert columns[0] == {110_000: 14, 262_000: 34}[prompt_len]


def test_synchronous_scheduler_never_defers_a_free() -> None:
    """Without steps in flight the previous state block is always freeable."""
    h = SchedHarness(**LANES["27b-4096"], queue_depth=1)
    h.add_request("r", make_prompt(1, 120_000), max_tokens=40)
    while h.busy():
        h.step()
        for manager in _mamba_managers(h):
            assert not manager.stale_state_block_idxs
    assert h.multiblock_chunks > 0


@pytest.mark.parametrize("queue_depth", [2, 3])
@pytest.mark.parametrize("lane", sorted(LANES))
def test_deferred_frees_stay_bounded(lane: str, queue_depth: int) -> None:
    h = SchedHarness(**LANES[lane], queue_depth=queue_depth, check_pool=True)
    h.add_request("r", make_prompt(1, LONG_PROMPTS[lane][0]), max_tokens=40)
    longest = 0
    while h.busy():
        h.step()
        for manager in _mamba_managers(h):
            for idxs in manager.stale_state_block_idxs.values():
                assert idxs == sorted(set(idxs))
                longest = max(longest, len(idxs))
    assert 0 < longest <= queue_depth
    for manager in _mamba_managers(h):
        assert not manager.stale_state_block_idxs


# --------------------------------------------------------------------------
# Upstream's per-block schedule is untouched
# --------------------------------------------------------------------------
PER_BLOCK_CASES = [
    # (queue depth, random early output returns, pool too small)
    (1, False, False),
    (1, False, True),
    (2, False, False),
    (2, False, True),
    (3, False, False),
]


@pytest.mark.parametrize("queue_depth,jitter,preempt", PER_BLOCK_CASES)
@pytest.mark.parametrize("lane", sorted(LANES))
def test_per_block_schedule_is_identical_to_upstream(lane, queue_depth, jitter,
                                                     preempt) -> None:
    upstream = _upstream()
    steps = frees_deferred = preemptions = 0
    for seed in range(5):
        rng = random.Random(seed)
        block = LANES[lane]["block_size"]
        kwargs = dict(
            LANES[lane],
            queue_depth=queue_depth,
            # Early returns only without speculative decoding, see
            # test_state_oracle.py.
            jitter=jitter or (lane == "flashnext" and seed % 2 == 1),
            num_blocks=SMALL_POOLS[lane] if preempt else 2048,
            retention=rng.choice([0, 4 * block, None]),
            max_num_seqs=rng.choice([2, 8, 24]),
            seed=seed,
            check_pool=True,
        )
        a = SchedHarness(**dict(kwargs, universe=upstream, env=None))
        b = SchedHarness(**dict(kwargs, env={"SX_OPT_ALIGN_MULTIBLOCK": "0"}))
        assert not a.multiblock and not b.multiblock
        workload = concurrent_workload(
            seed, block, max_prompt_blocks=6.0 if preempt else 9.0
        )
        for action in workload:
            if action[0] == "add":
                a.add_request(*action[1:])
                b.add_request(*action[1:])
            elif a.busy():
                a.step()
                b.step()
                steps += 1
                assert a.snapshot() == b.snapshot(), (lane, seed, a.steps)
                frees_deferred += any(
                    manager.stale_state_block_idxs for manager in _mamba_managers(b)
                )
        while a.busy():
            a.step()
            b.step()
            steps += 1
            assert a.snapshot() == b.snapshot(), (lane, seed, a.steps)
        assert not b.busy()
        assert a.chunks == b.chunks and a.all_hits == b.all_hits
        preemptions += sum(r.num_preemptions for r in a.requests.values())
    assert steps > 500
    assert (preemptions > 0) == preempt
    # The new list is in use whenever steps overlap, and still nothing
    # differs: the blocks on it were freed by upstream's own scan.
    assert (frees_deferred > 0) == (queue_depth > 1)


# --------------------------------------------------------------------------
# Freed state blocks keep serving hits
# --------------------------------------------------------------------------
@pytest.mark.parametrize("queue_depth", [2, 3])
def test_freed_state_blocks_serve_prefix_hits(queue_depth: int) -> None:
    """A sibling hits a boundary state its still running peer has let go."""
    document = make_prompt(7, 100_000)

    def cached_state_tokens(h: SchedHarness, ref_cnt: int) -> set[int]:
        """Token counts of the first Mamba group's cached states."""
        kvu = h.universe.kv_cache_utils
        return {
            h.ident_by_hash[bytes(kvu.get_block_hash(blk.block_hash))][0]
            for blk in h.pool.blocks
            if blk.block_hash is not None
            and kvu.get_group_id(blk.block_hash) == h.mamba_group_ids[0]
            and blk.ref_cnt == ref_cnt
        }

    periodic = {7840 * j for j in range(1, 14)}
    results = []
    builds = [dict()]
    if align_boot.have_base():
        builds.append(dict(universe=_upstream()))
        # Upstream's allocator under multi-block chunks: `a` still holds all
        # of them but the last, which upstream's own free reaches.
        leaky = SchedHarness(layout="flashnext", block_size=784, retention=7840,
                             queue_depth=queue_depth,
                             universe=align_boot.load(upstream=[align_boot.ALLOCATOR]))
        leaky.add_request("a", document + make_prompt(1, 2_000), max_tokens=600)
        while leaky.requests["a"].num_output_tokens < 100:
            leaky.step()
        assert cached_state_tokens(leaky, ref_cnt=1) == periodic - {101_920}
    for build in builds:
        h = SchedHarness(layout="flashnext", block_size=784, retention=7840,
                         queue_depth=queue_depth, check_pool=True, **build)
        h.add_request("a", document + make_prompt(1, 2_000), max_tokens=600)
        while h.requests["a"].num_output_tokens < 100:
            h.step()
        # `a` decodes and holds its running state only. The periodic
        # boundary states it computed are cached and unreferenced.
        assert h.state_columns("a") == [1] * len(h.mamba_group_ids)
        assert cached_state_tokens(h, ref_cnt=0) == periodic
        h.add_request("b", document + make_prompt(2, 2_000), max_tokens=4)
        while not h.requests["b"].is_finished():
            h.step()
        assert not h.requests["a"].is_finished()
        h.run_until_idle()
        h.add_request("c", document + make_prompt(3, 2_000), max_tokens=4)
        h.run_until_idle()
        assert h.evictions == 0
        results.append([h.hits[rid] for rid in "abc"])
    # `b` restores the last periodic boundary inside the document while `a`
    # still runs; `c` the junction `b` detected.
    assert results[0] == [0, 94_080, 99_568]
    assert all(result == results[0] for result in results)


# --------------------------------------------------------------------------
# Abort and preemption
# --------------------------------------------------------------------------
@pytest.mark.parametrize("queue_depth", [2, 3])
@pytest.mark.parametrize("lane", sorted(LANES))
def test_abort_drops_the_deferred_frees(lane: str, queue_depth: int) -> None:
    h = SchedHarness(**LANES[lane], queue_depth=queue_depth, check_pool=True)
    prompt = make_prompt(1, LONG_PROMPTS[lane][0])
    h.add_request("r", prompt, max_tokens=8)
    while not all(m.stale_state_block_idxs.get("r") for m in _mamba_managers(h)):
        h.step()
    assert h.requests["r"].num_in_flight_tokens > 0
    h.abort("r")
    for manager in _mamba_managers(h):
        assert not manager.stale_state_block_idxs
        assert not manager.last_state_block_idx
    h.assert_pool_consistent()
    h.run_until_idle()
    assert h.pool.get_num_free_blocks() == h.pool.num_gpu_blocks - 1
    # The same prompt again starts from what was checkpointed and finishes.
    h.add_request("again", prompt, max_tokens=8)
    h.run_until_idle()
    assert h.pool.get_num_free_blocks() == h.pool.num_gpu_blocks - 1


@pytest.mark.parametrize("lane", sorted(LANES))
def test_preemption_drops_the_deferred_frees(lane: str) -> None:
    h = SchedHarness(**LANES[lane], queue_depth=2, check_pool=True,
                     num_blocks=SMALL_POOLS[lane])
    block = h.B
    # The long prompts do not fit the pool together.
    for i in range(5):
        h.add_request(f"r{i}", make_prompt(10 + i, 20 * block + 5), max_tokens=30)
    dropped = 0
    while h.busy():
        before = {
            rid: request.num_preemptions for rid, request in h.requests.items()
        }
        h.step()
        for rid, request in h.requests.items():
            if request.num_preemptions > before[rid] and request.status.name == (
                "PREEMPTED"
            ):
                dropped += 1
                for manager in _mamba_managers(h):
                    assert rid not in manager.stale_state_block_idxs
                    assert rid not in manager.last_state_block_idx
    assert dropped > 0
    h.run_until_idle()
    assert h.pool.get_num_free_blocks() == h.pool.num_gpu_blocks - 1
    for manager in _mamba_managers(h):
        assert not manager.stale_state_block_idxs


# --------------------------------------------------------------------------
# The allocator alone
# --------------------------------------------------------------------------
def _manager(universe: align_boot.Universe, num_spec: int):
    import torch

    pool = universe.block_pool.BlockPool(64, True, 4)
    spec = universe.kv_cache_interface.MambaSpec(
        block_size=4, shapes=((1,),), dtypes=(torch.float32,),
        mamba_cache_mode="align", num_speculative_blocks=num_spec,
    )
    manager = universe.allocator.MambaManager(
        spec, pool, enable_caching=True, kv_cache_group_id=0
    )
    return manager, pool


def _real(manager) -> list[int]:
    return [i for i, blk in enumerate(manager.req_to_blocks["r"]) if not blk.is_null]


@pytest.mark.parametrize("num_spec", [0, 3])
def test_allocator_frees_a_passed_state_block_once_processed(num_spec) -> None:
    """Chunks of three blocks, each scheduled while the previous one runs."""
    manager, pool = _manager(align_boot.load(), num_spec)
    spec_columns = lambda end: list(range(end, end + num_spec))  # noqa: E731

    def chunk(processed: int, scheduled: int) -> None:
        manager.remove_skipped_blocks("r", processed)
        manager.allocate_new_blocks("r", scheduled, scheduled)

    chunk(0, 12)
    assert _real(manager) == [2] + spec_columns(3)
    chunk(0, 24)  # the first chunk is in flight
    assert _real(manager) == [2, 5] + spec_columns(6)
    assert manager.stale_state_block_idxs == {}
    chunk(12, 36)  # column 2 feeds the chunk in flight: kept, but passed
    assert _real(manager) == [2, 5, 8] + spec_columns(9)
    assert manager.stale_state_block_idxs == {"r": [2]}
    chunk(24, 48)  # the chunk that read column 2 is done
    assert _real(manager) == [5, 8, 11] + spec_columns(12)
    assert manager.stale_state_block_idxs == {"r": [5]}
    # No further allocation: the remaining chunks complete one by one.
    manager.remove_skipped_blocks("r", 36)
    assert _real(manager) == [8, 11] + spec_columns(12)
    assert manager.stale_state_block_idxs == {"r": []}
    manager.remove_skipped_blocks("r", 48)
    assert _real(manager) == [11] + spec_columns(12)
    assert pool.get_num_free_blocks() == 63 - 1 - num_spec
    manager.free("r")
    assert manager.stale_state_block_idxs == {}
    assert pool.get_num_free_blocks() == 63


def test_allocator_keeps_upstream_tables_for_per_block_chunks() -> None:
    upstream = _upstream()
    for num_spec in (0, 3):
        tables = []
        for universe in (upstream, align_boot.load()):
            manager, pool = _manager(universe, num_spec)
            history = []
            for end in range(4, 44, 4):
                # One step in flight: the previous chunk is not processed.
                manager.remove_skipped_blocks("r", max(0, end - 8))
                manager.allocate_new_blocks("r", end, end)
                free = pool.free_block_queue.get_all_free_blocks()
                history.append(
                    (
                        [blk.block_id for blk in manager.req_to_blocks["r"]],
                        [blk.block_id for blk in free],
                    )
                )
            manager.free("r")
            history.append(pool.get_num_free_blocks())
            tables.append(history)
        assert tables[0] == tables[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
