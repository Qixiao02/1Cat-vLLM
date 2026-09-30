# SPDX-License-Identifier: Apache-2.0
"""Recurrent states stay exact under multi-block chunking.

Run (CPU only, no vLLM install needed; see ``align_boot.py``):

    python sx_tests/align-multiblock/test_state_oracle.py
    python -m pytest -q sx_tests/align-multiblock/test_state_oracle.py

The real ``Scheduler`` / ``AsyncScheduler`` with multi-block chunking on, a
worker model and the oracle of ``align_harness``: every forward starts from
the state after the request's own prefix, the worker only touches blocks the
request still holds, every Mamba block in the prefix cache holds exactly the
state its hash covers, reference counts match the block tables, and every
request produces the reference output.

Lanes: Flash-Next without speculative decoding (784-token state block),
MTP-shaped (Eagle tail rule, k = 4, 816), 27B-shaped (DFlash2, non-Eagle,
k = 7, draft sliding-window group, 4096 and 3296). Schedulers: synchronous,
async with one step in flight (the engine default) and with two.

Scenarios: identical resends, multi-turn conversations, shared system
prompts, aborts with steps in flight, preemption (pools too small for the
workload), and overlapping random arrivals with a random retention interval
and chunk cap.

Negative controls show that the oracle sees the failures it is there for: a
chunk crossing a boundary from mid-block, a state block freed while an
in-flight step still needs it, and a dropped boundary copy.

Two things are left out on purpose, because upstream d30469863 fails them
by itself (see ``test_upstream_*`` at the end, which pin that down):
* two steps in flight together with preemption: a frame that comes back
  after its request was preempted and resumed is applied to the resumed
  request, which then carries a token twice;
* random early output returns in the speculative lanes: a verify row that
  was given blocks past a state-block boundary and is scheduled again with
  fewer tokens trips an assertion in ``MambaManager.allocate_new_blocks``.
Expected: all pass.
"""

from __future__ import annotations

import os
import random
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402
from align_harness import (  # noqa: E402
    OracleError,
    SchedHarness,
    make_prompt,
    run_concurrent,
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
# Pools that hold about two of the workload's requests.
SMALL_POOLS = {"flashnext": 190, "mtp-eagle": 190, "27b-4096": 110, "27b-3296": 110}
DEPTHS = [1, 2, 3]


def _harness(lane: str, **overrides) -> SchedHarness:
    kwargs = dict(LANES[lane], check_pool=True)
    kwargs.update(overrides)
    h = SchedHarness(**kwargs)
    assert h.multiblock
    return h


def _eagle(lane: str) -> bool:
    return lane == "mtp-eagle"


def _assert_drained(h: SchedHarness) -> None:
    """Nothing is left behind once every request is done."""
    assert h.pool.get_num_free_blocks() == h.pool.num_gpu_blocks - 1
    for manager in h.manager.coordinator.single_type_managers:
        assert not any(manager.req_to_blocks.values())
        if type(manager).__name__ == "MambaManager":
            assert not manager.last_state_block_idx
            assert not manager.stale_state_block_idxs
            assert not manager._allocated_block_reqs


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
@pytest.mark.parametrize("queue_depth", DEPTHS)
@pytest.mark.parametrize("lane", sorted(LANES))
def test_identical_resends(lane: str, queue_depth: int) -> None:
    h = _harness(lane, queue_depth=queue_depth)
    block = h.B
    lengths = [block - 1, block, block + 1, 3 * block, 5 * block + 7, 23 * block + 300]
    for i, length in enumerate(lengths):
        prompt = make_prompt(10 + i, length)
        for attempt in range(3):
            h.add_request(f"p{i}.{attempt}", prompt, max_tokens=12)
            h.run_until_idle()
        first = h.prefill_chunk_ends(f"p{i}.0")
        reachable = (length - 1) // block * block - (block if _eagle(lane) else 0)
        hit = max(0, reachable)
        assert [h.hits[f"p{i}.{a}"] for a in range(3)] == [0, hit, hit], length
        # A resend computes exactly the chunks the first run had after the
        # hit, so a prefix-cache hit and a miss see the same chunk plan.
        for attempt in (1, 2):
            assert h.prefill_chunk_ends(f"p{i}.{attempt}") == [
                end for end in first if end > hit
            ]
    assert h.multiblock_chunks > 0
    _assert_drained(h)


@pytest.mark.parametrize("queue_depth", DEPTHS)
@pytest.mark.parametrize("lane", sorted(LANES))
def test_multi_turn(lane: str, queue_depth: int) -> None:
    h = _harness(lane, queue_depth=queue_depth)
    block = h.B
    conversation = make_prompt(20, 6 * block + 123)
    previous = 0
    for turn in range(4):
        rid = f"turn{turn}"
        # Long answers cross state-block boundaries while decoding.
        h.add_request(rid, conversation, max_tokens=block + 40 if turn % 2 else 30)
        h.run_until_idle()
        if turn:
            # At least the boundary the previous turn's resend would reach.
            reachable = (previous - 1) // block * block
            assert h.hits[rid] >= reachable - (block if _eagle(lane) else 0) > 0
        previous = len(conversation)
        conversation = list(h.requests[rid]._all_token_ids)
        conversation += make_prompt(30 + turn, 200)
    _assert_drained(h)


@pytest.mark.parametrize("queue_depth", DEPTHS)
@pytest.mark.parametrize("lane", sorted(LANES))
def test_shared_system_prompt(lane: str, queue_depth: int) -> None:
    h = _harness(lane, queue_depth=queue_depth)
    block = h.B
    system = make_prompt(40, 4 * block + block // 3)
    for i in range(4):
        h.add_request(f"s{i}", system + make_prompt(50 + i, block + 50), max_tokens=8)
        h.run_until_idle()
    hits = [h.hits[f"s{i}"] for i in range(4)]
    # The first request cannot know the junction. The second detects it and
    # ends a chunk there; every later one restores the state at it.
    junction = (4 - _eagle(lane)) * block
    assert hits == [0, 0, junction, junction]
    assert junction in h.prefill_chunk_ends("s1")
    _assert_drained(h)


@pytest.mark.parametrize("queue_depth", DEPTHS)
@pytest.mark.parametrize("lane", sorted(LANES))
def test_abort_with_steps_in_flight(lane: str, queue_depth: int) -> None:
    h = _harness(lane, queue_depth=queue_depth)
    block = h.B
    prompt = make_prompt(60, 30 * block + 17)
    h.add_request("gone", prompt, max_tokens=8)
    h.add_request("stays", make_prompt(61, 9 * block), max_tokens=40)
    for _ in range(3):
        h.step()
    assert 0 < h.requests["gone"].num_computed_tokens < len(prompt)
    h.abort("gone")
    h.run_until_idle()
    # Whatever the aborted request had checkpointed is still exact.
    h.add_request("again", prompt, max_tokens=8)
    h.run_until_idle()
    assert h.hits["again"] % block == 0
    _assert_drained(h)


def _random_harness(lane: str, seed: int, **overrides) -> SchedHarness:
    rng = random.Random(seed)
    base = LANES[lane]
    block = base["block_size"]
    env = dict(base.get("env", {}))
    env["SX_OPT_ALIGN_MAX_CHUNK_BLOCKS"] = str(rng.choice([0, 0, 0, 2, 4]))
    kwargs = dict(
        env=env,
        retention=rng.choice([0, 0, 4 * block, 10 * block]),
        max_num_seqs=rng.choice([2, 8, 24]),
        seed=seed,
    )
    if "budget" not in base:
        kwargs["budget"] = rng.choice([1000, 2048, 8192, 8192])
    kwargs.update(overrides)
    return _harness(lane, **kwargs)


@pytest.mark.parametrize("queue_depth", DEPTHS)
@pytest.mark.parametrize("lane", sorted(LANES))
def test_concurrent_arrivals(lane: str, queue_depth: int) -> None:
    multiblock_chunks = verify_rows = 0
    for seed in range(8):
        # Outputs coming back early at random is only safe to ask of upstream
        # without speculative decoding (see the module docstring).
        jitter = lane == "flashnext" and seed % 2 == 1
        h = _random_harness(lane, 100 * queue_depth + seed, queue_depth=queue_depth,
                            jitter=jitter, num_blocks=4096)
        run_concurrent(h, seed)
        assert sum(r.num_preemptions for r in h.requests.values()) == 0
        multiblock_chunks += h.multiblock_chunks
        verify_rows += h.verify_rows
        _assert_drained(h)
    assert multiblock_chunks > 0
    assert (verify_rows > 0) == bool(LANES[lane].get("num_spec"))


@pytest.mark.parametrize("queue_depth", [1, 2])
@pytest.mark.parametrize("lane", sorted(LANES))
def test_preemption(lane: str, queue_depth: int) -> None:
    preemptions = 0
    for seed in range(6):
        h = _random_harness(lane, 500 + 10 * queue_depth + seed,
                            queue_depth=queue_depth, num_blocks=SMALL_POOLS[lane])
        run_concurrent(h, 40 + seed, max_prompt_blocks=6.0)
        preemptions += sum(r.num_preemptions for r in h.requests.values())
        _assert_drained(h)
    assert preemptions > 0, "the pools were meant to be too small"


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------
def _unsafe_split(self, request, num_new_tokens, local=0, external=0):
    """Multi-block chunks from any start, aligned or not."""
    start = request.num_computed_tokens + local + external
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens
    block = self.mamba_state_block_size
    end = start + num_new_tokens
    if end < prefill_end and end // block * block > start:
        end = end // block * block
    return end - start


@pytest.mark.parametrize("queue_depth", [1, 2])
def test_oracle_detects_a_chunk_crossing_from_mid_block(queue_depth: int) -> None:
    # Dense retention caches the block the unaligned chunk leaves behind.
    h = SchedHarness(queue_depth=queue_depth, retention=None)
    h.sched._mamba_block_aligned_split = types.MethodType(_unsafe_split, h.sched)
    # r2 first gets the 192 tokens r1 leaves of the budget, then spans
    # several blocks from there.
    h.add_request("r1", make_prompt(1, 8000), max_tokens=4)
    h.add_request("r2", make_prompt(2, 8192), max_tokens=4)
    with pytest.raises(OracleError, match="never materialized"):
        h.run_until_idle()


@pytest.mark.parametrize("queue_depth", [2, 3])
@pytest.mark.parametrize("lane", ["flashnext", "27b-4096"])
def test_oracle_detects_a_state_block_freed_under_an_in_flight_step(
    lane: str, queue_depth: int
) -> None:
    def free_on_scheduled_tokens(h: SchedHarness) -> None:
        coordinator = h.manager.coordinator
        remove = coordinator.remove_skipped_blocks

        def eager(request_id, processed_computed_tokens, num_prompt_tokens=None):
            in_flight = h.requests[request_id].num_in_flight_tokens
            remove(request_id, processed_computed_tokens + in_flight, num_prompt_tokens)

        coordinator.remove_skipped_blocks = eager

    h = _harness(lane, queue_depth=queue_depth, check_pool=False)
    free_on_scheduled_tokens(h)
    h.add_request("r", make_prompt(1, 16 * h.B), max_tokens=20)
    with pytest.raises(OracleError, match="does not hold"):
        h.run_until_idle()
    # A worker that runs each step before the next one is scheduled cannot
    # see it; this is why the harness defers the worker by default.
    h = _harness(lane, queue_depth=queue_depth, eager_worker=True, check_pool=False)
    free_on_scheduled_tokens(h)
    h.add_request("r", make_prompt(1, 16 * h.B), max_tokens=20)
    h.run_until_idle()


@pytest.mark.parametrize("lane", ["mtp-eagle", "27b-4096"])
def test_oracle_detects_a_dropped_boundary_copy(lane: str) -> None:
    # Dense retention registers the boundary the verify tokens cross.
    h = SchedHarness(**LANES[lane], queue_depth=2, retention=None,
                     worker_bug="skip_postcopy", seed=1)
    h.add_request("r", make_prompt(1, h.B - 100), max_tokens=400)
    with pytest.raises(OracleError, match="never materialized"):
        h.run_until_idle()


def test_pool_check_detects_a_double_free() -> None:
    h = _harness("flashnext", queue_depth=2)
    h.add_request("r", make_prompt(1, 20_000), max_tokens=50)
    for _ in range(3):
        h.step()
    manager = h.manager.coordinator.single_type_managers[h.mamba_group_ids[0]]
    held = [blk for blk in manager.req_to_blocks["r"] if not blk.is_null]
    h.pool.free_blocks(held[:1])
    with pytest.raises(OracleError, match="holder"):
        h.assert_pool_consistent()


# --------------------------------------------------------------------------
# What upstream itself does not survive (hence not asked above)
# --------------------------------------------------------------------------
def _upstream() -> align_boot.Universe:
    if not align_boot.have_base():
        pytest.skip(f"base commit {align_boot.BASE_REV} is not readable")
    return align_boot.load(upstream=align_boot.PATCHED_MODULES)


def test_upstream_duplicates_a_token_with_two_steps_in_flight_and_preemption() -> None:
    upstream = _upstream()
    duplicated = 0
    for seed in range(12):
        h = SchedHarness(universe=upstream, layout="flashnext", block_size=784,
                         queue_depth=3, num_blocks=190, seed=seed,
                         max_num_seqs=random.Random(seed).choice([8, 24]))
        try:
            run_concurrent(h, 40 + seed, max_prompt_blocks=6.0)
        except Exception:  # noqa: S110 - what breaks first does not matter here
            pass
        for request in h.requests.values():
            outputs = list(request._output_token_ids)
            duplicated += any(a == b for a, b in zip(outputs, outputs[1:]))
    assert duplicated > 0


def test_upstream_asserts_when_a_verify_row_needs_fewer_blocks() -> None:
    # Which seeds trip it depends on the arrival pattern, so scan a range
    # instead of pinning one.
    upstream = _upstream()
    tripped = 0
    for lane in ("27b-4096", "27b-3296", "mtp-eagle"):
        for seed in range(40):
            kwargs = dict(LANES[lane], universe=upstream, env=None)
            h = SchedHarness(**kwargs, queue_depth=2, jitter=True,
                             num_blocks=4096, seed=seed)
            try:
                run_concurrent(h, seed)
            except AssertionError as exc:
                tripped += "num_required_blocks" in str(exc)
    assert tripped > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
