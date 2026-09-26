# SPDX-License-Identifier: Apache-2.0
"""CPU tests for SX_OPT_ALIGN_MULTIBLOCK (design_2 [MA-1] / design_3 [P1]).

No GPU needed. Run inside the deployed image AFTER the overlay is installed,
from a directory where ``import vllm`` resolves to the installed package:

    cd /tmp && /opt/venv/bin/python -m pytest -q \
        <repo>/sx_tests/align-chunking/test_align_multiblock_scheduler.py

What is asserted:
* chunk sequences for prompt lengths {450, 784, 785, 2048, 8192, 32768} at a
  token budget of 8192, for the old per-block rule and the new policy
  (tail checkpoint on/off, block cap), with and without prefix hits;
* with SX_OPT_ALIGN_MULTIBLOCK off, and for ANY chunk starting mid-block with
  it on, the split is identical to the verbatim git-baseline function
  (200k randomized cases);
* scheduler-level runs with the REAL KVCacheManager (hybrid coordinator with a
  Qwen3.8-like layout: QSA main + compressed + ring, 3 GDN groups, PLE state)
  and a recurrent-state oracle that mirrors the MRV2 pre-copy/forward: every
  forward starts from exactly the state of its own prefix and every Mamba block
  registered in the prefix cache holds exactly the state its hash covers;
  interior blocks of a multi-block chunk are null;
* prefix-hit lengths for a repeated prompt, a multi-turn follow-up and a shared
  system prompt (with/without the shared checkpoint), and the gating of the
  policy in ``Scheduler._sx_init_mamba_align_policy``.
Expected result: all tests pass.
"""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from align_sim import AlignOracleError, AlignSim, RealBackend  # noqa: E402

B = 784
BUDGET = 8192


def _baseline_mamba_block_aligned_split(
    self,
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Verbatim copy of Scheduler._mamba_block_aligned_split at git 71c1822."""
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


@pytest.fixture(scope="module")
def backend() -> RealBackend:
    import vllm

    print("vllm package:", vllm.__file__)
    return RealBackend()


def _shim(multiblock: bool, tail: bool = True, max_blocks: int = 0,
          eagle: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        mamba_state_block_size=B,
        use_eagle=eagle,
        _sx_align_multiblock=multiblock,
        _sx_align_tail=tail,
        _sx_align_max_blocks=max_blocks,
    )


def _split_fn():
    from vllm.v1.core.sched.scheduler import Scheduler

    return Scheduler._mamba_block_aligned_split


def _chunk_ends(split, shim, prompt_len: int, start: int = 0,
                budget: int = BUDGET) -> list[int]:
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_prompt_tokens=prompt_len,
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


def _old_ends(prompt_len: int, start: int = 0) -> list[int]:
    ends = list(range(start + B, prompt_len, B))
    if not ends or ends[-1] != prompt_len:
        ends.append(prompt_len)
    return ends


# expected chunk ends at budget 8192, no prefix hit
EXPECTED = {
    #        new(tail)                                    new(no tail)
    450: ([450], [450]),
    784: ([784], [784]),
    785: ([784, 785], [785]),
    2048: ([1568, 2048], [2048]),
    8192: ([7840, 8192], [8192]),
    32768: (
        [7840, 15680, 23520, 31360, 32144, 32768],
        [7840, 15680, 23520, 31360, 32768],
    ),
}


@pytest.mark.parametrize("prompt_len", sorted(EXPECTED))
def test_chunk_sequences_budget_8192(prompt_len: int) -> None:
    split = _split_fn()
    assert _chunk_ends(split, _shim(False), prompt_len) == _old_ends(prompt_len)
    assert _chunk_ends(_baseline_mamba_block_aligned_split, _shim(False),
                       prompt_len) == _old_ends(prompt_len)
    tail, no_tail = EXPECTED[prompt_len]
    assert _chunk_ends(split, _shim(True, tail=True), prompt_len) == tail
    assert _chunk_ends(split, _shim(True, tail=False), prompt_len) == no_tail


@pytest.mark.parametrize(
    "prompt_len,expected",
    [
        (8192, [3136, 6272, 7840, 8192]),
        (32768, [3136, 6272, 9408, 12544, 15680, 18816, 21952, 25088, 28224,
                 31360, 32144, 32768]),
        (2048, [1568, 2048]),
    ],
)
def test_chunk_block_cap(prompt_len: int, expected: list[int]) -> None:
    split = _split_fn()
    assert _chunk_ends(split, _shim(True, max_blocks=4), prompt_len) == expected


@pytest.mark.parametrize(
    "prompt_len,hit,expected",
    [
        (8192, 7840, [8192]),
        (8192, 784, [7840, 8192]),  # rest fits the budget; tail stop at 7840
        (32768, 32144, [32768]),
        (32768, 1568, [9408, 17248, 25088, 32144, 32768]),
        (2048, 1568, [2048]),
        (2048, 784, [1568, 2048]),
        (785, 784, [785]),
    ],
)
def test_chunk_sequences_after_prefix_hit(prompt_len, hit, expected) -> None:
    split = _split_fn()
    assert _chunk_ends(split, _shim(True), prompt_len, start=hit) == expected
    assert _chunk_ends(split, _shim(False), prompt_len, start=hit) == _old_ends(
        prompt_len, hit
    )


def test_split_matches_baseline_when_disabled_or_unaligned() -> None:
    split = _split_fn()
    rng = random.Random(0)
    checked = changed = 0
    for _ in range(200_000):
        prompt = rng.randrange(1, 40_000)
        outputs = rng.choice([0, 0, 0, rng.randrange(1, 3000)])
        start = rng.randrange(0, prompt + outputs)
        if rng.random() < 0.5:
            start = start // B * B
        request = SimpleNamespace(
            num_computed_tokens=start,
            num_prompt_tokens=prompt,
            num_tokens=prompt + outputs,
        )
        remaining = request.num_tokens - start
        if remaining <= 0:
            continue
        n = rng.randrange(1, min(remaining, 8192) + 1)
        eagle = rng.random() < 0.1
        old = _baseline_mamba_block_aligned_split(_shim(False, eagle=eagle), request, n)
        assert split(_shim(False, eagle=eagle), request, n) == old
        on = split(
            _shim(True, rng.random() < 0.5, rng.choice([0, 2, 4]), eagle), request, n
        )
        if start % B:
            assert on == old, (prompt, outputs, start, n, old, on)
        else:
            prefill_end = max(prompt, request.num_tokens - 1)
            assert old <= on <= n
            if start < prefill_end and on != n:
                assert (start + on) % B == 0
            changed += on != old
        checked += 1
    assert checked > 150_000 and changed > 10_000


# --------------------------------------------------------------------------
# Scheduler-level with the real KVCacheManager + recurrent-state oracle
# --------------------------------------------------------------------------
def _prompt(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(10, 30_000) for _ in range(n)]


@pytest.mark.parametrize("prompt_len", sorted(EXPECTED))
@pytest.mark.parametrize("policy", ["old", "tail", "notail"])
def test_repeat_prompt_bookkeeping(backend, prompt_len: int, policy: str) -> None:
    sim = AlignSim(backend, multiblock=policy != "old", tail=policy != "notail",
                   shared=True)
    prompt = _prompt(prompt_len, prompt_len)
    sim.add_request("first", prompt, max_tokens=4)
    sim.run_until_idle()
    sim.add_request("second", prompt, max_tokens=4)
    sim.run_until_idle()

    first = sim.prefill_chunk_ends("first")
    if policy == "old":
        assert first == _old_ends(prompt_len)
    else:
        assert first == EXPECTED[prompt_len][0 if policy == "tail" else 1]
    last_cache_position = (prompt_len - 1) // B * B
    # The second request hits at the deepest checkpoint <= n - 1: the tail
    # checkpoint when it exists, else the deepest chunk end.
    expected_hit = max(
        [0] + [e for e in first if e % B == 0 and e <= last_cache_position]
    )
    assert sim.hits["second"] == expected_hit
    if policy != "notail":
        assert sim.hits["second"] == last_cache_position


def test_concurrent_prompts_share_the_budget(backend) -> None:
    sim = AlignSim(backend, multiblock=True, shared=True)
    sim.add_request("r1", _prompt(1, 8192))
    sim.add_request("r2", _prompt(2, 8192))
    sim.run_until_idle()
    assert sim.prefill_chunk_ends("r1") == [7840, 8192]
    # r2 gets the 352-token remainder (unaligned), finishes its first block
    # with the old per-block rule, then continues with multi-block chunks.
    assert sim.prefill_chunk_ends("r2") == [352, 784, 7840, 8192]
    assert sim.multiblock_steps >= 2


def test_shared_system_prompt_checkpoint(backend) -> None:
    system = _prompt(99, 1000)
    results = {}
    for name, kwargs in {
        "old": dict(multiblock=False),
        "new_no_shared": dict(multiblock=True, shared=False),
        "new_shared": dict(multiblock=True, shared=True),
    }.items():
        sim = AlignSim(backend, **kwargs)
        for i in range(4):
            sim.add_request(f"s{i}", system + _prompt(100 + i, 2000))
            sim.run_until_idle()
        results[name] = [sim.hits[f"s{i}"] for i in range(4)]
        if name == "new_shared":
            # s1 stops at the 784 boundary of the shared prefix (checkpoint),
            # s2/s3 restore it and start there.
            assert sim.prefill_chunk_ends("s1") == [784, 2352, 3000]
            assert sim.prefill_chunk_ends("s2") == [2352, 3000]
    assert results["old"] == [0, 784, 784, 784]
    assert results["new_no_shared"] == [0, 0, 0, 0]
    assert results["new_shared"] == [0, 0, 784, 784]


@pytest.mark.parametrize("multiblock", [False, True])
def test_multi_turn_follow_up_hits_decode_checkpoint(backend, multiblock) -> None:
    sim = AlignSim(backend, multiblock=multiblock, shared=True)
    sim.add_request("t1", _prompt(7, 2048), max_tokens=400)
    sim.run_until_idle()
    follow_up = list(sim.requests["t1"]._all_token_ids) + _prompt(8, 300)
    sim.add_request("t2", follow_up, max_tokens=4)
    sim.run_until_idle()
    # 2048 prompt + 399 computed outputs cross the 2352 boundary in decode.
    assert sim.hits["t2"] == 2352


@pytest.mark.parametrize("hash_block_size", [784, 16])
@pytest.mark.parametrize("lag", [False, True])
def test_partial_prefix_hits(backend, hash_block_size: int, lag: bool) -> None:
    sim = AlignSim(backend, multiblock=True, shared=True,
                   hash_block_size=hash_block_size, lag_in_flight=lag)
    base = _prompt(5, 9000)
    sim.add_request("h1", base)
    sim.add_request("h2", base[:5000] + _prompt(6, 3000))
    sim.run_until_idle()
    sim.add_request("h3", base)
    sim.add_request("h4", base[:5000] + _prompt(7, 100))
    sim.run_until_idle()
    assert sim.prefill_chunk_ends("h1") == [7840, 8624, 9000]
    assert sim.hits["h3"] == 8624
    # h2 checkpointed the 4704 boundary of the 5000-token shared prefix.
    assert 4704 in sim.prefill_chunk_ends("h2")
    assert sim.hits["h4"] == 4704


@pytest.mark.parametrize("hash_block_size", [784, 16])
@pytest.mark.parametrize("lag", [False, True])
def test_preempted_prefill_resumes_from_its_multiblock_checkpoint(
    backend, hash_block_size: int, lag: bool
) -> None:
    """Recompute preemption after two multi-block chunks: the request's own
    chunk-end checkpoint (a real block amid null-padded columns) must serve
    the re-admission, and the replay must start from exactly that state."""
    sim = AlignSim(backend, multiblock=True, shared=True,
                   hash_block_size=hash_block_size, lag_in_flight=lag)
    sim.add_request("p1", _prompt(11, 20000), max_tokens=4)
    sim.step()
    sim.step()
    assert sim.prefill_chunk_ends("p1") == [7840, 15680]
    sim.preempt("p1")
    sim.run_until_idle()  # the oracle raises on any wrong state
    assert sim.hits["p1"] == 15680
    assert sim.prefill_chunk_ends("p1") == [7840, 15680, 19600, 20000]
    assert "p1" in sim.finished


@pytest.mark.parametrize("hash_block_size", [784, 16])
@pytest.mark.parametrize("lag", [False, True])
def test_preempted_decode_replays_outputs_from_decode_checkpoint(
    backend, hash_block_size: int, lag: bool
) -> None:
    """Preempt during decode after the 2352 boundary was crossed: the resumed
    request (prompt + outputs, prefill_end = num_tokens - 1) hits the decode
    checkpoint and replays the rest in one chunk from a block boundary."""
    sim = AlignSim(backend, multiblock=True, shared=True,
                   hash_block_size=hash_block_size, lag_in_flight=lag)
    sim.add_request("d1", _prompt(12, 2048), max_tokens=400)
    while sim.num_outputs["d1"] < 350:
        sim.step()
    num_tokens = sim.requests["d1"].num_tokens
    assert num_tokens > 2352 + 1
    sim.preempt("d1")
    sim.run_until_idle()
    assert sim.hits["d1"] == 2352
    assert (2352, num_tokens) in sim.chunks["d1"]
    assert sim.num_outputs["d1"] == 400


def test_oracle_detects_multiblock_from_unaligned_start(backend) -> None:
    """Negative control: the invariant the unaligned-start rule protects."""

    def unsafe_split(self, request, num_new_tokens, a=0, b=0):
        start = request.num_computed_tokens + a + b
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens
        end = start + num_new_tokens
        if end < prefill_end and end // B * B > start:
            end = end // B * B
        last = request.num_tokens - 1
        tail = last - last % B
        end = min((s for s in (tail,) if start < s < end), default=end)
        return end - start

    sim = AlignSim(backend, multiblock=True, split_fn=unsafe_split)
    sim.add_request("r1", _prompt(1, 8192))
    sim.add_request("r2", _prompt(2, 8192))
    with pytest.raises(AlignOracleError, match="never checkpointed"):
        sim.run_until_idle()


@pytest.mark.parametrize("seed", range(24))
def test_randomized_workloads_keep_cache_invariants(backend, seed: int) -> None:
    rng = random.Random(seed)
    sim = AlignSim(
        backend,
        multiblock=rng.random() < 0.85,
        tail=rng.random() < 0.7,
        max_blocks=rng.choice([0, 0, 2, 4]),
        shared=rng.random() < 0.6,
        budget=rng.choice([300, 1000, 2048, 8192, 8192]),
        lag_in_flight=rng.random() < 0.5,
        hash_block_size=rng.choice([784, 16]),
        max_num_seqs=rng.choice([2, 8, 24]),
    )
    bases = [_prompt(1000 + k, 6000) for k in range(3)]
    n_req = 0
    for _ in range(6):
        for _ in range(rng.randrange(1, 6)):
            share = rng.choice([0, 500, 784, 1568, 2000, 3000])
            prompt = rng.choice(bases)[:share] + _prompt(
                rng.randrange(1 << 30), rng.randrange(1, 4000)
            )
            sim.add_request(f"f{n_req}", prompt, max_tokens=rng.randrange(1, 900))
            n_req += 1
        for _ in range(rng.randrange(1, 30)):
            if sim.waiting or sim.running:
                sim.step()
    sim.run_until_idle()  # the oracle raises on any violation
    assert len(sim.finished) == n_req


@pytest.mark.parametrize("seed", range(12))
def test_randomized_workloads_with_preemption(backend, seed: int) -> None:
    rng = random.Random(10_000 + seed)
    sim = AlignSim(
        backend,
        multiblock=True,
        tail=rng.random() < 0.7,
        max_blocks=rng.choice([0, 0, 2, 4]),
        shared=rng.random() < 0.6,
        budget=rng.choice([1000, 2048, 8192, 8192]),
        lag_in_flight=rng.random() < 0.5,
        hash_block_size=rng.choice([784, 16]),
        max_num_seqs=rng.choice([2, 8, 24]),
    )
    bases = [_prompt(2000 + k, 9000) for k in range(2)]
    n_req = 0
    preemptions = 0
    for _ in range(8):
        for _ in range(rng.randrange(1, 4)):
            share = rng.choice([0, 784, 2000, 5000])
            prompt = rng.choice(bases)[:share] + _prompt(
                rng.randrange(1 << 30), rng.randrange(1, 6000)
            )
            sim.add_request(f"q{n_req}", prompt, max_tokens=rng.randrange(1, 900))
            n_req += 1
        for _ in range(rng.randrange(1, 20)):
            if sim.waiting or sim.running:
                sim.step()
            if sim.running and rng.random() < 0.15:
                sim.preempt(rng.choice(sim.running).request_id)
                preemptions += 1
    sim.run_until_idle()  # the oracle raises on any violation
    assert len(sim.finished) == n_req
    assert preemptions > 0


# --------------------------------------------------------------------------
# Policy gating in Scheduler.__init__
# --------------------------------------------------------------------------
def _policy_scheduler(*, align=True, spec=None, eagle=False, num_spec=0,
                      connector=None, prefix=True):
    from vllm.v1.core.sched.scheduler import Scheduler

    s = object.__new__(Scheduler)
    s.need_mamba_block_aligned_split = align
    s.use_eagle = eagle
    s.num_spec_tokens = num_spec
    s.connector = connector
    s.cache_config = SimpleNamespace(enable_prefix_caching=prefix, block_size=784)
    s.mamba_state_block_size = 784
    cfg = SimpleNamespace(speculative_config=spec)
    return s, cfg


@pytest.mark.parametrize(
    "env,contract,kwargs,expected",
    [
        ({}, True, {}, (True, True, 0, True)),
        ({"SX_OPT_ALIGN_MULTIBLOCK": "0"}, True, {}, (False, True, 0, False)),
        ({}, False, {}, (False, True, 0, False)),
        ({"SX_OPT_ALIGN_MULTIBLOCK": "force"}, False, {}, (True, True, 0, True)),
        ({}, True, {"spec": object()}, (False, True, 0, False)),
        ({"SX_OPT_ALIGN_MULTIBLOCK": "force"}, True, {"eagle": True},
         (False, True, 0, False)),
        ({}, True, {"connector": object()}, (False, True, 0, False)),
        ({}, True, {"align": False}, (False, True, 0, False)),
        (
            {"SX_OPT_ALIGN_TAIL_CHECKPOINT": "0",
             "SX_OPT_ALIGN_MAX_CHUNK_BLOCKS": "4",
             "SX_OPT_ALIGN_SHARED_CHECKPOINT": "0"},
            True, {}, (True, False, 4, False),
        ),
    ],
)
def test_policy_gating(monkeypatch, env, contract, kwargs, expected) -> None:
    from vllm.v1.core.sched import scheduler as sched_mod

    for name in ("SX_OPT_ALIGN_MULTIBLOCK", "SX_OPT_ALIGN_TAIL_CHECKPOINT",
                 "SX_OPT_ALIGN_MAX_CHUNK_BLOCKS", "SX_OPT_ALIGN_SHARED_CHECKPOINT"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(
        sched_mod, "_sx_is_sm70_qwen38_align_contract", lambda cfg: contract
    )
    s, cfg = _policy_scheduler(**kwargs)
    s._sx_init_mamba_align_policy(cfg)
    assert (
        s._sx_align_multiblock,
        s._sx_align_tail,
        s._sx_align_max_blocks,
        s._sx_align_shared,
    ) == expected


def test_contract_helper_matches_deployment_shape(monkeypatch) -> None:
    import torch

    from vllm.config import vllm as config_vllm
    from vllm.v1.core.sched import scheduler as sched_mod

    text = SimpleNamespace(
        hidden_size=2560, num_hidden_layers=48, num_experts=512,
        num_experts_per_tok=10, moe_intermediate_size=640, hc_count=4,
        hc_lowrank=320, num_attention_heads=24, num_key_value_heads=2,
        indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
    )
    model = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        multimodal_config=SimpleNamespace(language_model_only=True),
        dtype=torch.float16, hf_text_config=text,
    )
    parallel = SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1)
    cfg = SimpleNamespace(model_config=model, speculative_config=None,
                          parallel_config=parallel)
    monkeypatch.setattr(config_vllm, "_any_participating_device_is_capability",
                        lambda c, cap: cap == (7, 0))
    assert sched_mod._sx_is_sm70_qwen38_align_contract(cfg)
    monkeypatch.setattr(config_vllm, "_any_participating_device_is_capability",
                        lambda c, cap: False)
    assert not sched_mod._sx_is_sm70_qwen38_align_contract(cfg)
    parallel.tensor_parallel_size = 2
    monkeypatch.setattr(config_vllm, "_any_participating_device_is_capability",
                        lambda c, cap: True)
    assert not sched_mod._sx_is_sm70_qwen38_align_contract(cfg)
