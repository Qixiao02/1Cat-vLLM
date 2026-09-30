# SPDX-License-Identifier: Apache-2.0
"""Admission of align multi-block prefill chunking (``Scheduler.__init__``).

Run (CPU only, no vLLM install needed; see ``align_boot.py``):

    python sx_tests/align-multiblock/test_gating.py
    python -m pytest -q sx_tests/align-multiblock/test_gating.py

Every case constructs the real ``Scheduler`` with a stand-in config and
reads the policy it resolved and the lines it logged. The model-shape
contract is the real ``_is_sm70_qwen38_decode_compile_contract``; only the
device capability is faked.

Asserted:
* the default lanes: no speculative decoding is admitted, every speculative
  method is refused unless ``SX_OPT_ALIGN_MULTIBLOCK_SPEC`` lists it;
* each refusal, in admission order, with its logged reason: switch off, KV
  connector, lane, Model Runner V1, hardware / model contract, dense
  retention;
* ``SX_OPT_ALIGN_MULTIBLOCK=force`` skips the contract and nothing else;
* dense retention is decided exactly as ``MambaManager.reachable_block_mask``
  decides it, for every interval / alignment combination tried;
* a failing contract check (renamed config helper, broken config object)
  logs a warning and keeps per-block chunking;
* the two startup warnings of the DFlash lane, and that they stay silent
  for the documented 27B flags;
* outside ``align`` mode nothing is logged.
Expected: all pass.
"""

from __future__ import annotations

import logging
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402
from align_harness import SchedHarness  # noqa: E402

DISABLED = (
    "SX align multi-block prefill chunking disabled ({}); every Mamba state "
    "block boundary ends a prefill chunk."
)
ENABLED = (
    "SX align multi-block prefill chunking enabled ({} lane): state block {}, "
    "hit alignment {}, retention={}, max_chunk_blocks={}, single-request "
    "chunk={} tokens."
)
FAILED = "SX align multi-block: contract check failed; keeping per-block chunking."

# Flash-Next: Qwen3.8 contract model, 784-token state block, budget 8192.
NO_SPEC = dict(layout="flashnext", block_size=784, num_blocks=64)
MTP = dict(layout="flashnext", block_size=816, num_blocks=64, num_spec=4,
           method="mtp")
# 27B: Qwen3.5 hybrid with DFlash2, k=7, draft sliding-window group.
DFLASH = dict(layout="27b", block_size=4096, num_blocks=64, num_spec=7,
              method="dflash", budget=16384, threshold=8192)


def _build(base: dict, env: dict[str, str] | None = None, **overrides):
    return SchedHarness(**{**base, **overrides}, env=env)


def _assert_refused(h: SchedHarness, reason: str) -> None:
    assert not h.multiblock
    assert h.sched._sx_align_lane is None
    assert (h.sched._sx_align_retention, h.sched._sx_align_max_blocks) == (0, 0)
    assert h.messages() == [DISABLED.format(reason)]


def _lane_reason(method: str) -> str:
    return (
        "speculative decoding keeps per-block checkpoints (method "
        f"{method} is not admitted by SX_OPT_ALIGN_MULTIBLOCK_SPEC)"
    )


def _contract_reason(lane: str) -> str:
    return f"outside the SM70 contract of the {lane} lane"


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------
def test_no_spec_lane_is_on_by_default() -> None:
    h = _build(NO_SPEC)
    assert h.multiblock
    assert h.sched._sx_align_lane == "no-spec"
    assert h.sched._sx_align_hit_alignment == 784
    assert h.messages() == [
        ENABLED.format("no-spec", 784, 784, "replay/shared only", "budget", 7840)
    ]


@pytest.mark.parametrize("base,method", [(DFLASH, "dflash"), (MTP, "mtp")])
def test_speculative_lanes_are_off_by_default(base, method) -> None:
    _assert_refused(_build(base), _lane_reason(method))
    # `force` is about the model contract; it does not admit a lane.
    _assert_refused(
        _build(base, {"SX_OPT_ALIGN_MULTIBLOCK": "force"}), _lane_reason(method)
    )


@pytest.mark.parametrize(
    "base,spec,lane",
    [
        (DFLASH, "dflash", "dflash"),
        (DFLASH, "dflash,mtp", "dflash"),
        (DFLASH, " DFlash , mtp ", "dflash"),
        (DFLASH, "mtp", None),
        (DFLASH, "0", None),
        (DFLASH, "", None),
        (MTP, "dflash,mtp", "mtp"),
        (MTP, "mtp", "mtp"),
        (MTP, "dflash", None),
    ],
)
def test_spec_switch_lists_the_admitted_methods(base, spec, lane) -> None:
    h = _build(base, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": spec})
    assert h.sched._sx_align_lane == lane
    assert h.multiblock == (lane is not None)
    if lane is None:
        _assert_refused(h, _lane_reason(base["method"]))


def test_dflash_lane_log_line() -> None:
    h = _build(DFLASH, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash"})
    assert h.messages() == [
        ENABLED.format("dflash", 4096, 4096, "replay/shared only", "budget", 8192)
    ]


@pytest.mark.parametrize(
    "overrides,method",
    [
        # Other speculative methods have no lane, whatever is listed.
        (dict(method="eagle3"), "eagle3"),
        (dict(method="dflash_ddtree"), "dflash_ddtree"),
        (dict(method="draft_model"), "draft_model"),
        # Not the Qwen4Exp MTP head.
        (dict(native_mtp=False), "mtp"),
        (dict(parallel_drafting=True), "mtp"),
    ],
)
def test_other_speculative_methods_are_refused(overrides, method) -> None:
    env = {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash,mtp"}
    _assert_refused(_build(MTP, env, **overrides), _lane_reason(method))
    env["SX_OPT_ALIGN_MULTIBLOCK"] = "force"
    _assert_refused(_build(MTP, env, **overrides), _lane_reason(method))


def test_unknown_lane_names_are_reported() -> None:
    h = _build(MTP, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "1,eagle3"})
    assert h.messages(logging.WARNING) == [
        "SX_OPT_ALIGN_MULTIBLOCK_SPEC: ignoring unknown lane(s) 1, eagle3; "
        "known lanes are dflash, mtp."
    ]
    assert not h.multiblock


# --------------------------------------------------------------------------
# Refusals, in admission order
# --------------------------------------------------------------------------
def test_switch_off() -> None:
    _assert_refused(
        _build(NO_SPEC, {"SX_OPT_ALIGN_MULTIBLOCK": "0"}), "SX_OPT_ALIGN_MULTIBLOCK=0"
    )
    # Checked before the connector and the lane.
    _assert_refused(
        _build(DFLASH, {"SX_OPT_ALIGN_MULTIBLOCK": "0"}, connector=True),
        "SX_OPT_ALIGN_MULTIBLOCK=0",
    )


@pytest.mark.parametrize("mode", ["1", "force"])
def test_kv_connector(mode) -> None:
    reason = "KV connector keeps per-block checkpoints"
    env = {"SX_OPT_ALIGN_MULTIBLOCK": mode}
    _assert_refused(_build(NO_SPEC, env, connector=True), reason)
    # Checked before the lane.
    _assert_refused(_build(DFLASH, env, connector=True), reason)


@pytest.mark.parametrize("mode", ["1", "force"])
def test_model_runner_v1(mode) -> None:
    env = {"SX_OPT_ALIGN_MULTIBLOCK": mode}
    reason = "Model Runner V2 is required"
    _assert_refused(_build(NO_SPEC, env, runner_v2=False), reason)
    # Checked after the lane.
    _assert_refused(_build(DFLASH, env, runner_v2=False), _lane_reason("dflash"))


CONTRACT_CASES = [
    # (base, env, overrides, lane)
    (NO_SPEC, {}, dict(sm70=False), "no-spec"),
    (NO_SPEC, {}, dict(tp=2), "no-spec"),
    (NO_SPEC, {}, dict(model="Qwen3_5ForConditionalGeneration"), "no-spec"),
    (MTP, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "mtp"}, dict(sm70=False), "mtp"),
    # The decode compile contract admits MTP with k = 4 only.
    (MTP, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "mtp"}, dict(num_spec=2), "mtp"),
    (DFLASH, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash"}, dict(sm70=False), "dflash"),
    (DFLASH, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash"}, dict(model="qwen38"),
     "dflash"),
]


@pytest.mark.parametrize("base,env,overrides,lane", CONTRACT_CASES)
def test_contract_and_force(base, env, overrides, lane) -> None:
    _assert_refused(_build(base, env, **overrides), _contract_reason(lane))
    forced = _build(base, {**env, "SX_OPT_ALIGN_MULTIBLOCK": "force"}, **overrides)
    assert forced.multiblock
    assert forced.sched._sx_align_lane == lane


def test_contract_admits_the_production_shapes() -> None:
    assert _build(NO_SPEC).sched._sx_align_lane == "no-spec"
    env = {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash,mtp"}
    assert _build(MTP, env).sched._sx_align_lane == "mtp"
    assert _build(DFLASH, env).sched._sx_align_lane == "dflash"
    # The DFlash lane takes any Qwen3.5-family target.
    assert _build(DFLASH, env, model="Qwen3_5ForCausalLM").multiblock


# --------------------------------------------------------------------------
# Dense retention
# --------------------------------------------------------------------------
def _dense_by_mask(h: SchedHarness) -> bool:
    """The rule, evaluated here from the layout with the real mask function."""
    specs = [group.kv_cache_spec for group in h.kv_cache_config.kv_cache_groups]
    alignment = math.lcm(*(spec.block_size for spec in specs if spec.prefix_cacheable))
    mask = h.universe.allocator.MambaManager.reachable_block_mask
    interval = h.vllm_config.cache_config.prefix_cache_retention_interval
    return any(
        mask(0, 1, alignment, spec, interval, ()) is None
        for spec in specs
        if type(spec).__name__ == "MambaSpec"
    )


@pytest.mark.parametrize("mode", ["1", "force"])
@pytest.mark.parametrize("attn_block", [None, 16, 392, 1568])
@pytest.mark.parametrize("retention", [None, 0, 1568, 3136, 7840, 78400])
def test_dense_retention_follows_reachable_block_mask(retention, attn_block,
                                                      mode) -> None:
    alignment = math.lcm(784, attn_block or 784)
    h = _build(NO_SPEC, {"SX_OPT_ALIGN_MULTIBLOCK": mode}, retention=retention,
               attn_block_size=attn_block, num_blocks=4096, hash_block_size=8)
    dense = _dense_by_mask(h)
    assert h.multiblock == (not dense)
    # Dense: no interval, a hit alignment other than the state block, or an
    # interval of at most one block.
    assert dense == (retention is None or alignment != 784 or retention == 784)
    if dense:
        _assert_refused(
            h,
            "dense Mamba prefix-cache retention "
            f"(prefix_cache_retention_interval={retention}, hit alignment "
            f"{alignment}, state block 784)",
        )
    else:
        assert h.sched._sx_align_retention == (retention or 0)
        assert h.sched._sx_align_hit_alignment == 784


def test_retention_of_one_block_is_dense() -> None:
    h = _build(NO_SPEC, retention=784)
    assert _dense_by_mask(h)
    _assert_refused(
        h,
        "dense Mamba prefix-cache retention (prefix_cache_retention_interval=784, "
        "hit alignment 784, state block 784)",
    )


def test_positive_retention_log_line_and_chunk() -> None:
    h = _build(NO_SPEC, retention=3136)
    assert h.messages() == [ENABLED.format("no-spec", 784, 784, 3136, "budget", 3136)]
    h = _build(NO_SPEC, retention=7840)
    assert h.messages() == [ENABLED.format("no-spec", 784, 784, 7840, "budget", 7840)]


# --------------------------------------------------------------------------
# Failing contract check
# --------------------------------------------------------------------------
def test_renamed_config_helper_keeps_per_block_chunking() -> None:
    universe = align_boot.load()
    name = "_is_sm70_qwen38_decode_compile_contract"
    helper = universe.config_vllm.__dict__.pop(name)
    try:
        h = _build(NO_SPEC, universe=universe)
    finally:
        setattr(universe.config_vllm, name, helper)
    assert not h.multiblock
    assert h.messages() == [FAILED]
    assert h.messages(logging.WARNING) == [FAILED]
    # The lane works again once the helper is back.
    assert _build(NO_SPEC, universe=universe).multiblock


def test_broken_speculative_config_keeps_per_block_chunking() -> None:
    from align_harness import make_spec_config

    h = None
    original = make_spec_config

    def broken(*args, **kwargs):
        spec = original(*args, **kwargs)
        del spec.use_qwen4_exp_mtp
        return spec

    import align_harness

    align_harness.make_spec_config = broken
    try:
        h = _build(MTP, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "mtp"})
    finally:
        align_harness.make_spec_config = original
    assert not h.multiblock
    assert h.messages(logging.WARNING) == [FAILED]


# --------------------------------------------------------------------------
# Other switches and warnings
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,blocks,chunk", [("3", 3, 2352), ("0", 0, 7840), ("abc", 0, 7840),
                           ("-2", 0, 7840), (" 12 ", 12, 7840)]
)
def test_max_chunk_blocks(value, blocks, chunk) -> None:
    h = _build(NO_SPEC, {"SX_OPT_ALIGN_MAX_CHUNK_BLOCKS": value})
    assert h.sched._sx_align_max_blocks == blocks
    assert h.messages() == [
        ENABLED.format("no-spec", 784, 784, "replay/shared only", blocks or "budget",
                       chunk)
    ]


WINDOW = "is outside the 8000-8192 token window"
WHOLE_BUDGET = "takes the whole step budget"


@pytest.mark.parametrize(
    "overrides,chunk,window,whole_budget",
    [
        # The documented 27B flags: block 2048 (grid 4096), threshold 8192,
        # budget 16384.
        (dict(), 8192, False, False),
        (dict(block_size=4000, threshold=8000), 8000, False, False),
        # Today's grid: chunks of two 3296 blocks miss the window.
        (dict(block_size=3296, threshold=8000), 6592, True, False),
        # Threshold 8000 on the 4096 grid floors to one block.
        (dict(threshold=8000), 4096, True, False),
        # No threshold: the budget is the chunk.
        (dict(threshold=0), 16384, True, True),
        (dict(budget=8192, threshold=0), 8192, False, True),
        (dict(budget=8192, threshold=8192), 8192, False, True),
    ],
)
def test_dflash_lane_startup_warnings(overrides, chunk, window, whole_budget) -> None:
    h = _build(DFLASH, {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "dflash"}, **overrides)
    assert h.multiblock
    assert f"single-request chunk={chunk} tokens." in h.messages()[0]
    warnings = h.messages(logging.WARNING)
    assert any(WINDOW in w for w in warnings) == window
    assert any(WHOLE_BUDGET in w for w in warnings) == whole_budget
    assert len(warnings) == window + whole_budget


def test_other_lanes_do_not_warn() -> None:
    assert _build(NO_SPEC, budget=7840).messages(logging.WARNING) == []
    env = {"SX_OPT_ALIGN_MULTIBLOCK_SPEC": "mtp"}
    assert _build(MTP, env).messages(logging.WARNING) == []


def test_outside_align_mode_nothing_is_logged() -> None:
    h = _build(NO_SPEC, cache_mode="none")
    assert not h.sched.need_mamba_block_aligned_split
    assert not h.multiblock
    assert h.messages() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
