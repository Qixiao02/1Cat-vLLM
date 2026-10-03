# SPDX-License-Identifier: Apache-2.0
"""Arithmetic of the steady-state KV budget (SX_OPT_KV_STEADY_BUDGET).

Run (CPU is enough, no vLLM install and no GPU needed; see ``kv_budget_boot.py``):

    python -m pytest -q sx_tests/kv-steady-budget/test_kv_steady_budget_cpu.py

Every number is bytes of one GPU rank, built from the V100-SXM2-32GB figures of
the 2026-10-01 runs (sx_bench/results/2026-10-01-fork-vs-official-best/): the
native-MTP lane at --gpu-memory-utilization 0.87 had a KV cache of ~2.0 GiB per
rank and a highest nvidia-smi memory.used of 31811 MiB. The activation peak,
the CUDA context and the card's usable total are assumptions (named below) and
only change the absolute values, not the relations that are asserted:

* the switch: only ``1`` turns it on; unset, ``0`` and anything else keep the
  old behaviour;
* the physical bound does not depend on --gpu-memory-utilization, the
  utilisation bound does, and the KV cache is the smaller of the two;
* at the lane's reference point the plan predicts exactly the headroom as the
  free memory at the steady peak, and it gains over the old sizing exactly
  ``total - reference_peak - headroom`` (nothing else), whatever the activation
  peak and the CUDA context are;
* the CUDA context is taken out of the post-sizing allocation without changing
  the physical bound (it is inside the measured free memory);
* an explicit SX_OPT_KV_STEADY_RESERVE_MIB, a lane without a reference, bad
  values, and the audit that compares the measured growth with the plan.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_budget_boot  # noqa: E402

kb = kv_budget_boot.load_budget()
MiB = kb.MiB

# Assumed card: the usable total CUDA reports (nvidia-smi shows 32768 MiB, CUDA
# hands out 258 MiB less). Everything below is derived from these.
TOTAL = 32510 * MiB
CONTEXT = 900 * MiB  # CUDA context + NCCL + custom all-reduce, outside the budget
ACTIVATION = 1200 * MiB  # torch peak increase of the measured profile
KV_TODAY = 2048 * MiB  # ~2.0 GiB: what util 0.87 gave
UTIL_REF = 0.87
# weights + non-torch + activation peak + residue, so that util 0.87 leaves KV_TODAY
NON_KV = int(UTIL_REF * TOTAL) - KV_TODAY


def inputs(util: float = 0.93, *, lane: str = "mtp", graph_reserve: int | None = None,
           context: int = CONTEXT, non_kv: int = NON_KV, total: int = TOTAL):
    live = non_kv - ACTIVATION  # weights + non-torch + residue stay allocated
    return kb.BudgetInputs(
        total_memory=total,
        requested_memory=int(util * total),
        util=util,
        init_cuda_memory=context,
        free_after_profile=total - context - live,
        non_kv_cache_memory=non_kv,
        activation_peak=ACTIVATION,
        graph_reserve=ACTIVATION if graph_reserve is None else graph_reserve,
        lane=lane,
    )


# ---------------------------------------------------------------- the switch


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), ("0", False), ("1", True), (" 1 ", True),
     ("true", False), ("2", False), ("on", False)],
)
def test_only_one_turns_the_switch_on(value, expected):
    env = {} if value is None else {kb.ENV_SWITCH: value}
    assert kb.steady_budget_enabled(env) is expected


def test_switch_reads_the_process_environment(monkeypatch):
    monkeypatch.delenv(kb.ENV_SWITCH, raising=False)
    assert kb.steady_budget_enabled() is False
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    assert kb.steady_budget_enabled() is True


def test_strict_flag(monkeypatch):
    monkeypatch.delenv(kb.ENV_STRICT, raising=False)
    assert kb.strict_enabled() is False
    monkeypatch.setenv(kb.ENV_STRICT, "1")
    assert kb.strict_enabled() is True


@pytest.mark.parametrize("raw", ["-1", "abc", "nan", "inf"])
def test_bad_memory_knobs_are_refused(raw):
    with pytest.raises(ValueError, match="SX_OPT_KV_STEADY"):
        kb.read_mib("SX_OPT_KV_STEADY_HEADROOM_MIB", 512, {"SX_OPT_KV_STEADY_HEADROOM_MIB": raw})


def test_read_mib_defaults_and_values():
    assert kb.read_mib("X", 7, {}) == 7
    assert kb.read_mib("X", 7, {"X": ""}) == 7
    assert kb.read_mib("X", None, {}) is None
    assert kb.read_mib("X", 7, {"X": "512.9"}) == 512


# ------------------------------------------------------------------- the plan


def test_the_utilisation_bound_is_the_upstream_arithmetic():
    plan = kb.plan_kv_budget(inputs(0.93))
    assert plan.kv_utilisation == int(0.93 * TOTAL) - NON_KV - ACTIVATION
    # What the sizing gives without the switch is kept for the log.
    assert plan.kv_unreserved == int(0.93 * TOTAL) - NON_KV
    assert plan.kv_unreserved - plan.kv_utilisation == plan.graph_reserve


def test_physical_bound_does_not_depend_on_the_utilisation():
    bounds = {kb.plan_kv_budget(inputs(u)).kv_physical for u in (0.87, 0.90, 0.93, 0.95, 0.99)}
    assert len(bounds) == 1


def test_kv_is_the_smaller_bound_and_names_it():
    low = kb.plan_kv_budget(inputs(0.87))
    high = kb.plan_kv_budget(inputs(0.99))
    assert low.limiting == "utilisation" and low.kv_bytes == low.kv_utilisation
    assert high.limiting == "physical" and high.kv_bytes == high.kv_physical
    assert low.kv_bytes < high.kv_bytes


def test_kv_grows_with_utilisation_until_the_physical_bound():
    sizes = [kb.plan_kv_budget(inputs(u)).kv_bytes for u in (0.87, 0.90, 0.93, 0.95, 0.97, 0.99)]
    assert sizes == sorted(sizes)
    assert sizes[-1] == sizes[-2] == kb.plan_kv_budget(inputs(0.99)).kv_physical


@pytest.mark.parametrize("util", [0.93, 0.95, 0.99])
def test_physical_bound_leaves_exactly_the_headroom(util):
    plan = kb.plan_kv_budget(inputs(util))
    assert plan.limiting == "physical"
    assert plan.predicted_free_at_peak == plan.headroom == 512 * MiB


def test_predicted_free_memory_is_larger_when_the_utilisation_limits():
    plan = kb.plan_kv_budget(inputs(0.87))
    assert plan.limiting == "utilisation"
    assert plan.predicted_free_at_peak > plan.headroom


@pytest.mark.parametrize("activation", [400, 1200, 1800])
@pytest.mark.parametrize("context", [300, 900, 1500])
def test_gain_over_todays_sizing_is_the_unused_headroom(activation, context):
    """At the reference point the physical bound reproduces today's peak, so the
    KV cache grows by (total - reference peak - headroom) and by nothing else."""
    global ACTIVATION
    saved, ACTIVATION = ACTIVATION, activation * MiB
    try:
        inp = inputs(0.99, context=context * MiB)
        plan = kb.plan_kv_budget(inp)
    finally:
        ACTIVATION = saved
    ref = kb.LANE_REFERENCES["mtp"]
    expected_gain = TOTAL - ref.peak_mib * MiB - 512 * MiB
    assert plan.limiting == "physical"
    assert plan.kv_bytes - KV_TODAY == expected_gain


def test_context_moves_out_of_the_post_sizing_allocation_one_for_one():
    a = kb.plan_kv_budget(inputs(0.95, context=600 * MiB))
    b = kb.plan_kv_budget(inputs(0.95, context=1400 * MiB))
    assert a.post_sizing - b.post_sizing == 800 * MiB
    # The context is inside the measured free memory, so the KV size is the same.
    assert a.kv_physical == b.kv_physical


def test_plan_predicts_headroom_for_every_lane_reference():
    for lane in ("mtp", "nomtp"):
        plan = kb.plan_kv_budget(inputs(0.99, lane=lane))
        assert plan.limiting == "physical"
        assert plan.predicted_free_at_peak == 512 * MiB


def test_nomtp_reference_is_its_own_row():
    mtp = kb.plan_kv_budget(inputs(0.99, lane="mtp"))
    nomtp = kb.plan_kv_budget(inputs(0.99, lane="nomtp"))
    assert mtp.post_sizing != nomtp.post_sizing
    assert "nomtp" in nomtp.post_sizing_source


def test_utilisation_that_stops_limiting():
    plan = kb.plan_kv_budget(inputs(0.99))
    at = kb.plan_kv_budget(inputs(plan.util_to_fill_physical))
    below = kb.plan_kv_budget(inputs(plan.util_to_fill_physical - 0.01))
    assert at.kv_bytes == pytest.approx(plan.kv_physical, abs=2 * MiB)
    assert below.kv_bytes < plan.kv_physical


def test_explicit_reserve_replaces_the_lane_reference():
    env = {kb.ENV_RESERVE_MIB: "1500"}
    plan = kb.plan_kv_budget(inputs(0.99), env)
    assert plan.post_sizing == 1500 * MiB
    assert kb.ENV_RESERVE_MIB in plan.post_sizing_source
    base = kb.plan_kv_budget(inputs(0.99))
    assert plan.kv_physical - base.kv_physical == base.post_sizing - 1500 * MiB


def test_explicit_headroom_is_used():
    base = kb.plan_kv_budget(inputs(0.99))
    plan = kb.plan_kv_budget(inputs(0.99), {kb.ENV_HEADROOM_MIB: "256"})
    assert plan.headroom == 256 * MiB
    assert plan.kv_physical - base.kv_physical == 256 * MiB


def test_lane_without_a_reference_uses_the_graph_reserve_plus_extra():
    plan = kb.plan_kv_budget(inputs(0.99, lane="other"))
    assert plan.post_sizing == ACTIVATION + kb.DEFAULT_UNREFERENCED_EXTRA_MIB * MiB
    plan = kb.plan_kv_budget(inputs(0.99, lane="other"), {"SX_OPT_KV_STEADY_EXTRA_MIB": "0"})
    assert plan.post_sizing == ACTIVATION


def test_post_sizing_never_below_the_graph_reserve():
    # A device whose context is larger than the reference overshoot.
    plan = kb.plan_kv_budget(inputs(0.99, context=9000 * MiB))
    assert plan.post_sizing == ACTIVATION


def test_explicit_zero_graph_reserve_is_respected():
    plan = kb.plan_kv_budget(inputs(0.93, graph_reserve=0))
    assert plan.graph_reserve == 0
    assert plan.kv_utilisation == int(0.93 * TOTAL) - NON_KV


def test_plan_lines_carry_the_machine_readable_summary():
    plan = kb.plan_kv_budget(inputs(0.93))
    text = "\n".join(plan.lines)
    assert "KV steady budget [kv=%d total=%d" % (plan.kv_bytes, TOTAL) in text
    for key in ("requested=", "free_after_profile=", "post_sizing=", "headroom=", "limiting="):
        assert key in text


def test_negative_budget_is_reported_not_hidden():
    plan = kb.plan_kv_budget(inputs(0.50))
    assert plan.kv_bytes < 0 and plan.limiting == "utilisation"


def test_estimate_table_for_the_four_utilisations():
    """The numbers quoted in the change description (assumed card, see above)."""
    rows = {}
    for util in (0.87, 0.90, 0.93, 0.95):
        plan = kb.plan_kv_budget(inputs(util))
        rows[util] = (plan.kv_bytes / MiB, plan.limiting)
    # 0.87/0.90 are bound by the utilisation (graph reserve now charged);
    # 0.93 and up by the physical bound, which is 187 MiB above today's 2048.
    assert rows[0.87][1] == rows[0.90][1] == "utilisation"
    assert rows[0.93][1] == rows[0.95][1] == "physical"
    assert rows[0.93][0] == rows[0.95][0] == pytest.approx(2048 + 187, abs=1)
    assert rows[0.93][0] > KV_TODAY / MiB


# ------------------------------------------------------------------ the audit


def audit_inputs(**over):
    base = dict(
        free_after_profile=6500 * MiB,
        kv_bytes_allocated=2200 * MiB,
        free_after_kv=4290 * MiB,  # 10 MiB of non-KV in initialize_kv_cache
        free_at_end=1700 * MiB,
        cached_free=900 * MiB,
        activation_peak=1200 * MiB,
        graph_capture_bytes=1400 * MiB,
        post_sizing_planned=2600 * MiB,
        headroom_planned=512 * MiB,
        load_margin=384 * MiB,
    )
    base.update(over)
    return kb.AuditInputs(**base)


def test_audit_measures_the_growth_without_the_activation_cache():
    res = kb.audit(audit_inputs())
    assert res.kv_init_extra == 10 * MiB
    assert res.warmup_growth == 2590 * MiB
    # 900 MiB of the growth is the allocator holding memory for activations.
    assert res.post_sizing_measured == 10 * MiB + 2590 * MiB - 900 * MiB
    assert res.activation_deficit == 300 * MiB


def test_audit_ok_when_the_plan_assumed_more():
    res = kb.audit(audit_inputs(free_at_end=2400 * MiB))
    assert res.ok and res.surplus > 0
    assert "OK" in res.lines[-1]


def test_audit_short_names_the_remedy():
    res = kb.audit(audit_inputs(free_at_end=1000 * MiB))
    assert not res.ok and res.surplus < 0
    assert "SHORT by" in res.lines[-1]
    assert f"{kb.ENV_RESERVE_MIB}={res.suggested_reserve_mib}" in res.lines[-1]


def test_audit_suggestion_makes_the_plan_exact():
    """Feeding the suggested reserve back gives predicted free == headroom."""
    res = kb.audit(audit_inputs(free_at_end=1000 * MiB))
    inp = inputs(0.99, lane="other")
    plan = kb.plan_kv_budget(inp, {kb.ENV_RESERVE_MIB: str(res.suggested_reserve_mib)})
    assert plan.predicted_free_at_peak == plan.headroom


def test_audit_activation_fully_cached_costs_nothing_more():
    res = kb.audit(audit_inputs(cached_free=1500 * MiB))
    assert res.activation_deficit == 0
    assert res.post_sizing_measured == 10 * MiB + 2590 * MiB - 1200 * MiB


def test_audit_never_reports_negative_growth():
    res = kb.audit(audit_inputs(free_at_end=5000 * MiB))
    assert res.warmup_growth == 0


# ------------------------------------------------------------- large warm-up


def test_steady_warmup_tokens_default_is_a_full_chunk():
    assert kb.steady_warmup_tokens(8192, 32768, {}) == 8192
    assert kb.steady_warmup_tokens(8192, 4096, {}) == 4096


def test_steady_warmup_tokens_override_and_disable():
    assert kb.steady_warmup_tokens(8192, 32768, {kb.ENV_WARMUP_TOKENS: "2048"}) == 2048
    assert kb.steady_warmup_tokens(8192, 32768, {kb.ENV_WARMUP_TOKENS: "0"}) == 0
    assert kb.steady_warmup_tokens(8192, 32768, {kb.ENV_WARMUP_TOKENS: "99999"}) == 8192


def test_fit_check_for_the_large_warmup():
    assert kb.steady_warmup_fits(100, 99)
    assert not kb.steady_warmup_fits(100, 100)  # block 0 is the null block
    assert kb.steady_warmup_fits(1000, 1)
