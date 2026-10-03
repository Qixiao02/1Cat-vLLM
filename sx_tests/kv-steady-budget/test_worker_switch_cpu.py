# SPDX-License-Identifier: Apache-2.0
"""``Worker.determine_available_memory`` with and without SX_OPT_KV_STEADY_BUDGET.

Run (CPU is enough, no vLLM install and no GPU needed; see ``kv_budget_boot.py``):

    python -m pytest -q sx_tests/kv-steady-budget/test_worker_switch_cpu.py

The methods under test are the repository's own, cut out of
``vllm/v1/worker/gpu_worker.py`` with ``ast``; the memory profiler, ``torch``,
the platform and the logger are fakes, and the graph-reserve helpers are the
real ones from ``cudagraph_utils.py``. A GPU adds only the real measurements.

Asserted:
* switch off (unset or 0): the KV budget is exactly the old arithmetic,
  ``requested - non_kv - graph_estimate``, nothing is logged about the steady
  budget, the SM70 graph reserve is not applied and no plan is kept;
* switch on, SM70 + V2: the graph reserve is the profiled activation peak
  (upstream 4bbaf64fc), and the KV budget is the plan's ``min(utilisation bound,
  physical bound)``; the log has the one-line summary the V100 script parses;
* switch on but not SM70/V2: a warning, the old arithmetic;
* an explicit ``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`` or
  ``VLLM_V2_CUDAGRAPH_MEM_MIB`` is honoured as the graph reserve;
* an explicit ``kv_cache_memory_bytes`` skips all of it;
* the phase marks and the audit do nothing without a plan, log the free memory
  deltas with one, report OK/SHORT, and raise only in strict mode.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_budget_boot as boot  # noqa: E402

kb = boot.load_budget()
MiB = boot.MiB
TOTAL = 32510 * MiB
CONTEXT = 900 * MiB
WEIGHTS = 21000 * MiB
NON_TORCH = 1300 * MiB
RESIDUE = 200 * MiB
ACTIVATION = 1200 * MiB
UTIL = 0.93
NON_KV = WEIGHTS + NON_TORCH + RESIDUE + ACTIVATION


class FakeRunner:
    def __init__(self, graph_estimate: int = 0) -> None:
        self.model_memory_usage = WEIGHTS
        self.calls = 0
        self.graph_estimate = graph_estimate

    def profile_run(self) -> None:
        self.calls += 1

    def profile_cudagraph_memory(self) -> int:
        return self.graph_estimate


def make_profile() -> boot.ProfileResult:
    return boot.ProfileResult(
        weights_memory=WEIGHTS,
        non_torch_increase=NON_TORCH,
        before_profile=boot.Snapshot(
            torch_peak=WEIGHTS + RESIDUE,
            torch_memory=WEIGHTS + RESIDUE,
        ),
        after_profile=boot.Snapshot(
            free_memory=TOTAL - CONTEXT - (WEIGHTS + RESIDUE + NON_TORCH),
        ),
    )


def make_worker(*, util=UTIL, spec_method=None, v2=True, **kwargs):
    cuda = boot.FakeCuda()
    if "lane_contract" not in kwargs:
        # The admitted Qwen3.8 lanes: native MTP, or no speculative config.
        kwargs["lane_contract"] = {"mtp": "mtp", None: "nomtp"}.get(spec_method)
    cls, logger = boot.worker_class(
        profile=make_profile(),
        profile_torch_peak=WEIGHTS + RESIDUE + ACTIVATION,
        cuda=cuda,
        **kwargs,
    )
    worker = cls()
    worker.device = "cuda:0"
    worker.use_v2_model_runner = v2
    worker.init_snapshot = SimpleNamespace(
        free_memory=TOTAL - CONTEXT,
        total_memory=TOTAL,
        cuda_memory=CONTEXT,
        torch_memory=0,
    )
    worker.requested_memory = int(util * TOTAL)
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None, gpu_memory_utilization=util
    )
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=boot._Mode.FULL_AND_PIECEWISE
        ),
        speculative_config=(
            None if spec_method is None else SimpleNamespace(method=spec_method)
        ),
    )
    worker.model_runner = FakeRunner()
    return worker, logger, cuda


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("SX_OPT_KV_STEADY") or name.startswith(
            "VLLM_V2_CUDAGRAPH_MEM"
        ):
            monkeypatch.delenv(name, raising=False)


def legacy_budget(graph: int = 0) -> int:
    return int(UTIL * TOTAL) - NON_KV - graph


# ----------------------------------------------------------------- switch off


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_switch_off_is_the_old_arithmetic(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(kb.ENV_SWITCH, value)
    worker, logger, _ = make_worker(spec_method="mtp")
    assert worker.determine_available_memory() == legacy_budget()
    assert worker.available_kv_cache_memory_bytes == legacy_budget()
    assert not hasattr(worker, "_sx_steady_plan")
    assert "KV steady" not in logger.text()
    assert "SM70 graph memory reserve" not in logger.text()
    assert worker.cudagraph_memory_estimate == 0


def test_switch_off_does_not_apply_the_sm70_reserve_even_with_the_estimator_on():
    worker, logger, _ = make_worker(spec_method="mtp", estimate_cudagraphs=True)
    worker.determine_available_memory()
    assert worker.cudagraph_memory_estimate == 0
    assert "SM70 graph memory reserve" not in logger.text()


def test_switch_off_keeps_an_explicit_runner_estimate():
    worker, _, _ = make_worker(spec_method="mtp")
    worker.model_runner = FakeRunner(graph_estimate=300 * MiB)
    assert worker.determine_available_memory() == legacy_budget(300 * MiB)


# ------------------------------------------------------------------ switch on


def test_switch_on_applies_the_activation_peak_as_graph_reserve(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, logger, _ = make_worker(util=0.87, spec_method="mtp")
    available = worker.determine_available_memory()
    assert worker.cudagraph_memory_estimate == ACTIVATION
    assert "SM70 graph memory reserve before KV allocation: 1.17 GiB" in logger.text()
    plan = worker._sx_steady_plan
    assert plan.graph_reserve == ACTIVATION
    # util 0.87 is the utilisation-bound case: upstream's arithmetic.
    assert plan.limiting == "utilisation"
    assert available == legacy_budget_at(0.87, ACTIVATION) == plan.kv_bytes


def legacy_budget_at(util: float, graph: int) -> int:
    return int(util * TOTAL) - NON_KV - graph


def test_switch_on_takes_the_physical_bound_at_a_high_utilisation(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, logger, _ = make_worker(util=0.99, spec_method="mtp")
    available = worker.determine_available_memory()
    plan = worker._sx_steady_plan
    assert plan.limiting == "physical"
    assert available == plan.kv_physical < plan.kv_utilisation
    assert plan.predicted_free_at_peak == plan.headroom
    text = logger.text("info")
    assert "KV steady budget: util 0.9900" in text
    assert "physical bound" in text
    assert "KV steady budget [kv=%d total=%d" % (available, TOTAL) in text
    assert worker.available_kv_cache_memory_bytes == available


def test_kv_grows_with_the_utilisation_up_to_the_physical_bound(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    sizes = []
    for util in (0.87, 0.90, 0.93, 0.95, 0.99):
        worker, _, _ = make_worker(util=util, spec_method="mtp")
        sizes.append(worker.determine_available_memory())
    assert sizes == sorted(sizes)
    assert sizes[-1] == sizes[-2] > sizes[0]


def test_nomtp_lane_uses_its_own_reference(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, _, _ = make_worker(util=0.99, spec_method=None)
    worker.determine_available_memory()
    assert "nomtp" in worker._sx_steady_plan.post_sizing_source
    worker, _, _ = make_worker(util=0.99, spec_method="mtp")
    worker.determine_available_memory()
    assert "lane reference mtp" in worker._sx_steady_plan.post_sizing_source


def test_dflash_lane_has_no_reference(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, _, _ = make_worker(util=0.99, spec_method="dflash")
    worker.determine_available_memory()
    assert "unreferenced" in worker._sx_steady_plan.post_sizing_source


@pytest.mark.parametrize("spec_method", [None, "mtp"])
def test_a_model_outside_the_admitted_lanes_has_no_reference(monkeypatch, spec_method):
    """The references are measurements of the Qwen3.8 TP4 lanes, not of MTP."""
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, _, _ = make_worker(util=0.99, spec_method=spec_method, lane_contract=None)
    worker.determine_available_memory()
    plan = worker._sx_steady_plan
    assert "unreferenced" in plan.post_sizing_source
    assert plan.post_sizing == ACTIVATION + kb.DEFAULT_UNREFERENCED_EXTRA_MIB * MiB


def test_switch_on_without_v2_or_sm70_changes_nothing(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    for kwargs in ({"v2": False}, {"is_sm70": False}):
        worker, logger, _ = make_worker(spec_method="mtp", **kwargs)
        # without SM70 the upstream reserve block is skipped too
        assert worker.determine_available_memory() == legacy_budget()
        assert not hasattr(worker, "_sx_steady_plan")
    assert "only applies to SM70 on the V2 model runner" in logger.text("warning")


def test_estimator_disabled_means_no_graph_reserve(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, _, _ = make_worker(util=0.87, spec_method="mtp", estimate_cudagraphs=False)
    available = worker.determine_available_memory()
    assert worker._sx_steady_plan.graph_reserve == 0
    assert available == legacy_budget_at(0.87, 0)


def test_explicit_graph_reserve_override_is_honoured(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "700")
    worker, _, _ = make_worker(util=0.87, spec_method="mtp")
    available = worker.determine_available_memory()
    assert worker._sx_steady_plan.graph_reserve == 700 * MiB
    assert available == legacy_budget_at(0.87, 700 * MiB)


def test_explicit_kv_bytes_skips_the_plan(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, logger, _ = make_worker(spec_method="mtp")
    worker.cache_config.kv_cache_memory_bytes = 128 * MiB
    assert worker.determine_available_memory() == 128 * MiB
    assert not hasattr(worker, "_sx_steady_plan")
    assert "KV steady" not in logger.text()


# ------------------------------------------------------------- marks and audit


def planned_worker(monkeypatch, **kwargs):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    worker, logger, cuda = make_worker(util=0.99, spec_method="mtp", **kwargs)
    worker.determine_available_memory()
    return worker, logger, cuda


def test_idle_cache_is_released_only_with_a_plan(monkeypatch):
    worker, logger, cuda = make_worker(spec_method="mtp")
    worker.determine_available_memory()
    worker._sx_release_idle_cache()
    assert type(worker).calls["empty_cache"] == 0
    assert "after_empty_cache" not in logger.text()

    worker, logger, cuda = planned_worker(monkeypatch)
    worker._sx_release_idle_cache()
    assert type(worker).calls["empty_cache"] == 1
    assert "KV steady phase after_empty_cache" in logger.text()


def test_idle_cache_release_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(kb.ENV_EMPTY_CACHE, "0")
    worker, logger, cuda = planned_worker(monkeypatch)
    worker._sx_release_idle_cache()
    assert type(worker).calls["empty_cache"] == 0
    assert "after_empty_cache" not in logger.text()


def test_marks_and_audit_do_nothing_without_a_plan():
    worker, logger, cuda = make_worker(spec_method="mtp")
    worker.determine_available_memory()
    cuda.free = 123
    worker._sx_mark("dummy_runs")
    worker._sx_after_kv_cache_init(SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=5)]))
    worker._sx_steady_audit(0)
    assert not hasattr(worker, "_sx_steady_marks")
    assert not hasattr(worker, "_sx_kv_bytes_allocated")
    assert "KV steady" not in logger.text()


def run_startup(worker, cuda, *, kv, init_extra, warmup_growth, cached):
    """Drive the hooks the way initialize_from_config / compile_or_warm_up_model do."""
    plan = worker._sx_steady_plan
    f_prof = worker._sx_steady_marks[0][1]
    cuda.free = f_prof - kv - init_extra
    worker._sx_after_kv_cache_init(
        SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=kv)])
    )
    cuda.free -= warmup_growth
    cuda.reserved, cuda.allocated = cached, 0
    worker._sx_steady_audit(1000 * MiB)
    return plan


def test_audit_reports_ok_when_the_growth_matches_the_plan(monkeypatch):
    worker, logger, cuda = planned_worker(monkeypatch)
    plan = worker._sx_steady_plan
    # What the plan assumed: after the KV cache, activations (cached) + P.
    run_startup(
        worker, cuda, kv=plan.kv_bytes, init_extra=20 * MiB,
        warmup_growth=ACTIVATION + plan.post_sizing - plan.graph_reserve - 400 * MiB,
        cached=ACTIVATION,
    )
    text = logger.text()
    assert "KV steady phase after_kv_cache" in text
    assert "KV steady phase end_of_warmup" in text
    assert "KV steady audit: OK" in text
    assert logger.text("error") == ""


def test_audit_reports_short_as_an_error(monkeypatch):
    worker, logger, cuda = planned_worker(monkeypatch)
    plan = worker._sx_steady_plan
    run_startup(
        worker, cuda, kv=plan.kv_bytes, init_extra=20 * MiB,
        warmup_growth=ACTIVATION + plan.post_sizing + 600 * MiB,
        cached=ACTIVATION,
    )
    assert "KV steady audit: SHORT by" in logger.text("error")
    assert kb.ENV_RESERVE_MIB in logger.text("error")


def test_strict_mode_raises_when_short(monkeypatch):
    monkeypatch.setenv(kb.ENV_STRICT, "1")
    worker, logger, cuda = planned_worker(monkeypatch)
    plan = worker._sx_steady_plan
    with pytest.raises(RuntimeError, match="KV steady audit"):
        run_startup(
            worker, cuda, kv=plan.kv_bytes, init_extra=0,
            warmup_growth=ACTIVATION + plan.post_sizing + 900 * MiB,
            cached=ACTIVATION,
        )


def test_strict_mode_does_not_raise_when_ok(monkeypatch):
    monkeypatch.setenv(kb.ENV_STRICT, "1")
    worker, logger, cuda = planned_worker(monkeypatch)
    plan = worker._sx_steady_plan
    run_startup(
        worker, cuda, kv=plan.kv_bytes, init_extra=0,
        warmup_growth=ACTIVATION, cached=ACTIVATION,
    )
    assert "KV steady audit: OK" in logger.text()


def test_load_margin_env_changes_the_audit(monkeypatch):
    worker, logger, cuda = planned_worker(monkeypatch)
    plan = worker._sx_steady_plan
    monkeypatch.setenv(kb.ENV_LOAD_MIB, "0")
    run_startup(
        worker, cuda, kv=plan.kv_bytes, init_extra=0,
        warmup_growth=ACTIVATION + plan.post_sizing - plan.graph_reserve,
        cached=ACTIVATION,
    )
    assert "+0 MiB load margin" in logger.text()
