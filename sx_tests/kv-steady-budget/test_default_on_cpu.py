# SPDX-License-Identifier: Apache-2.0
"""``SX_OPT_KV_STEADY_BUDGET`` is on by default for the two admitted Qwen3.8 TP4 lanes.

Run (CPU is enough, no vLLM install and no GPU needed):

    python -m pytest -q sx_tests/kv-steady-budget/test_default_on_cpu.py

``VllmConfig.__post_init__`` sets ``SX_OPT_KV_STEADY_BUDGET=1`` when the operator
has not set the variable, the model is the exact Qwen3.8 TP4 topology (no-MTP, or
native MTP with 1..7 uniform rows) and the runner is V2. The two ``if`` statements
(the auto-enable and the ``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`` default that
follows it) and the lane-contract helpers are cut out of the repository's
``vllm/config/vllm.py`` with ``ast`` and run against fake configs, so the real
contract functions decide.

Asserted: the two lanes switch on and keep the graph-memory estimator on; any
other model, topology or speculative method stays off; an explicit ``0`` or ``1``
is never overwritten; the V2 property is not read for a model outside the lanes.
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_budget_boot as boot  # noqa: E402

CONFIG = os.path.join(boot.REPO, "vllm", "config", "vllm.py")
SWITCH = "SX_OPT_KV_STEADY_BUDGET"
ESTIMATE = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"
HELPERS = (
    "_sx_env_on",
    "_sx_mtp_lane_enabled",
    "_sx_kv_steady_budget_enabled",
    "_is_sm70_qwen38_nomtp_dual_compile_contract",
    "_is_sm70_qwen38_mtp_lane_contract",
    "_is_sm70_qwen38_lane_contract",
)


def load():
    source = open(CONFIG, encoding="utf-8").read()
    tree = ast.parse(source)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            parts.append(boot._segment(source, node))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_SX_MTP_LANE_MAX_K"
            for t in node.targets
        ):
            parts.append(boot._segment(source, node))
    assert len(parts) == len(HELPERS) + 1, "helpers not all found"
    enable = estimate = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if enable is None and SWITCH in test and "_is_sm70_qwen38_lane_contract" in test:
            enable = textwrap.dedent(boot._segment(source, node))
        elif estimate is None and ESTIMATE in test:
            estimate = textwrap.dedent(boot._segment(source, node))
    assert enable is not None, "auto-enable statement not found"
    assert estimate is not None, "estimator default statement not found"
    return "\n\n".join(parts), enable, estimate


HELPER_SOURCE, ENABLE, ESTIMATE_DEFAULT = load()


def text_config(**over):
    values = dict(
        hidden_size=2560,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        num_attention_heads=24,
        num_key_value_heads=2,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    values.update(over)
    return SimpleNamespace(**values)


def model_config(**over):
    return SimpleNamespace(
        architectures=["Qwen4ExpForCausalLM"],
        dtype="float16",
        hf_text_config=text_config(**over),
        multimodal_config=None,
    )


def mtp_config(target, k=4, method="mtp", **over):
    values = dict(
        method=method,
        use_qwen4_exp_mtp=lambda: True,
        num_speculative_tokens=k,
        num_speculative_state_tokens=lambda: k,
        parallel_drafting=False,
        rejection_sample_method="standard",
        target_model_config=target,
    )
    values.update(over)
    return SimpleNamespace(**values)


def parallel(tp=4, pp=1):
    return SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=pp)


def run(monkeypatch, *, model, spec=None, par=None, v2=True, switch=None, lane_env=None):
    for name in (SWITCH, ESTIMATE, "SX_OPT_MTP_LANE"):
        monkeypatch.delenv(name, raising=False)
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    if lane_env is not None:
        monkeypatch.setenv("SX_OPT_MTP_LANE", lane_env)
    logger = boot.RecordingLogger()
    namespace = {
        "os": os,
        "logger": logger,
        "torch": SimpleNamespace(float16="float16"),
        "Any": object,
        "self": SimpleNamespace(
            model_config=model,
            speculative_config=spec,
            parallel_config=par or parallel(),
            use_v2_model_runner=v2,
        ),
    }
    exec(HELPER_SOURCE, namespace)  # noqa: S102
    exec(ENABLE, namespace)  # noqa: S102
    exec(ESTIMATE_DEFAULT, namespace)  # noqa: S102
    return os.environ.get(SWITCH), os.environ.get(ESTIMATE), logger.text()


def test_nomtp_lane_switches_on_and_keeps_the_estimator_on(monkeypatch):
    switch, estimate, text = run(monkeypatch, model=model_config())
    assert switch == "1" and estimate is None
    assert "Auto-enabling the KV steady-state budget" in text
    assert "SX_OPT_KV_STEADY_BUDGET=0 to size it the old way" in text


@pytest.mark.parametrize("k", [1, 4, 7])
def test_mtp_lane_switches_on(monkeypatch, k):
    target = model_config()
    switch, estimate, text = run(
        monkeypatch, model=target, spec=mtp_config(target, k=k)
    )
    assert switch == "1" and estimate is None
    assert "Auto-enabling the KV steady-state budget" in text


def test_mtp_lane_through_the_target_model_config(monkeypatch):
    # The draft model's own config is what ``model_config`` holds in some paths.
    target = model_config()
    draft = model_config(num_hidden_layers=1)
    switch, _, _ = run(monkeypatch, model=draft, spec=mtp_config(target))
    assert switch == "1"


@pytest.mark.parametrize(
    "case",
    [
        "k8",
        "dflash",
        "tp2",
        "pp2",
        "other_hidden",
        "other_experts",
        "parallel_drafting",
        "non_standard_rejection",
        "mtp_lane_off",
        "no_v2",
    ],
)
def test_everything_outside_the_two_lanes_stays_off(monkeypatch, case):
    target = model_config()
    kwargs: dict = dict(model=target)
    if case == "k8":
        kwargs["spec"] = mtp_config(target, k=8)
    elif case == "dflash":
        kwargs["spec"] = mtp_config(target, method="dflash")
    elif case == "tp2":
        kwargs["par"] = parallel(tp=2)
    elif case == "pp2":
        kwargs["par"] = parallel(pp=2)
    elif case == "other_hidden":
        kwargs["model"] = model_config(hidden_size=4096)
    elif case == "other_experts":
        kwargs["model"] = model_config(num_experts=256)
    elif case == "parallel_drafting":
        kwargs["spec"] = mtp_config(target, parallel_drafting=True)
    elif case == "non_standard_rejection":
        kwargs["spec"] = mtp_config(target, rejection_sample_method="probabilistic")
    elif case == "mtp_lane_off":
        kwargs["spec"] = mtp_config(target)
        kwargs["lane_env"] = "0"
    elif case == "no_v2":
        kwargs["v2"] = False
    switch, estimate, text = run(monkeypatch, **kwargs)
    assert switch is None, case
    assert estimate == "0", case
    assert "Auto-enabling the KV steady-state budget" not in text


def test_an_explicit_zero_opts_out(monkeypatch):
    switch, estimate, text = run(monkeypatch, model=model_config(), switch="0")
    assert switch == "0" and estimate == "0"
    assert "Auto-enabling the KV steady-state budget" not in text


def test_an_explicit_one_is_kept_and_not_logged_as_automatic(monkeypatch):
    switch, estimate, text = run(monkeypatch, model=model_config(), switch="1")
    assert switch == "1" and estimate is None
    assert "Auto-enabling the KV steady-state budget" not in text
    assert "SX_OPT_KV_STEADY_BUDGET=1: keeping the graph memory estimate on" in text


class _SelfWithProperty:
    """``use_v2_model_runner`` is a property that raises for an invalid config."""

    def __init__(self, model):
        self.model_config = model
        self.speculative_config = None
        self.parallel_config = parallel()

    @property
    def use_v2_model_runner(self):
        raise AssertionError("read the V2 property for a model outside the lanes")


def test_the_v2_property_is_not_read_outside_the_lanes(monkeypatch):
    for name in (SWITCH, ESTIMATE):
        monkeypatch.delenv(name, raising=False)
    namespace = {
        "os": os,
        "logger": boot.RecordingLogger(),
        "torch": SimpleNamespace(float16="float16"),
        "Any": object,
        "self": _SelfWithProperty(model_config(hidden_size=4096)),
    }
    exec(HELPER_SOURCE, namespace)  # noqa: S102
    exec(ENABLE, namespace)  # noqa: S102
    assert SWITCH not in os.environ
