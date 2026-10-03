# SPDX-License-Identifier: Apache-2.0
"""``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`` auto-default of the SM70 compile-graph lane.

Run (CPU is enough, no vLLM install and no GPU needed):

    python -m pytest -q sx_tests/kv-steady-budget/test_config_default_cpu.py

``VllmConfig.__post_init__`` sets the graph-memory estimator to 0 for the SM70
Flash-V100 compile-graph lane unless the operator set it. With
SX_OPT_KV_STEADY_BUDGET=1 on the V2 runner it stays on (upstream 4bbaf64fc does
that for every V2 run; the fork keeps its old default unless the switch is on).
The ``if``/``elif``/``else`` statement is cut out of the repository's
``vllm/config/vllm.py`` with ``ast`` and run with a fake ``self``.

Asserted: explicit value untouched in every case; switch off or V1: set to 0
and logged as before; switch on + V2: left unset, with the log line that says so.
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
NAME = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"


def load():
    source = open(CONFIG, encoding="utf-8").read()
    tree = ast.parse(source)
    helper = next(
        boot._segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sx_kv_steady_budget_enabled"
    )
    statement = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and NAME in ast.unparse(node.test):
            statement = textwrap.dedent(boot._segment(source, node))
            break
    assert statement is not None, "auto-default statement not found"
    return helper, statement


HELPER, STATEMENT = load()


def run(monkeypatch, *, v2: bool, switch: str | None, explicit: str | None):
    for name in (NAME, "SX_OPT_KV_STEADY_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    if switch is not None:
        monkeypatch.setenv("SX_OPT_KV_STEADY_BUDGET", switch)
    if explicit is not None:
        monkeypatch.setenv(NAME, explicit)
    logger = boot.RecordingLogger()
    namespace = {
        "os": os,
        "logger": logger,
        "self": SimpleNamespace(use_v2_model_runner=v2),
    }
    exec(HELPER, namespace)  # noqa: S102
    exec(STATEMENT, namespace)  # noqa: S102
    return os.environ.get(NAME), logger.text()


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("switch", [None, "0", "1"])
def test_an_explicit_value_is_never_touched(monkeypatch, v2, switch):
    value, text = run(monkeypatch, v2=v2, switch=switch, explicit="1")
    assert value == "1" and text == ""


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("switch", [None, "0", "true"])
def test_switch_off_keeps_the_old_default(monkeypatch, v2, switch):
    value, text = run(monkeypatch, v2=v2, switch=switch, explicit=None)
    assert value == "0"
    assert "Auto-setting VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in text


def test_switch_on_without_v2_keeps_the_old_default(monkeypatch):
    value, text = run(monkeypatch, v2=False, switch="1", explicit=None)
    assert value == "0"
    assert "Auto-setting" in text


def test_switch_on_with_v2_leaves_the_estimator_on(monkeypatch):
    value, text = run(monkeypatch, v2=True, switch="1", explicit=None)
    assert value is None
    assert "SX_OPT_KV_STEADY_BUDGET=1: keeping the graph memory estimate on" in text


class _SelfWithProperty:
    """``use_v2_model_runner`` is a property that can raise for an invalid config."""

    @property
    def use_v2_model_runner(self):
        raise AssertionError("read the V2 property with the switch off")


@pytest.mark.parametrize("switch", [None, "0"])
def test_switch_off_does_not_evaluate_the_v2_property(monkeypatch, switch):
    for name in (NAME, "SX_OPT_KV_STEADY_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    if switch is not None:
        monkeypatch.setenv("SX_OPT_KV_STEADY_BUDGET", switch)
    namespace = {"os": os, "logger": boot.RecordingLogger(), "self": _SelfWithProperty()}
    exec(HELPER, namespace)  # noqa: S102
    exec(STATEMENT, namespace)  # noqa: S102
    assert os.environ[NAME] == "0"
