# SPDX-License-Identifier: Apache-2.0
"""The full-chunk warm-up of SX_OPT_KV_STEADY_BUDGET (vllm/v1/worker/gpu/warmup.py).

Run (CPU is enough, no vLLM install and no GPU needed):

    python -m pytest -q sx_tests/kv-steady-budget/test_warmup_steady_cpu.py

``warmup_kernels`` runs 6-48 token prompts; the workspaces of a full prefill
chunk (indexer score buffers, grouped page4 workspace, GDN chunk temporaries)
are first allocated by the first long real request, after the KV cache is sized
and outside any audit. With the switch on one full-chunk prefill joins the
warm-up profiles. The two helpers that decide which prompt lengths run are cut
out of the repository's ``warmup.py`` with ``ast`` and run against a fake model
runner.

Asserted:
* switch off: the prompt lengths are exactly what the kernels advertise plus
  the default, as before;
* switch on: one more length, min(max_num_batched_tokens, max_model_len), never
  one that exceeds the token budget or is not larger than the default;
* SX_OPT_KV_STEADY_WARMUP_TOKENS shortens or disables it;
* the kernel-advertised lengths are kept.
"""

from __future__ import annotations

import ast
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_budget_boot as boot  # noqa: E402

kb = boot.load_budget()
WARMUP = os.path.join(boot.REPO, "vllm", "v1", "worker", "gpu", "warmup.py")


def load_helpers():
    source = open(WARMUP, encoding="utf-8").read()
    wanted = {"_steady_warmup_tokens", "_kernel_prefill_warmup_token_counts"}
    parts = [
        boot._segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(parts) == len(wanted), "warm-up helpers not found"
    namespace = {"kv_steady_budget": kb, "GPUModelRunner": object}
    exec(compile("\n\n".join(parts), WARMUP, "exec"), namespace)  # noqa: S102
    return namespace


HELPERS = load_helpers()


def runner(batched=8192, max_len=32768, advertised=((33,), (33, 48))):
    layers = {
        f"layer{i}": SimpleNamespace(kernel_warmup_prefill_token_counts=counts)
        for i, counts in enumerate(advertised)
    }
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=batched),
        max_model_len=max_len,
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("SX_OPT_KV_STEADY"):
            monkeypatch.delenv(name, raising=False)


def counts(model_runner, default=6):
    return HELPERS["_kernel_prefill_warmup_token_counts"](model_runner, default)


def test_switch_off_is_the_advertised_lengths_only():
    assert counts(runner()) == (6, 33, 48)
    assert HELPERS["_steady_warmup_tokens"](runner()) == 0


def test_switch_on_adds_one_full_chunk(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    assert counts(runner()) == (6, 33, 48, 8192)


def test_the_chunk_is_bounded_by_the_window(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    assert counts(runner(batched=8192, max_len=4096)) == (6, 33, 48, 4096)


def test_a_tiny_budget_adds_nothing(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    # Not larger than the default prompt: nothing to add.
    assert counts(runner(batched=6, max_len=32768), default=6) == (6,)


@pytest.mark.parametrize(
    "value,expected",
    [("2048", (6, 33, 48, 2048)), ("0", (6, 33, 48)), ("99999", (6, 33, 48, 8192))],
)
def test_tokens_override(monkeypatch, value, expected):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    monkeypatch.setenv(kb.ENV_WARMUP_TOKENS, value)
    assert counts(runner()) == expected


def test_a_length_already_advertised_is_not_duplicated(monkeypatch):
    monkeypatch.setenv(kb.ENV_SWITCH, "1")
    assert counts(runner(advertised=((33,), (8192,)))) == (6, 33, 8192)
