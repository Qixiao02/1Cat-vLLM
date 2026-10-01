# SPDX-License-Identifier: Apache-2.0
"""The whole Qwen4Exp loading stack with every SX_OPT_LOAD_* / MOE_LOAD_INDEX switch.

Run (CPU is enough, no vLLM install needed; see ``pipeline_harness.py``):

    python -m pytest -q sx_tests/load-index/test_load_pipeline.py
    python sx_tests/load-index/bench_load_pipeline.py        # cost per tensor

Three nested AutoWeightsLoader levels (outer model -> causal LM -> Qwen4ExpModel),
the QSA scale remap, maybe_fuse_shared_experts and FusedMoE.load_weights, all cut
from the real source; toy modules and a recording weight_loader.

Asserted, for every one of the 2^3 switch combinations against all switches off
(the original code paths, each checked against a verbatim copy in the other
tests of this group):
* the set of loaded names, the sequence of expert weight_loader calls (names,
  shard ids, expert ids, shapes, tensor values) and every parameter value are
  identical, with the checkpoint in sorted order and in random order (random
  order makes FusedMoE.load_weights run once per tensor, so the cached index is
  reused across thousands of short calls);
* mappings with redundant experts (one tensor loads into several physical
  experts) and 512 experts;
* a checkpoint name that matches no module raises the same ValueError.
Expected: all pass.
"""

from __future__ import annotations

import itertools
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import load_boot  # noqa: E402
import pipeline_harness as ph  # noqa: E402

U = ph.utils_namespace()
M = ph.model_namespace()
make_mapping = load_boot.make_expert_params_mapping()

COMBOS = list(itertools.product([False, True], repeat=3))


class _Model:
    def named_parameters(self):
        return []


def set_switches(monkeypatch, moe_index: bool, can_skip: bool, qsa_remap: bool):
    monkeypatch.setenv("SX_OPT_MOE_LOAD_INDEX", "1" if moe_index else "0")
    monkeypatch.setitem(U, "_SX_LOAD_CAN_SKIP", can_skip)
    monkeypatch.setitem(M, "_SX_LOAD_QSA_REMAP", qsa_remap)


def run(num_layers: int, num_experts: int, redundant: int, qsa_ids, order: str,
        extra_names=()):
    model, counter = ph.build_model(
        num_layers,
        lambda: make_mapping(
            _Model(), "gate_proj", "down_proj", "up_proj", num_experts, redundant, True
        ),
        qsa_ids=qsa_ids,
    )
    counter.record_values = True
    names = list(ph.checkpoint_names(num_layers, num_experts, qsa_ids=qsa_ids))
    names += list(extra_names)
    if order == "shuffled":
        random.Random(11).shuffle(names)
    try:
        loaded = model.load_weights(ph.tensor_stream(names, distinct=True))
        error = None
    except Exception as exc:  # noqa: BLE001 - the error is part of the behaviour
        loaded, error = None, (type(exc).__name__, str(exc))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return loaded, counter.calls, state, error


def assert_same(reference, other, label):
    loaded0, calls0, state0, error0 = reference
    loaded1, calls1, state1, error1 = other
    assert error1 == error0, label
    assert loaded1 == loaded0, label
    assert calls1 == calls0, label
    assert state1.keys() == state0.keys()
    for key in state0:
        assert torch.equal(state1[key], state0[key]), (label, key)


@pytest.mark.parametrize(
    ("num_layers", "num_experts", "redundant", "order"),
    [
        (3, 512, 0, "sorted"),
        (3, 16, 0, "shuffled"),
        (3, 16, 6, "sorted"),
        (3, 16, 6, "shuffled"),
    ],
)
def test_all_switch_combinations_load_identically(
    monkeypatch, num_layers, num_experts, redundant, order
) -> None:
    qsa_ids = (0, 2)
    set_switches(monkeypatch, False, False, False)
    reference = run(num_layers, num_experts, redundant, qsa_ids, order)
    loaded, calls, state, error = reference
    assert error is None and loaded
    # QSA scales really were remapped and the experts really were called.
    assert any(k.endswith("layers.0.self_attn.k_scale") for k in loaded)
    assert len(calls) >= num_layers * num_experts * 12
    if redundant:
        assert len(calls) > num_layers * num_experts * 12
    assert state["language_model.model.layers.0.self_attn.k_scale"] != 1.0
    for combo in COMBOS:
        set_switches(monkeypatch, *combo)
        assert_same(reference, run(num_layers, num_experts, redundant, qsa_ids, order),
                    (combo, order))


def test_unknown_module_raises_the_same_error(monkeypatch) -> None:
    extra = ["model.language_model.layers.1.mlp.nonexistent.weight"]
    set_switches(monkeypatch, False, False, False)
    reference = run(2, 8, 0, (0,), "sorted", extra)
    assert reference[3] is not None and reference[3][0] == "ValueError"
    assert "nonexistent" in reference[3][1]
    for combo in COMBOS:
        set_switches(monkeypatch, *combo)
        assert_same(reference, run(2, 8, 0, (0,), "sorted", extra), combo)
