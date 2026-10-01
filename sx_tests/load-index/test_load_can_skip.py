# SPDX-License-Identifier: Apache-2.0
"""AutoWeightsLoader._can_skip without generators (SX_OPT_LOAD_CAN_SKIP).

Run (CPU is enough, no vLLM install needed; see ``load_boot.py``):

    python -m pytest -q sx_tests/load-index/test_load_can_skip.py

Asserted, with the real class cut from vllm/model_executor/models/utils.py
against a verbatim copy of the previous method (``reference_original.py``):
* identical answers (and identical exceptions, raised at the same entry) for
  random prefix / substring lists, empty lists, empty strings, duplicates, and
  lists changed after the loader was built (they are read live);
* the loader's own default skip list (rotary tables) is honoured;
* the switch off runs the two any() expressions, still identical;
* the switch is read from SX_OPT_LOAD_CAN_SKIP ("0" off, anything else on).
Expected: all pass.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import load_boot  # noqa: E402
import pipeline_harness as ph  # noqa: E402
import reference_original  # noqa: E402

NS = ph.utils_namespace()
AutoWeightsLoader = NS["AutoWeightsLoader"]
original = reference_original.can_skip_original


@pytest.fixture(params=[True, False], ids=["switch-on", "switch-off"])
def switch(request):
    saved = NS["_SX_LOAD_CAN_SKIP"]
    NS["_SX_LOAD_CAN_SKIP"] = request.param
    yield request.param
    NS["_SX_LOAD_CAN_SKIP"] = saved


def outcome(fn, loader, name):
    try:
        return ("value", fn(loader, name))
    except Exception as exc:  # noqa: BLE001 - the exception is part of the behaviour
        return ("raise", type(exc).__name__)


def test_random_lists_and_names(switch) -> None:
    rng = random.Random(5)
    words = ["visual.", "mtp.", "hashstats_", "token_lookup", "x", "ab", "a", "",
             "layers.", "rotary_emb.inv_freq", ".", "language_model.model."]
    hits = misses = 0
    for _ in range(4000):
        prefixes = [rng.choice(words) for _ in range(rng.randint(0, 3))]
        substrs = [rng.choice(words) for _ in range(rng.randint(0, 5))]
        loader = AutoWeightsLoader(
            None, skip_prefixes=prefixes or None, skip_substrs=substrs or None
        )
        for _ in range(25):
            name = ".".join(rng.choice(words + ["3", "mlp", "experts"])
                            for _ in range(rng.randint(0, 6)))
            expected = original(loader, name)
            got = loader._can_skip(name)
            assert got is expected, (prefixes, substrs, name)
            hits += expected
            misses += not expected
    assert hits > 5000 and misses > 5000


def test_default_skip_list_of_the_loader(switch) -> None:
    loader = AutoWeightsLoader(None)
    assert loader.skip_substrs == AutoWeightsLoader.ROTARY_EMBEDS_UNUSED_WEIGHTS
    for name in ["model.rotary_emb.inv_freq", "a.rotary_pos_emb.inv_freq.x",
                 "model.layers.0.weight", "rotary_emb.cos_cached", ""]:
        assert loader._can_skip(name) is original(loader, name)
    assert loader._can_skip("model.rotary_emb.sin_cached") is True
    assert loader._can_skip("model.layers.0.weight") is False


def test_lists_are_read_live(switch) -> None:
    loader = AutoWeightsLoader(None, skip_prefixes=["visual."], skip_substrs=["mtp."])
    assert loader._can_skip("language_model.x") is False
    loader.skip_prefixes.append("language_model.")
    assert loader._can_skip("language_model.x") is True
    loader.skip_substrs.append("hashstats_")
    assert loader._can_skip("layers.0.hashstats_x") is True
    loader.skip_substrs[:] = []
    assert loader._can_skip("layers.0.hashstats_x") is False
    loader.skip_prefixes = ("visual.",)  # a tuple works the same
    assert loader._can_skip("visual.x") is original(loader, "visual.x") is True


def test_bad_entries_fail_at_the_same_point(switch) -> None:
    for prefixes, substrs in [
        (["a", None], []),
        ([None], ["x"]),
        (["a"], [3, "x"]),
        ([], [None]),
        (["a", 5], ["b"]),
    ]:
        loader = AutoWeightsLoader(None)
        loader.skip_prefixes = prefixes
        loader.skip_substrs = substrs
        for name in ["a.x", "b.x", "zzz", "x", ""]:
            assert outcome(AutoWeightsLoader._can_skip, loader, name) == outcome(
                original, loader, name
            ), (prefixes, substrs, name)


def test_switch_is_read_from_the_environment(monkeypatch) -> None:
    source = load_boot.cut_assignments(load_boot.MODELS_UTILS, ["_SX_LOAD_CAN_SKIP"])
    for value, expected in [(None, True), ("1", True), ("0", False), (" 0 ", False),
                            ("", True), ("off", True)]:
        if value is None:
            monkeypatch.delenv("SX_OPT_LOAD_CAN_SKIP", raising=False)
        else:
            monkeypatch.setenv("SX_OPT_LOAD_CAN_SKIP", value)
        ns = load_boot.exec_source(source, load_boot.MODELS_UTILS)
        assert ns["_SX_LOAD_CAN_SKIP"] is expected, value
