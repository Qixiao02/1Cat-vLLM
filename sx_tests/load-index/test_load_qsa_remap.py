# SPDX-License-Identifier: Apache-2.0
"""_remap_qsa_cache_scale_name: early return for names that cannot be remapped
(SX_OPT_LOAD_QSA_REMAP).

Run (CPU is enough, no vLLM install needed; see ``load_boot.py``):

    python -m pytest -q sx_tests/load-index/test_load_qsa_remap.py

Asserted, with the real function cut from vllm/models/qwen4_exp/nvidia/model.py
against a verbatim copy of the previous one (``reference_original.py``):
* identical result on the cases of tests/models/qwen4_exp/test_weight_loading.py;
* identical result on every table key at every position (boundary and
  non-boundary markers, several markers in one name, ids in and out of the
  layer set, the empty set) and on 200k random names;
* the switch off runs the per-layer scan for every name, still identical;
* the switch is read from SX_OPT_LOAD_QSA_REMAP ("0" off, anything else on).
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

NS = ph.model_namespace()
remap = NS["_remap_qsa_cache_scale_name"]
original = reference_original.remap_qsa_cache_scale_name_original
KEYS = list(NS["_QSA_CACHE_SCALE_SUFFIXES"])


@pytest.fixture(params=[True, False], ids=["switch-on", "switch-off"])
def switch(request):
    saved = NS["_SX_LOAD_QSA_REMAP"]
    NS["_SX_LOAD_QSA_REMAP"] = request.param
    yield request.param
    NS["_SX_LOAD_QSA_REMAP"] = saved


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("layers.0.self_attn.k_proj.k_scale", "layers.0.self_attn.k_scale"),
        ("layers.0.self_attn.v_proj.output_scale", "layers.0.self_attn.v_scale"),
        (
            "language_model.model.layers.0.self_attn.attn.k_scale",
            "language_model.model.layers.0.self_attn.k_scale",
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
        ),
        ("layers.1.self_attn.k_proj.k_scale", "layers.1.self_attn.k_proj.k_scale"),
    ],
)
def test_cases_of_the_repo_test_file(switch, name: str, expected: str) -> None:
    assert remap(name, frozenset({0})) == expected == original(name, frozenset({0}))


def test_every_key_at_every_position(switch) -> None:
    ids_sets = [frozenset(), frozenset({0}), frozenset({3}), frozenset({0, 3, 12}),
                frozenset(range(48))]
    prefixes = ["", "language_model.model.", "x", "a.", "xlayers.9.self_attn.",
                "layers.3.self_attn.", "layers.12.self_attn.layers.3.self_attn."]
    changed = 0
    for key in KEYS:
        for layer in (0, 3, 12, 40):
            for prefix in prefixes:
                for tail in ("", "x", ".x", "_", ".bias"):
                    for glue in ("layers.", "xlayers.", "_layers.", "."):
                        name = f"{prefix}{glue}{layer}.self_attn.{key}{tail}"
                        for ids in ids_sets:
                            got = remap(name, ids)
                            assert got == original(name, ids), (name, ids)
                            changed += got != name
    assert changed > 300  # the interesting branch ran, not just the early return


def test_several_markers_in_one_name(switch) -> None:
    for name in [
        "layers.1.self_attn.layers.2.self_attn.k_scale",
        "layers.2.self_attn.layers.1.self_attn.k_scale",
        "xlayers.1.self_attn.layers.1.self_attn.k_scale",  # first hit not on a boundary
        "layers.1.self_attn.k_proj.k_scale.layers.1.self_attn.k_proj.k_scale",
        "layers.1.self_attn.layers.1.self_attn.v_proj.output_scale",
    ]:
        for ids in (frozenset({1}), frozenset({2}), frozenset({1, 2}),
                    frozenset({2, 1, 7})):
            assert remap(name, ids) == original(name, ids), (name, ids)


def test_random_names(switch) -> None:
    rng = random.Random(77)
    tokens = ["language_model", "model", "layers", "self_attn", "k_proj", "v_proj",
              "attn", "k_scale", "_k_scale", "v_scale", "_v_scale", "output_scale",
              "indexer", "weight_scale", "weight_scale_2", "weight", "input_scale",
              "mlp", "experts", "x", "1", "3", "12", "gate_proj", "_", ""]
    changed = 0
    for _ in range(200_000):
        parts = []
        for _ in range(rng.randint(1, 9)):
            token = rng.choice(tokens)
            if token == "layers" and rng.random() < 0.7:
                parts += ["layers", rng.choice(["0", "1", "3", "12", "40", "x"]),
                          "self_attn"]
            else:
                parts.append(token)
        name = ".".join(parts)
        if rng.random() < 0.5:  # a name close to a real cache-scale tensor
            layer = rng.choice(["0", "1", "3", "12", "40", "x"])
            name = (
                rng.choice(["", "language_model.model.", "m.", "x"])
                + rng.choice(["layers.", "layers.", "xlayers."])
                + f"{layer}.self_attn."
                + rng.choice(KEYS + [".".join(parts), "k_proj", "indexer.weight"])
                + rng.choice(["", "", "", "x", ".x"])
            )
        if rng.random() < 0.1:
            name = rng.choice(["x", "_"]) + name
        ids = frozenset(rng.sample(range(0, 14), rng.randint(0, 5)))
        got = remap(name, ids)
        assert got == original(name, ids), (name, ids)
        changed += got != name
    assert changed > 1000


def test_expert_names_return_at_once() -> None:
    """The point of the switch: no per-layer work for names like these."""
    ids = frozenset(range(1, 48, 4))
    for name in [
        "layers.3.mlp.experts.12.gate_proj.weight",
        "layers.3.mlp.experts.12.gate_proj.weight_scale",
        "layers.3.mlp.experts.12.gate_proj.weight_scale_2",
        "layers.3.mlp.experts.12.gate_proj.input_scale",
        "layers.3.self_attn.indexer.index_qk_proj.weight_scale",
    ]:
        assert not name.endswith(tuple(KEYS))
        assert remap(name, ids) is name or remap(name, ids) == name


def test_switch_is_read_from_the_environment(monkeypatch) -> None:
    source = load_boot.cut_assignments(load_boot.QWEN4_MODEL, ["_SX_LOAD_QSA_REMAP"])
    for value, expected in [(None, True), ("1", True), ("0", False), (" 0 ", False),
                            ("", True), ("off", True)]:
        if value is None:
            monkeypatch.delenv("SX_OPT_LOAD_QSA_REMAP", raising=False)
        else:
            monkeypatch.setenv("SX_OPT_LOAD_QSA_REMAP", value)
        ns = load_boot.exec_source(source, load_boot.QWEN4_MODEL)
        assert ns["_SX_LOAD_QSA_REMAP"] is expected, value
