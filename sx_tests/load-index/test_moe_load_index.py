# SPDX-License-Identifier: Apache-2.0
"""FusedMoE.load_weights: token index instead of a linear scan (SX_OPT_MOE_LOAD_INDEX).

Run (CPU is enough, no vLLM install needed; see ``load_boot.py``):

    python -m pytest -q sx_tests/load-index/test_moe_load_index.py
    python sx_tests/load-index/bench_moe_load_index.py        # old vs new timing

Asserted:
* index.matches() returns exactly the entries the original loop processes (every
  entry whose weight_name is a substring of the qualified name), in mapping
  order, each once, and the whole mapping when none matches; for mappings built
  by the real make_expert_params_mapping (512 / 8+redundant / 4 experts, the
  gate/down/up, w1/w2/w3 and linear/linear_1/linear_v schemes, with and without
  the fused 3D aliases and ``base_layer.``, the multi-scheme mapping of the
  Transformers backend) on tens of thousands of generated tensor names: valid
  expert names with every scale / zero-point suffix, ids that are prefixes of
  one another (1 / 11 / 111), other layers, names without "experts", names
  that contain a key at a place that is not a token boundary, repeated keys,
  empty qualifiers;
* the same on random mappings and names over a tiny alphabet (random
  substrings of names as keys, empty keys, glued tokens, duplicate keys): no
  false negative, ever;
* the index is built once per layer, reused, and rebuilt when
  ``expert_mapping`` is rebound or changes length; mappings that cannot be
  indexed (entries that are not 4-tuples, non-str keys, too many dot-less
  keys, unsized) give no index and the original loop runs, with its original
  exception;
* the generator end to end (the real FusedMoE.load_weights text with a fake
  weight_loader): identical event log (weights consumed, weight_loader calls
  with identical tensors and arguments, names yielded) with the switch on, off
  and against the verbatim original; the ``expert_mapping is None`` error is
  raised at the first ``next()`` as before; fused 3D tensors; redundant
  experts; failing loaders;
* the index is much faster than the scan.
Expected: all pass.
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import load_boot  # noqa: E402
import reference_original  # noqa: E402

idx = load_boot.expert_mapping_index()
make_mapping = load_boot.make_expert_params_mapping()
load_weights_new = load_boot.fused_moe_load_weights()
load_weights_old = reference_original.load_weights_original

ENV = "SX_OPT_MOE_LOAD_INDEX"


class _Model:
    """Only ``named_parameters`` is used by make_expert_params_mapping."""

    def __init__(self, base_layer: bool = False) -> None:
        self._names = ["experts.base_layer.w13_weight"] if base_layer else ["a.b"]

    def named_parameters(self):
        return [(name, None) for name in self._names]


SCHEMES = {
    "gate": ("gate_proj", "down_proj", "up_proj"),
    "w": ("w1", "w2", "w3"),
    "linear": ("linear", "linear_1", "linear_v"),
}


def build_mapping(
    scheme: str = "gate",
    num_experts: int = 512,
    redundant: int = 0,
    fused: bool = True,
    base_layer: bool = False,
):
    gate, down, up = SCHEMES[scheme]
    return make_mapping(
        _Model(base_layer), gate, down, up, num_experts, redundant, fused
    )


def transformers_style_mapping(num_experts: int = 16, redundant: int = 2):
    """The mapping of vllm/model_executor/models/transformers/moe.py."""
    mapping = [
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w13_weight", "experts.gate_up_proj", 1, "w3"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
    ]
    for scheme in SCHEMES:
        mapping.extend(build_mapping(scheme, num_experts, redundant, fused=False))
    return mapping


MAPPINGS = {
    "512-gate-fused": lambda: build_mapping("gate", 512, 0, True),
    "512-gate-nofused": lambda: build_mapping("gate", 512, 0, False),
    "512-w-fused": lambda: build_mapping("w", 512, 0, True),
    "8+4-redundant-fused": lambda: build_mapping("gate", 8, 4, True),
    "12+30-redundant": lambda: build_mapping("w", 12, 30, False),
    "4-linear": lambda: build_mapping("linear", 4, 0, True),
    "512-gate-base_layer": lambda: build_mapping("gate", 512, 0, True, True),
    "16-transformers-3-schemes": lambda: transformers_style_mapping(16, 2),
    "1-expert": lambda: build_mapping("gate", 1, 0, True),
}

LAYER_NAMES = [
    "model.language_model.model.layers.0.mlp.experts",
    "model.layers.47.mlp.experts",
    "layers.3.mlp.experts",
    "experts",
    "mlp.experts",
    "a.experts.b",
    "xexperts",
    "",
]

SUFFIXES = [
    "weight",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "weight_zero_point",
    "weight_packed",
    "weight_shape",
    "qweight",
    "scales",
    "qzeros",
    "g_idx",
    "bias",
    "k_scale",
    "",
]


def reference_matches(mapping, qual_name):
    """The original loop, reduced to its selection: all entries in mapping order."""
    out = []
    for param_name, weight_name, expert_id, shard_id in mapping:
        if weight_name not in qual_name:
            continue
        out.append((param_name, weight_name, expert_id, shard_id))
    return out


def new_matches(index, mapping, qual_name):
    """The new selection exactly as load_weights iterates it."""
    out = []
    for param_name, weight_name, expert_id, shard_id in index.matches(
        qual_name, mapping
    ):
        if weight_name not in qual_name:
            continue
        out.append((param_name, weight_name, expert_id, shard_id))
    return out


def gen_expert_names(rng: random.Random, mapping, count: int):
    """Tensor names (expert_name, layer_name) of every flavour, as the loader sees them."""
    keys = [entry[1] for entry in mapping]
    max_id = max(
        (int(k.split(".")[1]) for k in keys if k.split(".")[1].isdigit()), default=0
    )
    all_projs = sorted(
        {p for scheme in SCHEMES.values() for p in scheme} | {"gate_up_proj"}
    )
    own_projs = sorted({k.split(".")[2] for k in keys if len(k.split(".")) >= 4})
    names: list[tuple[str, str]] = []
    ids = [0, 1, 2, 10, 11, 12, 100, 111, 255, 511, 512, 513, max_id, max_id + 1, 1111]
    ids = [i for i in ids if i <= max_id + 1] + list(range(min(max_id + 1, 4)))

    def pick_proj() -> str:
        if own_projs and rng.random() < 0.7:
            return rng.choice(own_projs)
        return rng.choice(all_projs)

    for _ in range(count):
        layer = rng.choice(LAYER_NAMES)
        kind = rng.random()
        if kind < 0.55:  # a regular per-expert tensor
            e = rng.choice(ids) if rng.random() < 0.6 else rng.randint(0, max_id + 3)
            proj = pick_proj()
            suffix = rng.choice(SUFFIXES)
            base = "base_layer." if rng.random() < 0.15 else ""
            expert_name = f"{e}.{proj}.{base}{suffix}".rstrip(".")
            if rng.random() < 0.05:
                expert_name = "experts." + expert_name  # ckpt names that keep "experts."
        elif kind < 0.65:  # fused 3D aliases and lookalikes
            expert_name = rng.choice(
                [
                    "gate_up_proj",
                    "down_proj",
                    "gate_up_proj_bias",
                    "gate_up_proj.weight",
                    "down_proj.weight_scale",
                    "gate_up_projx",
                    "xgate_up_proj",
                    "experts.gate_up_proj",
                    "experts.down_proj",
                    "experts.down_proj_scale",
                ]
            )
        elif kind < 0.75:  # not an expert tensor at all
            expert_name = rng.choice(
                [
                    "gate.weight",
                    "shared_expert.gate_proj.weight",
                    "e_score_correction_bias",
                    "norm.weight",
                    "w13_weight",
                    "mlp.gate_proj.weight",
                    "11.up_proj",
                    "",
                    ".",
                    "..",
                    "experts",
                    "experts.",
                    "experts.1",
                    "experts.1.",
                    "experts..gate_proj.",
                    "12gate_proj.weight",
                ]
            )
        elif kind < 0.85:  # a key at a non-boundary position, glued or repeated
            e = rng.choice(ids)
            proj = pick_proj()
            expert_name = rng.choice(
                [
                    f"x{layer}.{e}.{proj}.weight",
                    f"experts{e}.{proj}.weight",
                    f"sub.experts.{e}.{proj}.weight",
                    f"experts.{e}.{proj}.weight.experts.{e + 1}.{proj}.weight",
                    f"experts.{e}.{proj}.experts.{e}.{proj}.",
                    f"experts.{e}.{proj}x.weight",
                    f"experts.{e}x.{proj}.weight",
                    f"xexperts.{e}.{proj}.weight",
                    f"experts.{e}.{proj}",
                    f"experts.{e}.{proj}.",
                ]
            )
        else:  # prefix ids: 1 / 11 / 111 in every position
            a, b = rng.sample([1, 11, 111, 0, 10, 110], 2)
            proj = pick_proj()
            expert_name = rng.choice(
                [
                    f"{a}.{proj}.weight",
                    f"{a}.{b}.{proj}.weight",
                    f"{a}{b}.{proj}.weight",
                    f"experts.{a}.{b}.{proj}.weight",
                ]
            )
        names.append((expert_name, layer))
    return names


# ---------------------------------------------------------------- selection


@pytest.mark.parametrize("name", sorted(MAPPINGS))
def test_selection_equals_the_scan_on_generated_names(name: str) -> None:
    mapping = MAPPINGS[name]()
    index = idx.build_expert_mapping_index(mapping)
    assert index is not None
    rng = random.Random(f"select-{name}")
    names = gen_expert_names(rng, mapping, 6000)
    # Positive examples built from the mapping itself: every key as a tensor
    # name, with and without a following token.
    for key in rng.sample([e[1] for e in mapping], min(300, len(mapping))):
        names.append((key + "weight", rng.choice(LAYER_NAMES)))
        names.append((key.rstrip(".").split("experts.", 1)[-1] + ".weight_scale", "m.experts"))
    seen_match = seen_none = 0
    for expert_name, layer in names:
        qual = f"{layer}.{expert_name}"
        expected = reference_matches(mapping, qual)
        got = new_matches(index, mapping, qual)
        assert got == expected, qual
        raw = index.matches(qual, mapping)
        if expected:
            seen_match += 1
            # the raw selection is exactly the matches, in mapping order, once each
            assert [e for e in raw] == expected, qual
        else:
            seen_none += 1
            assert raw is mapping, qual  # fallback: the caller scans everything
    assert seen_match > 150 and seen_none > 150  # both branches really ran


def test_selection_keeps_original_order_for_several_matches() -> None:
    # Fused aliases: both gate_up entries (w1 shard 0, w3 shard 1) match one name.
    mapping = build_mapping("gate", 512, 0, True)
    index = idx.build_expert_mapping_index(mapping)
    got = index.matches("m.experts.gate_up_proj", mapping)
    assert [(e[2], e[3]) for e in got] == [(0, "w1"), (1, "w3")]
    # Redundant experts: logical expert 3 appears at physical ids 3, 11 and 19.
    mapping = build_mapping("gate", 8, 16, False)
    index = idx.build_expert_mapping_index(mapping)
    got = index.matches("m.experts.3.gate_proj.weight", mapping)
    assert [e[2] for e in got] == [3, 11, 19]
    assert got == reference_matches(mapping, "m.experts.3.gate_proj.weight")
    # A name that contains two keys: both entries, in mapping order, not name order.
    name = "m.experts.5.up_proj.weight.experts.2.gate_proj.weight"
    got = index.matches(name, mapping)
    assert got == reference_matches(mapping, name)
    assert [e[2] for e in got] == [2, 5, 10, 13, 18, 21]  # mapping order, not name order
    positions = [mapping.index(e) for e in got]
    assert positions == sorted(positions)


def test_selection_non_boundary_and_prefix_ids() -> None:
    mapping = build_mapping("gate", 512, 0, True)
    index = idx.build_expert_mapping_index(mapping)
    for qual, count in [
        ("xexperts.1.gate_proj.weight", 1),  # key starts inside a token
        ("experts.11.gate_proj.weight", 1),  # 11 is not 1
        ("experts.111.gate_proj.weight", 1),
        ("experts.1.gate_proj", 0),  # needs the trailing dot
        ("experts.1.gate_proj.", 1),
        ("experts.1.gate_projx.weight", 0),
        ("experts.1x.gate_proj.weight", 0),
        ("experts.gate_up_proj_bias", 2),  # fused alias, key continues in the name
        ("experts.down_proj.weight", 1),
        ("experts.512.gate_proj.weight", 0),  # out of range
        ("", 0),
        (".", 0),
    ]:
        got = index.matches(qual, mapping)
        expected = reference_matches(mapping, qual)
        assert (got if got is not mapping else []) == expected, qual
        assert len(expected) == count, qual


def _random_case(rng: random.Random):
    tokens = ["experts", "e", "1", "11", "111", "gate", "up", "w", "x", "", "ab", "a"]
    glue = ["", "", "", "x", "1", "_"]

    def make_name() -> str:
        parts = [rng.choice(tokens) + rng.choice(glue) for _ in range(rng.randint(0, 7))]
        return ".".join(parts)

    pool = [make_name() for _ in range(rng.randint(1, 8))]
    keys: list[str] = []
    for _ in range(rng.randint(1, 60)):
        if rng.random() < 0.6:  # a substring of a name: starts/ends anywhere
            name = rng.choice(pool)
            a = rng.randint(0, len(name))
            b = rng.randint(a, len(name))
            keys.append(name[a:b])
        else:
            parts = [rng.choice(tokens) for _ in range(rng.randint(1, 5))]
            keys.append(".".join(parts) + rng.choice(["", ".", "."]))
    mapping = [(f"p{i}", key, i % 5, "w1") for i, key in enumerate(keys)]
    names = pool + [make_name() for _ in range(40)]
    return mapping, names


def test_selection_has_no_false_negatives_on_random_mappings() -> None:
    rng = random.Random(20260930)
    checked = hits = 0
    for _ in range(3000):
        mapping, names = _random_case(rng)
        index = idx.build_expert_mapping_index(mapping)
        if index is None:  # too many dot-less keys: the original loop runs
            continue
        for name in names:
            expected = reference_matches(mapping, name)
            raw = index.matches(name, mapping)
            checked += 1
            if expected:
                hits += 1
                assert raw == expected, (mapping, name)
            else:
                assert raw is mapping, (mapping, name)
    assert checked > 100_000 and hits > 20_000


def test_index_is_not_built_when_it_would_not_narrow() -> None:
    dotless = [(f"p{i}", f"key{i}", i, "w1") for i in range(65)]
    assert idx.build_expert_mapping_index(dotless) is None
    ok = [(f"p{i}", f"key{i}", i, "w1") for i in range(64)]
    index = idx.build_expert_mapping_index(ok)
    assert index is not None
    assert reference_matches(ok, "m.key7.x") == new_matches(index, ok, "m.key7.x")
    # Entries that the original loop could not unpack, non-str keys, unsized.
    assert idx.build_expert_mapping_index([("a", "k.1.2.", 0)]) is None
    assert idx.build_expert_mapping_index([("a", None, 0, "w1")]) is None
    assert idx.build_expert_mapping_index([("a", b"k.1.2.", 0, "w1")]) is None
    assert idx.build_expert_mapping_index(5) is None
    owner = type("O", (), {})()
    assert idx.get_expert_mapping_index(owner, iter([])) is None  # no len()


# -------------------------------------------------------------------- cache


def test_index_cached_on_the_layer_and_invalidated_on_change() -> None:
    owner = type("Layer", (), {})()
    mapping = build_mapping("gate", 8, 0, True)
    first = idx.get_expert_mapping_index(owner, mapping)
    assert first is not None and idx.get_expert_mapping_index(owner, mapping) is first
    # An equal but different list is a rebinding: rebuilt (and still correct).
    other = list(mapping)
    second = idx.get_expert_mapping_index(owner, other)
    assert second is not first and idx.get_expert_mapping_index(owner, other) is second
    # Rebinding to a mapping with other keys answers for the new keys.
    w_mapping = build_mapping("w", 8, 0, True)
    third = idx.get_expert_mapping_index(owner, w_mapping)
    assert third.matches("m.experts.3.w1.weight", w_mapping) != w_mapping
    assert new_matches(third, w_mapping, "m.experts.3.w1.weight") == reference_matches(
        w_mapping, "m.experts.3.w1.weight"
    )
    # Growing the same list object drops the cache too.
    grown = list(mapping)
    index = idx.get_expert_mapping_index(owner, grown)
    grown.extend(build_mapping("w", 8, 0, False))
    index2 = idx.get_expert_mapping_index(owner, grown)
    assert index2 is not index
    qual = "m.experts.3.w1.weight"
    assert new_matches(index2, grown, qual) == reference_matches(grown, qual) != []
    # An unindexable mapping is remembered as such (no rebuild per call).
    bad = [("a", None, 0, "w1")]
    assert idx.get_expert_mapping_index(owner, bad) is None
    assert owner._sx_expert_mapping_index[0] is bad


def test_switch_reads_the_environment_per_call(monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert idx.moe_load_index_enabled() is True
    for value, expected in [("1", True), ("0", False), (" 0 ", False), ("", True),
                            ("off", True), ("true", True)]:
        monkeypatch.setenv(ENV, value)
        assert idx.moe_load_index_enabled() is expected, value


# ------------------------------------------------------------ end to end


class FakeMoE:
    """What FusedMoE.load_weights touches: names, parameters, weight_loader."""

    def __init__(self, layer_name: str, mapping, fail: str = "mod5"):
        self.layer_name = layer_name
        self.expert_mapping = mapping
        self.events: list = []
        self.fail = fail

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return ("param", name)

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                      return_success):
        ok = (expert_id + len(weight_name)) % 5 != 0 if self.fail == "mod5" else True
        self.events.append(
            (
                "load",
                param,
                weight_name,
                shard_id,
                expert_id,
                return_success,
                tuple(loaded_weight.shape),
                loaded_weight.data_ptr(),
                ok,
            )
        )
        return ok


def run_loader(fn, layer_name, mapping, weights, fail="mod5"):
    """Event log: ("consume", name) / ("load", ...) / ("yield", name) in order."""
    moe = FakeMoE(layer_name, mapping, fail)

    def source():
        for name, tensor in weights:
            moe.events.append(("consume", name))
            yield name, tensor

    try:
        for param in fn(moe, source()):
            moe.events.append(("yield", param))
    except Exception as exc:  # noqa: BLE001 - the error is part of the behaviour
        moe.events.append(("raise", type(exc).__name__, str(exc)))
    return moe.events


def make_tensors(rng: random.Random, names_layers, mapping_has_fused: bool):
    out = []
    for expert_name, _layer in names_layers:
        fused_like = "gate_up_proj" in expert_name or "down_proj" in expert_name
        if fused_like and "." not in expert_name.replace("experts.", "", 1) and rng.random() < 0.8:
            t = torch.randn(3, 4, 2)  # fused 3D: E=3, chunked in two along dim 1
        elif rng.random() < 0.3:
            t = torch.randn(4, 2)
        else:
            t = torch.randn(3)
        out.append(t)
    return out


@pytest.mark.parametrize("name", sorted(MAPPINGS))
def test_generator_identical_with_switch_on_off_and_to_the_original(
    name: str, monkeypatch
) -> None:
    mapping = MAPPINGS[name]()
    rng = random.Random(f"e2e-{name}")
    # One layer name per run (load_weights sees one module); several runs.
    total_loads = total_yields = 0
    for layer in LAYER_NAMES:
        pairs = gen_expert_names(rng, mapping, 250)
        tensors = make_tensors(rng, pairs, True)
        weights = [(en, t) for (en, _), t in zip(pairs, tensors, strict=True)]
        logs = {}
        for label, fn, env in [
            ("original", load_weights_old, None),
            ("off", load_weights_new, "0"),
            ("on", load_weights_new, "1"),
            ("on-again", load_weights_new, "1"),
        ]:
            if env is None:
                monkeypatch.delenv(ENV, raising=False)
            else:
                monkeypatch.setenv(ENV, env)
            # the same mapping object each time: "on-again" reuses the cached index
            logs[label] = run_loader(fn, layer, mapping, weights)
        assert logs["off"] == logs["original"], (name, layer)
        assert logs["on"] == logs["original"], (name, layer)
        assert logs["on-again"] == logs["original"], (name, layer)
        total_loads += sum(1 for e in logs["original"] if e[0] == "load")
        total_yields += sum(1 for e in logs["original"] if e[0] == "yield")
    assert total_loads > 300 and 0 < total_yields < total_loads  # failing loaders too


def test_generator_raises_the_same_error_at_the_same_point(monkeypatch) -> None:
    """A 3D tensor under a per-expert name: shard_idx = expert_id overruns chunk(2)."""
    mapping = build_mapping("gate", 8, 0, True)
    weights = [
        ("0.gate_proj.weight", torch.randn(3)),
        ("1.up_proj.weight", torch.randn(3, 4, 2)),  # expert_id 1: fine
        ("2.down_proj.weight", torch.randn(3, 4, 2)),  # w2: whole tensor, fine
        ("3.gate_proj.weight", torch.randn(3, 4, 2)),  # w1 with expert_id 3: IndexError
        ("4.gate_proj.weight", torch.randn(3)),
    ]
    logs = []
    for fn, env in [(load_weights_old, "1"), (load_weights_new, "0"),
                    (load_weights_new, "1")]:
        monkeypatch.setenv(ENV, env)
        logs.append(run_loader(fn, "m.experts", mapping, weights, fail="none"))
    assert logs[0] == logs[1] == logs[2]
    assert logs[0][-1][:2] == ("raise", "IndexError")
    assert sum(1 for e in logs[0] if e[0] == "consume") == 4  # stopped at the 4th


def test_generator_with_fused_3d_weights(monkeypatch) -> None:
    mapping = build_mapping("gate", 512, 0, True)
    gate_up = torch.randn(4, 6, 5)
    down = torch.randn(4, 5, 3)
    weights = [
        ("gate_up_proj", gate_up),
        ("down_proj", down),
        ("0.gate_proj.weight", torch.randn(6, 5)),
        ("gate_up_proj_scale", torch.randn(4, 2, 5)),
    ]
    for env in ("0", "1"):
        monkeypatch.setenv(ENV, env)
        log = run_loader(load_weights_new, "model.layers.1.mlp.experts", mapping,
                         weights, fail="none")
        monkeypatch.delenv(ENV, raising=False)
        expected = run_loader(load_weights_old, "model.layers.1.mlp.experts", mapping,
                              weights, fail="none")
        assert log == expected
    loads = [e for e in expected if e[0] == "load"]
    # gate_up_proj: 4 experts for shard w1 and 4 for w3 (chunked along dim 1)
    assert [e[3] for e in loads[:8]] == ["w1"] * 4 + ["w3"] * 4
    assert [e[4] for e in loads[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_generator_none_mapping_raises_at_first_next(monkeypatch) -> None:
    for env in ("0", "1"):
        monkeypatch.setenv(ENV, env)
        moe = FakeMoE("m.experts", None)
        gen = load_weights_new(moe, iter([("0.gate_proj.weight", torch.zeros(1))]))
        # a generator: nothing happens (and nothing is consumed) until next()
        with pytest.raises(ValueError, match="expert_mapping"):
            next(gen)
    moe = FakeMoE("m.experts", None)
    with pytest.raises(ValueError, match="expert_mapping"):
        next(load_weights_old(moe, iter([])))


def test_generator_is_lazy_and_consumes_the_input_one_tensor_at_a_time(
    monkeypatch,
) -> None:
    mapping = build_mapping("gate", 8, 0, False)
    consumed: list[str] = []

    def source():
        for i in range(4):
            consumed.append(str(i))
            yield f"{i}.gate_proj.weight", torch.zeros(2)

    monkeypatch.setenv(ENV, "1")
    moe = FakeMoE("m.experts", mapping, fail="none")
    gen = load_weights_new(moe, source())
    assert consumed == []
    assert next(gen) == "w13_weight"
    assert consumed == ["0"]
    assert next(gen) == "w13_weight"
    assert consumed == ["0", "1"]


def test_generator_original_error_for_a_bad_mapping(monkeypatch) -> None:
    """A mapping the original loop chokes on is not indexed: same exception."""
    weights = [("0.gate_proj.weight", torch.zeros(2))]
    for bad, exc in [
        ([("a", None, 0, "w1")], TypeError),
        ([("a", "experts.0.gate_proj.", 0)], ValueError),
        (5, TypeError),
    ]:
        monkeypatch.setenv(ENV, "1")
        errors = []
        for fn in (load_weights_old, load_weights_new):
            moe = FakeMoE("m.experts", bad, fail="none")
            with pytest.raises(exc) as info:
                list(fn(moe, iter(weights)))
            errors.append((type(info.value), str(info.value)))
        assert errors[0] == errors[1]


def test_generator_partial_progress_before_a_bad_entry_matches(monkeypatch) -> None:
    """Entries before a bad one are processed first, as in the original loop."""
    good = build_mapping("gate", 2, 0, False)
    bad = good[:3] + [("a", None, 0, "w1")] + good[3:]
    weights = [("0.gate_proj.weight", torch.zeros(2))]
    logs = []
    for fn, env in [(load_weights_old, "1"), (load_weights_new, "1"),
                    (load_weights_new, "0")]:
        monkeypatch.setenv(ENV, env)
        moe = FakeMoE("m.experts", bad, fail="none")
        with pytest.raises(TypeError):
            for param in fn(moe, iter(weights)):
                moe.events.append(("yield", param))
        logs.append(moe.events)
    assert logs[0] == logs[1] == logs[2] and logs[0]


# ------------------------------------------- a real nn.Module (repo test case)


def _fused_experts_module(layer_name: str = "model.layers.0.mlp.experts"):
    """tests/models/qwen4_exp/test_weight_loading.py::
    test_fused_mtp_expert_checkpoint_loads_every_expert, with the cut-out
    FusedMoE.load_weights / make_expert_params_mapping."""
    import torch.nn as nn

    class FakeFusedExperts(nn.Module):
        load_weights = load_weights_new

        def __init__(self) -> None:
            super().__init__()
            self.layer_name = layer_name
            self.w13_weight = nn.Parameter(torch.empty(1))
            self.w2_weight = nn.Parameter(torch.empty(1))
            self.calls: list = []
            self.expert_mapping = make_mapping(self, "gate_proj", "down_proj",
                                               "up_proj", 2, 0, True)

        def weight_loader(self, *, param, loaded_weight, weight_name, shard_id,
                          expert_id, return_success):
            assert return_success
            param_name = "w13" if param is self.w13_weight else "w2"
            self.calls.append((param_name, shard_id, expert_id, loaded_weight.clone()))
            return True

    return FakeFusedExperts()


@pytest.mark.parametrize("env", ["1", "0"])
def test_repo_case_fused_mtp_checkpoint_on_a_real_module(monkeypatch, env) -> None:
    monkeypatch.setenv(ENV, env)
    experts = _fused_experts_module()
    gate_up = torch.arange(2 * 6 * 2).reshape(2, 6, 2)
    down = torch.arange(2 * 2 * 3).reshape(2, 2, 3)
    loaded = list(experts.load_weights([("gate_up_proj", gate_up), ("down_proj", down)]))
    assert loaded == ["w13_weight"] * 4 + ["w2_weight"] * 2
    assert [(n, s, e) for n, s, e, _ in experts.calls] == [
        ("w13", "w1", 0), ("w13", "w1", 1), ("w13", "w3", 0), ("w13", "w3", 1),
        ("w2", "w2", 0), ("w2", "w2", 1),
    ]
    torch.testing.assert_close(experts.calls[0][3], gate_up[0, :3])
    torch.testing.assert_close(experts.calls[3][3], gate_up[1, 3:])
    torch.testing.assert_close(experts.calls[5][3], down[1])
    if env == "1":  # the index was built on the module and survives a deepcopy
        import copy

        assert experts.expert_mapping and experts._sx_expert_mapping_index[2] is not None
        clone = copy.deepcopy(experts)
        cached = clone._sx_expert_mapping_index
        assert cached[0] is clone.expert_mapping and cached[0] is not experts.expert_mapping
        assert list(clone.load_weights([("down_proj", down)])) == ["w2_weight"] * 2
        assert clone._sx_expert_mapping_index is cached  # reused, not rebuilt


def test_mapping_of_the_repo_test_is_what_the_helper_builds() -> None:
    assert build_mapping("gate", 2, 0, True) == [
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w13_weight", "experts.gate_up_proj", 1, "w3"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
        ("experts.w2_", "experts.1.down_proj.", 1, "w2"),
        ("experts.w13_", "experts.1.up_proj.", 1, "w3"),
    ]


# ---------------------------------------------------------------- speed


def test_index_is_much_faster_than_the_scan() -> None:
    mapping = build_mapping("gate", 512, 0, True)
    index = idx.build_expert_mapping_index(mapping)
    rng = random.Random(1)
    quals = [
        f"model.layers.{rng.randrange(48)}.mlp.experts.{rng.randrange(512)}."
        f"{rng.choice(SCHEMES['gate'])}.{rng.choice(SUFFIXES[:4])}"
        for _ in range(1500)
    ]
    start = time.perf_counter()
    old = [reference_matches(mapping, q) for q in quals[:150]]
    t_old = (time.perf_counter() - start) / 150
    start = time.perf_counter()
    new = [new_matches(index, mapping, q) for q in quals]
    t_new = (time.perf_counter() - start) / len(quals)
    assert new[:150] == old
    assert t_old / t_new > 8, (t_old, t_new)
