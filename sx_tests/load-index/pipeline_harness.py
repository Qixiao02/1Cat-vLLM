# SPDX-License-Identifier: Apache-2.0
"""A synthetic Qwen4Exp weight-loading pipeline built from the real code.

The three nested loaders of the Swift-1.5 model (outer ``Qwen4ExpFor
ConditionalGeneration`` -> ``Qwen4ExpForCausalLM`` -> ``Qwen4ExpModel``), each
an ``AutoWeightsLoader`` with its mapper, skip lists and ignored suffixes, the
QSA scale remap and ``maybe_fuse_shared_experts`` in ``Qwen4ExpModel.load_
weights``, and ``FusedMoE.load_weights`` for ``layers.N.mlp.experts`` - all cut
from the real source files by ``load_boot`` (nothing is copied by hand). The
modules, parameters and ``weight_loader`` are toy ones; only the Python that
runs once per checkpoint tensor is the real thing.

Used by the pipeline equivalence tests (every ``SX_OPT_LOAD_*`` switch on/off
gives the same loaded names and the same loader calls) and by
``bench_load_pipeline.py`` (cost per tensor of each stage).

Not a test file.
"""

from __future__ import annotations

import itertools
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

import load_boot

_CACHE: dict[str, Any] = {}

IGNORED_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]


class StageMissingLayer(nn.Module):
    pass


class PPMissingLayer(nn.Identity):
    pass


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight.reshape(param.shape))


def _identity_decorator(fn):
    return fn


def utils_namespace() -> dict[str, Any]:
    """WeightsMapper, AutoWeightsLoader, maybe_fuse_shared_experts, maybe_prefix."""
    if "utils" in _CACHE:
        return _CACHE["utils"]
    path = load_boot.MODELS_UTILS
    source = "\n\n".join(
        [
            load_boot.cut_assignments(path, ["_SX_LOAD_CAN_SKIP"]),
            load_boot.cut_function(path, "maybe_prefix"),
            "@dataclass\n" + load_boot.cut_class(path, "WeightsMapper").replace(
                "@dataclass\n", "", 1
            ),
            load_boot.cut_class(path, "AutoWeightsLoader"),
            load_boot.cut_function(path, "maybe_fuse_shared_experts"),
        ]
    )
    ns = load_boot.exec_source(
        source,
        path,
        dataclass=dataclass,
        field=field,
        itertools=itertools,
        re=re,
        nn=nn,
        Callable=Callable,
        Literal=Literal,
        StageMissingLayer=StageMissingLayer,
        PPMissingLayer=PPMissingLayer,
        default_weight_loader=default_weight_loader,
        support_quantized_model_reload_from_hp_weights=_identity_decorator,
    )
    _CACHE["utils"] = ns
    return ns


def model_namespace() -> dict[str, Any]:
    """_remap_qsa_cache_scale_name and the Qwen4Exp mappers."""
    if "model" in _CACHE:
        return _CACHE["model"]
    path = load_boot.QWEN4_MODEL
    utils = utils_namespace()
    source = "\n\n".join(
        [
            load_boot.cut_assignments(
                path,
                [
                    "_QWEN3_5_WEIGHTS_MAPPER",
                    "_HC_WEIGHTS_MAPPER",
                    "_SX_LOAD_QSA_REMAP",
                    "_QSA_CACHE_SCALE_SUFFIXES",
                    "_QSA_CACHE_SCALE_SUFFIX_KEYS",
                ],
            ),
            load_boot.cut_function(path, "_remap_qsa_cache_scale_name"),
        ]
    )
    ns = load_boot.exec_source(source, path, WeightsMapper=utils["WeightsMapper"])
    _CACHE["model"] = ns
    return ns


# ------------------------------------------------------------------ the model


class Counter:
    """What the toy weight loaders record (kept tiny: this is a hot path)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.record = True
        self.record_values = False


class FakeExperts(nn.Module):
    """FusedMoE stand-in: real ``load_weights``, toy ``weight_loader``."""

    def __init__(self, layer_name: str, mapping, counter: Counter) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.expert_mapping = mapping
        self._counter = counter
        for proj in ("w13", "w2"):
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                self.register_parameter(
                    f"{proj}_{suffix}",
                    nn.Parameter(torch.zeros(1), requires_grad=False),
                )

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                      return_success):
        counter = self._counter
        if counter.record:
            entry = (weight_name, shard_id, expert_id, tuple(loaded_weight.shape))
            if counter.record_values:
                entry += (float(loaded_weight.reshape(-1)[0]),)
            counter.calls.append(entry)
        return True


def _bind_fused_moe_load_weights() -> None:
    FakeExperts.load_weights = load_boot.fused_moe_load_weights()


class ScaleHolder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.v_scale = nn.Parameter(torch.ones(()), requires_grad=False)


class Layer(nn.Module):
    def __init__(self, idx: int, prefix: str, mapping, counter: Counter,
                 with_attention: bool) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(2)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(2, 2, bias=False)
        self.mlp.experts = FakeExperts(f"{prefix}.layers.{idx}.mlp.experts", mapping,
                                       counter)
        if with_attention:
            self.self_attn = ScaleHolder()


class Inner(nn.Module):
    """Qwen4ExpModel: remap -> fuse -> AutoWeightsLoader(mapper)."""

    def __init__(self, num_layers: int, mapping, counter: Counter, qsa_ids,
                 skip_substrs: list[str], prefix: str) -> None:
        super().__init__()
        self._qsa_layer_ids = frozenset(qsa_ids)
        self._skip_substrs = skip_substrs
        self.layers = nn.ModuleList(
            Layer(i, prefix, mapping, counter, i in self._qsa_layer_ids)
            for i in range(num_layers)
        )
        self.embed_tokens = nn.Embedding(2, 2)
        self.config = type("C", (), {"num_experts": 8})()

    def load_weights(self, weights):
        U, M = utils_namespace(), model_namespace()
        remap = M["_remap_qsa_cache_scale_name"]
        weights = (
            (remap(name, self._qsa_layer_ids), weight) for name, weight in weights
        )
        weights = U["maybe_fuse_shared_experts"](
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
            enabled=False,
        )
        loader = U["AutoWeightsLoader"](
            self,
            skip_substrs=list(self._skip_substrs),
            ignore_unexpected_suffixes=IGNORED_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=M["_QWEN3_5_WEIGHTS_MAPPER"]
                                   | M["_HC_WEIGHTS_MAPPER"])


class Causal(nn.Module):
    """Qwen4ExpForCausalLM."""

    def __init__(self, inner: Inner) -> None:
        super().__init__()
        self.model = inner
        self.lm_head = nn.Linear(2, 2, bias=False)

    def load_weights(self, weights):
        U = utils_namespace()
        loader = U["AutoWeightsLoader"](
            self, skip_substrs=["mtp."], ignore_unexpected_suffixes=IGNORED_SUFFIXES.copy()
        )
        return loader.load_weights(
            weights,
            mapper=U["WeightsMapper"](
                orig_to_new_prefix={"model.language_model.": "model."}
            ),
        )


class Outer(nn.Module):
    """Qwen4ExpForConditionalGeneration (language-model-only)."""

    def __init__(self, causal: Causal) -> None:
        super().__init__()
        self.language_model = causal
        self.visual = StageMissingLayer()

    def load_weights(self, weights):
        U = utils_namespace()
        loader = U["AutoWeightsLoader"](
            self,
            skip_prefixes=["visual."],
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=IGNORED_SUFFIXES.copy(),
        )
        return loader.load_weights(
            weights,
            mapper=U["WeightsMapper"](
                orig_to_new_prefix={
                    "model.visual.": "visual.",
                    "lm_head.": "language_model.lm_head.",
                    "model.language_model.": "language_model.model.",
                }
            ),
        )


def build_model(num_layers: int, mapping_factory: Callable[[], Any],
                qsa_ids=(), skip_substrs=None):
    """Outer model; ``mapping_factory`` is called once per layer (own list each)."""
    _bind_fused_moe_load_weights()
    counter = Counter()
    prefix = "language_model.model"
    inner = Inner(
        num_layers,
        None,
        counter,
        qsa_ids,
        skip_substrs
        or ["hashstats_", "token_lookup", "hyper_connection_mixer.block_inject_weight"],
        prefix,
    )
    for layer in inner.layers:
        layer.mlp.experts.expert_mapping = mapping_factory()
    return Outer(Causal(inner)), counter


def checkpoint_names(num_layers: int, num_experts: int, qsa_ids=(),
                     extra: bool = True) -> Iterable[str]:
    """Checkpoint tensor names in the order of a sorted safetensors shard."""
    base = "model.language_model.layers"
    for layer in range(num_layers):
        if extra:
            yield f"{base}.{layer}.input_layernorm.weight"
            if layer in qsa_ids:
                yield f"{base}.{layer}.self_attn.k_proj.k_scale"
                yield f"{base}.{layer}.self_attn.v_proj.v_scale"
                yield f"{base}.{layer}.self_attn.k_proj.output_scale"
        for expert in range(num_experts):
            for proj in ("down_proj", "gate_proj", "up_proj"):
                for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    yield f"{base}.{layer}.mlp.experts.{expert}.{proj}.{suffix}"
        if extra:
            yield f"{base}.{layer}.mlp.gate.weight"
            yield f"{base}.{layer}.mlp.gate.bias"  # ignored: unexpected suffix
            yield f"{base}.{layer}.ple.hashstats_x"  # skipped by skip_substrs
    if extra:
        yield "model.language_model.embed_tokens.weight"
        yield "mtp.layers.0.weight"  # skipped: mtp.
        yield "model.visual.blocks.0.attn.weight"  # skipped: visual.
        yield "lm_head.weight"


def tensor_stream(names: Iterable[str], distinct: bool = False):
    """(name, tensor) pairs; ``distinct`` fills tensor k with the value k."""
    scalar = torch.zeros(())
    vec = torch.zeros(1)
    two = torch.zeros(2)
    mat = torch.zeros(2, 2)
    for k, name in enumerate(names):
        if name.endswith("_scale") or "scale_2" in name:
            base = scalar
        elif name.endswith(("gate.weight", "embed_tokens.weight", "lm_head.weight")):
            base = mat
        elif name.endswith(("layernorm.weight", ".bias")):
            base = two
        else:
            base = vec
        yield name, (torch.full_like(base, float(k + 1)) if distinct else base)


def run_load(model, names, record: bool = True):
    return model.load_weights(tensor_stream(names))


sys.modules.setdefault("pipeline_harness", sys.modules[__name__])
