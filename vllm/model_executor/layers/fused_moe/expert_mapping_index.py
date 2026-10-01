# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookup index for ``FusedMoE.expert_mapping`` (SX_OPT_MOE_LOAD_INDEX).

``FusedMoE.load_weights`` receives one checkpoint tensor at a time and has to
find the ``expert_mapping`` entries whose ``weight_name`` is a substring of the
qualified tensor name (``"model.layers.3.mlp.experts.12.gate_proj.weight"`` is
matched by ``"experts.12.gate_proj."``). The original code scans the whole
mapping for every tensor: 512 experts x 3 projections is 1536 substring tests
per tensor, roughly 150k tensors per rank, about half of the Python time spent
loading the Swift-1.5 NVFP4 checkpoint.

The index answers the same question with a handful of dict lookups and an
exact result. Nothing is assumed about the checkpoint, the naming scheme or
where a key starts or ends inside the tensor name:

* A key ``K`` with at least two dots is ``t0.t1. ... .t(n-1)`` (``n >= 3``
  tokens, any of them may be empty; a key that ends in "." has an empty last
  token). If ``K`` is a substring of a name ``Q`` then the dots inside ``K``
  are dots of ``Q``, so the ``n - 2`` tokens between the first and the last dot
  of ``K`` are complete, consecutive tokens of ``Q.split(".")``. That window
  ``(t1, ..., t(n-2))`` is what the index is keyed on. ``t0`` may be the tail
  of a longer token of ``Q`` and the last token the head of one; both are
  settled by the ordinary substring test that is still applied to every
  candidate. The index therefore never misses a match (no false negatives);
  the substring test removes the false positives.
* A key with fewer than two dots has no complete interior token. Such keys
  (the three fused 3D-weight aliases ``experts.gate_up_proj`` and
  ``experts.down_proj``) are kept in a short list and tested directly for
  every name. If a mapping has more than ``_MAX_DIRECT_KEYS`` of them the index
  would not narrow anything and is not built at all (callers then run the
  original loop).

The matches are returned in the original mapping order and each entry at most
once, which is what the original loop produced (it never breaks after a match:
fused ``w1``/``w3`` aliases, redundant experts and multi-scheme mappings
legitimately match several entries). If no entry matches, the whole mapping is
returned so the caller's loop runs unchanged for that name; it finds nothing,
exactly as before. That fallback is only a belt-and-braces measure (the index
has no false negatives); it costs a full scan only for names that match no
entry at all, which the Qwen4Exp loading path does not produce.

The mapping is treated as immutable once assigned: ``FusedMoE`` stores it in
its constructor or the model assigns it once before loading, and nothing in the
tree appends to or edits a mapping that a layer already holds (the one
``.extend()`` on a mapping, in the Transformers backend, runs on a local list
before it is handed to the layer). The cached index is dropped when
``self.expert_mapping`` is rebound to another object or its length changes
between two ``load_weights`` calls; an edit made while a ``load_weights``
generator is running is not seen by that generator (the original loop would
see it). The index costs about 0.45 MiB of host memory per layer for 512
experts.
"""

from __future__ import annotations

import os
from typing import Any

ENV_MOE_LOAD_INDEX = "SX_OPT_MOE_LOAD_INDEX"

# Name of the per-layer cache attribute: (mapping object, its length, index).
_CACHE_ATTR = "_sx_expert_mapping_index"

# More tokenless keys than this and the index would not narrow the scan.
_MAX_DIRECT_KEYS = 64


def moe_load_index_enabled() -> bool:
    """SX_OPT_MOE_LOAD_INDEX: default on, "0" restores the original scan."""
    return os.environ.get(ENV_MOE_LOAD_INDEX, "1").strip() != "0"


class ExpertMappingIndex:
    """Token-window index over the ``weight_name`` column of an expert mapping."""

    __slots__ = ("entries", "direct", "windows", "lengths")

    def __init__(
        self,
        entries: tuple[Any, ...],
        direct: tuple[int, ...],
        windows: dict[tuple[str, ...], list[int]],
    ) -> None:
        self.entries = entries
        self.direct = direct
        self.windows = windows
        self.lengths = tuple(sorted({len(window) for window in windows}))

    def matches(self, qual_name: str, full_mapping: Any) -> Any:
        """The entries whose ``weight_name`` is a substring of ``qual_name``.

        Returns them in mapping order, or ``full_mapping`` itself when none
        matches (the caller's loop then reproduces the original scan).
        """
        tokens = qual_name.split(".")
        hits = [*self.direct]
        get = self.windows.get
        for length in self.lengths:
            # All windows of ``length`` consecutive tokens, built in C.
            for window in zip(*[tokens[i:] for i in range(length)]):
                positions = get(window)
                if positions is not None:
                    hits += positions
        entries = self.entries
        if len(hits) > 1:
            hits = sorted(set(hits))
        found = [entries[p] for p in hits if entries[p][1] in qual_name]
        return found if found else full_mapping


def build_expert_mapping_index(mapping: Any) -> ExpertMappingIndex | None:
    """Index ``mapping``, or ``None`` if it cannot be indexed exactly.

    ``None`` (an entry that is not a 4-tuple, a non-``str`` ``weight_name``,
    too many keys without an interior token, a mapping that cannot be turned
    into a tuple) makes the caller run the original loop, including whatever
    exception that loop raises for such a mapping.
    """
    try:
        entries = tuple(mapping)
        direct: list[int] = []
        windows: dict[tuple[str, ...], list[int]] = {}
        for position, (_, weight_name, _, _) in enumerate(entries):
            if not isinstance(weight_name, str):
                return None
            tokens = weight_name.split(".")
            if len(tokens) < 3:
                direct.append(position)
            else:
                windows.setdefault(tuple(tokens[1:-1]), []).append(position)
    except (TypeError, ValueError):
        return None
    if len(direct) > _MAX_DIRECT_KEYS:
        return None
    return ExpertMappingIndex(entries, tuple(direct), windows)


def get_expert_mapping_index(owner: Any, mapping: Any) -> ExpertMappingIndex | None:
    """The index of ``mapping``, built on first use and cached on ``owner``.

    The cache is valid while ``mapping`` is the same object with the same
    length; building an index that cannot be made is cached as ``None`` too.
    """
    try:
        length = len(mapping)
    except TypeError:
        return None
    cached = owner.__dict__.get(_CACHE_ATTR)
    if cached is not None and cached[0] is mapping and cached[1] == length:
        return cached[2]
    index = build_expert_mapping_index(mapping)
    object.__setattr__(owner, _CACHE_ATTR, (mapping, length, index))
    return index
