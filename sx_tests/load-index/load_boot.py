# SPDX-License-Identifier: Apache-2.0
"""Load the weight-loading code of the "load-index" group without a vLLM install.

``import vllm`` needs the whole serving stack. What this group changes is plain
Python on the checkpoint-loading path, so the functions are taken from the real
source files with ``ast`` and executed in a small namespace; nothing is copied
by hand, so the tests run the code that ships (also where vLLM is installed:
the tests then still exercise the files of this checkout, not an installed copy).

    fused_moe_load_weights()      FusedMoE.load_weights (a plain function that
                                  works on any object with the right attributes)
    make_expert_params_mapping()  FusedMoE.make_expert_params_mapping, with the
                                  real EplbState.build_initial_global_physical_
                                  to_logical_map behind it
    expert_mapping_index()        vllm/.../fused_moe/expert_mapping_index.py
    cut_function() / cut_assignments() / exec_source()
                                  helpers for the other tests in this group

Not a test file; see ``test_moe_load_index.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
import textwrap
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOE_DIR = os.path.join(REPO, "vllm", "model_executor", "layers", "fused_moe")
LAYER = os.path.join(MOE_DIR, "layer.py")
INDEX = os.path.join(MOE_DIR, "expert_mapping_index.py")
EPLB_STATE = os.path.join(REPO, "vllm", "distributed", "eplb", "eplb_state.py")
MODELS_UTILS = os.path.join(REPO, "vllm", "model_executor", "models", "utils.py")
QWEN4_MODEL = os.path.join(REPO, "vllm", "models", "qwen4_exp", "nvidia", "model.py")


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _segment(source: str, node: ast.AST) -> str:
    return textwrap.dedent(ast.get_source_segment(source, node, padded=True))


def cut_function(path: str, name: str, cls: str | None = None) -> str:
    """Source of a top-level function, or of a method of ``cls`` (no decorators)."""
    source = read(path)
    body = ast.parse(source).body
    if cls is not None:
        owner = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        body = owner.body
    found = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"{path}: {cls}.{name}: {len(found)} definitions"
    return _segment(source, found[0])


def cut_class(path: str, name: str) -> str:
    source = read(path)
    found = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == name
    ]
    assert len(found) == 1, f"{path}: class {name}"
    return _segment(source, found[0])


def cut_assignments(path: str, names: Sequence[str]) -> str:
    """Source of the listed top-level ``NAME = ...`` / ``NAME: T = ...`` statements."""
    source = read(path)
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = _segment(source, node)
    missing = [n for n in names if n not in found]
    assert not missing, f"{path}: not found: {missing}"
    return "\n\n".join(found[n] for n in names)


class _Logger(logging.Logger):
    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass

    info = warning = warning_once = info_once = debug_once = debug


def exec_source(source: str, filename: str, **namespace: Any) -> dict[str, Any]:
    """Execute ``source`` with a prelude of common imports; return its globals."""
    ns: dict[str, Any] = {
        "__name__": "sx_load_cut",
        "Any": Any,
        "Iterable": Iterable,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "torch": torch,
        "logger": _Logger("sx_load_cut"),
        "os": os,
    }
    ns.update(namespace)
    exec(compile(source, filename, "exec"), ns)  # noqa: S102
    return ns


def load_file(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CACHE: dict[str, Any] = {}


def expert_mapping_index() -> types.ModuleType:
    """The index module (a stdlib-only file, loaded by path)."""
    if "index" not in _CACHE:
        _CACHE["index"] = load_file("sx_expert_mapping_index", INDEX)
    return _CACHE["index"]


def fused_moe_load_weights(source: str | None = None):
    """``FusedMoE.load_weights`` cut from layer.py (or from ``source``).

    ``source`` lets a test compile any other text of the method, e.g. the
    verbatim original that serves as the oracle.
    """
    if source is None and "load_weights" in _CACHE:
        return _CACHE["load_weights"]
    idx = expert_mapping_index()
    text = source or cut_function(LAYER, "load_weights", cls="FusedMoE")
    ns = exec_source(
        text,
        LAYER,
        get_expert_mapping_index=idx.get_expert_mapping_index,
        moe_load_index_enabled=idx.moe_load_index_enabled,
    )
    fn = ns["load_weights"]
    if source is None:
        _CACHE["load_weights"] = fn
    return fn


def make_expert_params_mapping():
    """``FusedMoE.make_expert_params_mapping`` as ``f(model, gate, down, up, ...)``.

    ``model`` only has to offer ``named_parameters()`` (it decides ``base_layer.``).
    """
    if "make_mapping" in _CACHE:
        return _CACHE["make_mapping"]
    eplb_source = cut_function(
        EPLB_STATE, "build_initial_global_physical_to_logical_map", cls="EplbState"
    )
    eplb_ns = exec_source(eplb_source, EPLB_STATE)

    class EplbState:  # the real static method, executed from its source
        build_initial_global_physical_to_logical_map = staticmethod(
            eplb_ns["build_initial_global_physical_to_logical_map"]
        )

    text = cut_function(LAYER, "make_expert_params_mapping", cls="FusedMoE")
    ns = exec_source(text, LAYER, EplbState=EplbState)
    fn = ns["make_expert_params_mapping"]

    def call(model, gate, down, up, num_experts, num_redundant_experts=0,
             include_fused=False):
        return fn(None, model, gate, down, up, num_experts,
                  num_redundant_experts, include_fused)

    _CACHE["make_mapping"] = call
    return call
