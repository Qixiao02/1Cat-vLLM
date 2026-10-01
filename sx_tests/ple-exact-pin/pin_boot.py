# SPDX-License-Identifier: Apache-2.0
"""Load the PLE exact-size pinning code without a vLLM install or a GPU.

``import vllm`` needs the whole serving stack, and ``cudaHostRegister`` needs a
GPU. Two pieces are loaded from the repository files (never from an installed
vLLM, so the tests always exercise this tree):

    load_exact_pin()   ``vllm/models/qwen4_exp/nvidia/exact_pin.py`` itself, as
                       the module ``sx_exact_pin``. Its only vLLM import is the
                       logger, which is stubbed when vLLM does not import.
    layer_class(...)   ``Qwen4ExpPinnedHostEmbedding.materialize_tables`` cut
                       out of ``ple_layer.py`` with ``ast``, bound to a small
                       class with the attributes it reads. ``torch.empty`` for
                       the device half is redirected to the CPU, the host half
                       goes through the real ``allocate_host_table``, and
                       ``plan_ple_placement`` is the real one from
                       ``common/ple.py`` (also cut with ``ast``).

The tests replace the four driver/torch hooks of ``exact_pin`` (``_host_register``,
``_host_unregister``, ``_is_pinned``, ``_device_index``) with recorders; the
page-aligned mapping, the numpy/torch wrapping, the finalizer and the
fallbacks are the real code. Not a test file; see ``test_exact_pin.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
import types
from dataclasses import dataclass

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXACT_PIN = os.path.join(
    REPO, "vllm", "models", "qwen4_exp", "nvidia", "exact_pin.py"
)
PLE_LAYER = os.path.join(REPO, "vllm", "models", "qwen4_exp", "nvidia", "ple_layer.py")
COMMON_PLE = os.path.join(REPO, "vllm", "models", "qwen4_exp", "common", "ple.py")
MODULE_NAME = "sx_exact_pin"


def _ensure_logger_module() -> None:
    try:
        import vllm.logger  # noqa: F401, PLC0415

        return
    except Exception:  # noqa: BLE001 - any import failure means "not installed"
        pass
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        if name != "vllm.logger":
            del sys.modules[name]
    stub = types.ModuleType("vllm.logger")
    stub.init_logger = logging.getLogger  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = stub


def load_exact_pin() -> types.ModuleType:
    """The repository's exact_pin.py, loaded once under ``sx_exact_pin``."""
    module = sys.modules.get(MODULE_NAME)
    if module is not None:
        return module
    _ensure_logger_module()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, EXACT_PIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _segment(source: str, node: ast.AST) -> str:
    """Source of ``node`` including its decorators, indentation kept."""
    first = min(
        [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))]
    )
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def _plan_ple_placement():
    source = open(COMMON_PLE, encoding="utf-8").read()
    wanted = {"PLEPlacement", "plan_ple_placement"}
    parts = [
        _segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    assert len(parts) == len(wanted), "PLE placement helpers not found"
    namespace = {"dataclass": dataclass}
    exec("\n\n".join(parts), namespace)  # noqa: S102
    return namespace["plan_ple_placement"]


def _materialize_tables_source() -> str:
    source = open(PLE_LAYER, encoding="utf-8").read()
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "Qwen4ExpPinnedHostEmbedding"
        ):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "materialize_tables"
                ):
                    return _segment(source, item)
    raise AssertionError("materialize_tables not found")


class _Log:
    def info(self, *args: object, **kwargs: object) -> None:
        pass


def layer_class(
    *,
    available: int | None,
    budget: int,
    embedding_dim: int,
    total_rows: int,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    """A stand-in with the real ``materialize_tables``; returns (class, calls).

    ``calls`` lists what ``materialize_tables`` asked of ``torch.empty`` for
    the device half, so tests can see the device half was still allocated.
    """
    exact_pin = load_exact_pin()
    device_calls: list[tuple] = []
    real_empty = torch.empty

    def empty_on_cpu(*args, **kwargs):
        # The device half of the table: no GPU here, a CPU tensor will do.
        device_calls.append((args, dict(kwargs)))
        kwargs = {**kwargs, "device": "cpu"}
        return real_empty(*args, **kwargs)

    fake_torch = types.SimpleNamespace(
        device=torch.device,
        accelerator=types.SimpleNamespace(current_device_index=lambda: 0),
        empty=empty_on_cpu,
    )
    namespace = {
        "torch": fake_torch,
        "plan_ple_placement": _plan_ple_placement(),
        "available_host_bytes": lambda: available,
        "format_gib": lambda value: f"{value / 1024**3:.2f} GiB",
        "allocate_host_table": exact_pin.allocate_host_table,
        "logger": _Log(),
    }
    exec(  # noqa: S102
        "class Layer:\n" + _materialize_tables_source(), namespace
    )
    layer_cls = namespace["Layer"]

    def init(self) -> None:
        self.embedding_dim = embedding_dim
        self._meta_weight_shape = (total_rows, embedding_dim)
        self._meta_weight_dtype = dtype
        self.ple_device_table = None
        self.ple_host_storage = None
        self._device_rows = 0
        self._host_rows = 0
        self._device_table_ptr = 0

    layer_cls.__init__ = init
    layer_cls._resolve_host_budget = lambda self, device: budget
    return layer_cls, device_calls
