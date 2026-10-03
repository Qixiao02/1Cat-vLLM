# SPDX-License-Identifier: Apache-2.0
"""Load the compile-cache gating code without a vLLM install.

``import vllm`` needs the whole serving stack. The switch logic lives in
``vllm/compilation/sx_compile_cache.py`` (no vllm import at module level), and
the callers in ``vllm/envs.py``, ``vllm/config/vllm.py``,
``vllm/compilation/caching.py`` and ``vllm/models/qwen4_exp/nvidia/ple_layer.py``
are cut out with ``ast`` / marker comments and exec'd against stubs, the way
``sx_tests/ple-prefill/ple_boot.py`` does. Where vLLM is installed the stubs are
still used (the cut-out functions are the unit under test); nothing here
replaces a real module that is already imported.

Not a test file; see ``test_switch_cpu.py`` and friends.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import subprocess
import sys
import textwrap
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SX_MODULE = os.path.join(REPO, "vllm", "compilation", "sx_compile_cache.py")
ENVS = os.path.join(REPO, "vllm", "envs.py")
CONFIG = os.path.join(REPO, "vllm", "config", "vllm.py")
CACHING = os.path.join(REPO, "vllm", "compilation", "caching.py")
BACKENDS = os.path.join(REPO, "vllm", "compilation", "backends.py")
PLE = os.path.join(REPO, "vllm", "models", "qwen4_exp", "nvidia", "ple_layer.py")

# Last commit before the switch: the reference for "off is byte-identical".
BASE_REV = "8593319b2"


def read(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")


def git_show(rev: str, rel: str) -> str | None:
    """File content at a revision, or None when git/the object is missing."""
    try:
        out = subprocess.run(
            ["git", "-C", REPO, "show", f"{rev}:{rel}"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.decode("utf-8").replace("\r\n", "\n")


def load_sx_module() -> types.ModuleType:
    """vllm.compilation.sx_compile_cache with bare parent packages."""
    name = "vllm.compilation.sx_compile_cache"
    if name in sys.modules and getattr(sys.modules[name], "_sx_boot", False):
        return sys.modules[name]
    for parent in ("vllm", "vllm.compilation"):
        if parent not in sys.modules:
            module = types.ModuleType(parent)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[parent] = module
    spec = importlib.util.spec_from_file_location(name, SX_MODULE)
    module = importlib.util.module_from_spec(spec)
    module._sx_boot = True  # type: ignore[attr-defined]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["vllm.compilation"], "sx_compile_cache", module)
    return module


def function_source(source: str, name: str, cls: str | None = None) -> str:
    """Source text of a module-level function or a method of ``cls``."""
    tree = ast.parse(source)
    body = tree.body
    if cls is not None:
        node = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        body = node.body
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node, padded=True)
            assert segment is not None
            return textwrap.dedent(segment)
    raise KeyError(name)


def exec_functions(
    source: str, names: list[str], namespace: dict, cls: str | None = None
) -> dict:
    """Exec the named functions of ``source`` in ``namespace``; return it."""
    for name in names:
        exec(  # noqa: S102 - the repository's own source, under test
            compile(function_source(source, name, cls), f"<{name}>", "exec"),
            namespace,
        )
    return namespace


class RecordingLogger:
    """Stands in for vllm's logger; records (level, message % args)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, msg: str, *args: object) -> None:
        self.records.append((level, msg % args if args else msg))

    def info_once(self, msg: str, *args: object) -> None:
        self._add("info", msg, *args)

    def warning_once(self, msg: str, *args: object) -> None:
        self._add("warning", msg, *args)

    def info(self, msg: str, *args: object) -> None:
        self._add("info", msg, *args)

    def warning(self, msg: str, *args: object) -> None:
        self._add("warning", msg, *args)

    def debug(self, *args: object, **kwargs: object) -> None:
        pass


def silence_sx_logger() -> None:
    logging.getLogger("vllm.compilation.sx_compile_cache").setLevel(logging.ERROR)
