# SPDX-License-Identifier: Apache-2.0
"""Load the KV steady-budget code without a vLLM install or a GPU.

Three pieces are loaded from the repository files (never from an installed
vLLM, so the tests always exercise this tree):

    load_budget()        ``vllm/v1/worker/kv_steady_budget.py`` itself, as the
                         module ``sx_kv_steady_budget``. It imports nothing but
                         the standard library.
    worker_class(...)    the ``Worker`` methods of ``vllm/v1/worker/gpu_worker.py``
                         that implement the switch (``determine_available_memory``,
                         ``_sx_plan_steady_budget``, ``_sx_mark``,
                         ``_sx_after_kv_cache_init``, ``_sx_steady_audit``), cut
                         out with ``ast`` and bound to a small class. ``torch``,
                         ``envs``, the memory profiler and the platform are small
                         fakes with the same names; the arithmetic and the control
                         flow are the repository's own.
    install_graph_utils()  ``get_explicit_cudagraph_memory_reserve`` and
                         ``get_sm70_cudagraph_memory_reserve`` cut out of
                         ``vllm/v1/worker/gpu/cudagraph_utils.py`` and registered
                         as ``vllm.v1.worker.gpu.cudagraph_utils`` (the worker
                         imports the second one lazily).

Not a test file; see ``test_kv_steady_budget_cpu.py`` and
``test_worker_switch_cpu.py``.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import textwrap
import types
import typing
from dataclasses import dataclass, field

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUDGET = os.path.join(REPO, "vllm", "v1", "worker", "kv_steady_budget.py")
WORKER = os.path.join(REPO, "vllm", "v1", "worker", "gpu_worker.py")
GRAPH_UTILS = os.path.join(REPO, "vllm", "v1", "worker", "gpu", "cudagraph_utils.py")
MODULE_NAME = "sx_kv_steady_budget"

MiB = 1 << 20
GiB = 1 << 30

WORKER_METHODS = {
    "determine_available_memory",
    "_sx_plan_steady_budget",
    "_sx_mark",
    "_sx_cached_free_bytes",
    "_sx_release_idle_cache",
    "_sx_after_kv_cache_init",
    "_sx_steady_audit",
}


def load_budget() -> types.ModuleType:
    module = sys.modules.get(MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(MODULE_NAME, BUDGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _segment(source: str, node: ast.AST) -> str:
    first = min(
        [node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))]
    )
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def worker_method_sources(names: set[str]) -> dict[str, str]:
    """Source of the named ``Worker`` methods, dedented to module level."""
    source = open(WORKER, encoding="utf-8").read()
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "Worker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    found[item.name] = textwrap.dedent(_segment(source, item))
    missing = names - set(found)
    assert not missing, f"Worker methods not found: {sorted(missing)}"
    return found


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, msg: str, *args: object) -> None:
        self.records.append((level, msg % args if args else msg))

    def info(self, msg: str, *args: object, **kw: object) -> None:
        self._add("info", msg, *args)

    info_once = info

    def warning(self, msg: str, *args: object, **kw: object) -> None:
        self._add("warning", msg, *args)

    warning_once = warning

    def error(self, msg: str, *args: object, **kw: object) -> None:
        self._add("error", msg, *args)

    def debug(self, msg: str, *args: object, **kw: object) -> None:
        pass

    def text(self, level: str | None = None) -> str:
        return "\n".join(m for lv, m in self.records if level in (None, lv))


class _Mode:
    """Stands in for ``CUDAGraphMode`` (only NONE is compared)."""

    NONE = "NONE"
    FULL_AND_PIECEWISE = "FULL_AND_PIECEWISE"


class _EnvsView:
    """``vllm.envs`` as the worker reads it."""

    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS = True

    @property
    def VLLM_V2_CUDAGRAPH_MEM_MIB(self) -> float:  # noqa: N802
        return float(os.environ.get("VLLM_V2_CUDAGRAPH_MEM_MIB", "0") or 0)


def install_graph_utils(logger: RecordingLogger | None = None) -> types.ModuleType:
    """The real graph-reserve helpers as ``vllm.v1.worker.gpu.cudagraph_utils``."""
    wanted = {
        "get_explicit_cudagraph_memory_reserve",
        "get_sm70_cudagraph_memory_reserve",
    }
    source = open(GRAPH_UTILS, encoding="utf-8").read()
    parts = [
        _segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(parts) == len(wanted), "graph reserve helpers not found"
    namespace = {
        "os": os,
        "envs": _EnvsView(),
        "CUDAGraphMode": _Mode,
        "logger": logger or RecordingLogger(),
    }
    exec(compile("\n\n".join(parts), GRAPH_UTILS, "exec"), namespace)  # noqa: S102
    module = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
    for name in wanted:
        setattr(module, name, namespace[name])
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = package
    sys.modules["vllm.v1.worker.gpu.cudagraph_utils"] = module
    return module


@dataclass
class Snapshot:
    """What the fake ``memory_profiling`` hands to the worker."""

    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    torch_peak: int = 0
    non_torch_memory: int = 0


@dataclass
class ProfileResult:
    weights_memory: int = 0
    non_torch_increase: int = 0
    torch_peak_increase: int = 0
    non_kv_cache_memory: int = 0
    before_profile: Snapshot = field(default_factory=Snapshot)
    after_profile: Snapshot = field(default_factory=Snapshot)


class FakeCuda:
    """``torch.cuda`` as the audit reads it; ``free`` moves with the test.

    ``cached_default`` / ``cached_graph_pool`` are the idle bytes of the default
    pool and of a CUDA-graph private pool, as ``memory_snapshot`` reports them.
    """

    def __init__(self, free: int = 0, reserved: int = 0, allocated: int = 0) -> None:
        self.free = free
        self.reserved = reserved
        self.allocated = allocated
        self.cached_default = 0
        self.cached_graph_pool = 0
        self.snapshot_error: Exception | None = None

    def current_device(self):
        return 0

    def mem_get_info(self, device=None):
        return (self.free, 0)

    def memory_snapshot(self):
        if self.snapshot_error is not None:
            raise self.snapshot_error
        mib = 1 << 20
        return [
            {  # default pool: two idle blocks and one in use
                "device": 0,
                "segment_pool_id": (0, 0),
                "blocks": [
                    {"size": self.cached_default // 2, "state": "inactive"},
                    {"size": self.cached_default - self.cached_default // 2, "state": "inactive"},
                    {"size": 64 * mib, "state": "active_allocated"},
                ],
            },
            {  # a CUDA-graph private pool: idle blocks are kept for replay
                "device": 0,
                "segment_pool_id": (0, 7),
                "blocks": [{"size": self.cached_graph_pool, "state": "inactive"}],
            },
            {  # another GPU
                "device": 1,
                "segment_pool_id": (0, 0),
                "blocks": [{"size": 10**12, "state": "inactive"}],
            },
        ]

    def memory_reserved(self, device=None):
        return self.reserved

    def memory_allocated(self, device=None):
        return self.allocated


def format_gib(value: int) -> str:
    return str(round(value / GiB, 2))


def install_lane_contracts(lane_contract: str | None) -> None:
    """``vllm.config.vllm`` with the two lane predicates the worker imports.

    ``lane_contract`` is "mtp", "nomtp" or None (neither admitted lane).
    """
    for name in ("vllm", "vllm.config"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = package
    stub = types.ModuleType("vllm.config.vllm")
    stub._is_sm70_qwen38_mtp_lane_contract = (  # type: ignore[attr-defined]
        lambda model, spec, parallel: lane_contract == "mtp" and spec is not None
    )
    stub._is_sm70_qwen38_nomtp_dual_compile_contract = (  # type: ignore[attr-defined]
        lambda model, spec, parallel: lane_contract == "nomtp" and spec is None
    )
    sys.modules["vllm.config.vllm"] = stub


def worker_class(
    *,
    profile: ProfileResult,
    profile_torch_peak: int,
    estimate_cudagraphs: bool = True,
    is_sm70: bool = True,
    is_cuda: bool = True,
    lane_contract: str | None = "mtp",
    cuda: FakeCuda | None = None,
    logger: RecordingLogger | None = None,
):
    """A class with the real switch methods of ``Worker``; returns (cls, logger)."""
    logger = logger or RecordingLogger()
    budget = load_budget()
    install_graph_utils(logger)
    install_lane_contracts(lane_contract)
    calls = {"empty_cache": 0, "collect": 0}

    envs = _EnvsView()
    envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS = estimate_cudagraphs  # type: ignore[misc]

    @contextlib.contextmanager
    def memory_profiling(init_snapshot, weights_memory=0):
        yield profile

    def empty_cache() -> None:
        calls["empty_cache"] += 1

    accelerator = types.SimpleNamespace(
        memory_stats=lambda device: {"allocated_bytes.all.peak": profile_torch_peak},
        synchronize=lambda *a, **k: None,
        empty_cache=empty_cache,
    )
    fake_torch = types.SimpleNamespace(
        accelerator=accelerator,
        inference_mode=lambda: (lambda f: f),
        cuda=cuda or FakeCuda(),
    )
    platform = types.SimpleNamespace(
        is_cuda=lambda: is_cuda,
        # strict: only the SM70 spellings are accepted, so a wrong literal fails
        is_device_capability=lambda cap, device_id=None: is_sm70 and cap in (70, (7, 0)),
    )
    namespace = {
        "torch": fake_torch,
        "envs": envs,
        "CUDAGraphMode": _Mode,
        "current_platform": platform,
        "logger": logger,
        "memory_profiling": memory_profiling,
        "format_gib": format_gib,
        "GiB_bytes": GiB,
        "kv_budget": budget,
        "Any": typing.Any,
        "KVCacheConfig": object,
        "gc": types.SimpleNamespace(
            collect=lambda: calls.__setitem__("collect", calls["collect"] + 1)
        ),
    }
    sources = worker_method_sources(WORKER_METHODS)
    exec(  # noqa: S102
        compile("\n\n".join(sources.values()), WORKER, "exec"), namespace
    )
    cls = type("FakeWorker", (), {name: namespace[name] for name in sources})
    cls.calls = calls  # type: ignore[attr-defined]
    return cls, logger
