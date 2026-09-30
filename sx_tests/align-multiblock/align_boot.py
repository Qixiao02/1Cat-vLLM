# SPDX-License-Identifier: Apache-2.0
"""Stub loader for the align multi-block tests (CPU only, no vLLM install).

``import vllm`` needs the whole serving stack. The scheduler and the KV-cache
core do not: this loader imports those modules from the source tree as they
are and replaces everything else they import with inert stubs.

Loaded from source (``REAL_MODULES``): the scheduler, the async scheduler, the
request queues, ``Request``, the KV-cache manager, coordinator, block pool and
single-type managers, and the KV-cache specs. Nothing in them is modified.

Stubbed: logging, env knobs, hashing (pickle + SHA-256, as ``sha256`` in the
real module), sampling params, engine outputs, stats, connectors. The stub of
``vllm.config.vllm`` carries the real contract predicates, cut out of the real
file with ``ast``, and a device-capability switch that tests set
(``universe.config_vllm.SM70``).

A ``Universe`` is one isolated set of these modules. ``load()`` reads the
working tree. ``load(upstream=[...])`` takes the listed modules from the
upstream base commit instead (``git show``), which gives the tests pristine
upstream behaviour to compare against:

    patched = align_boot.load()
    upstream = align_boot.load(upstream=align_boot.PATCHED_MODULES)

The base commit is ``SX_ALIGN_BASE_REV`` (default ``d30469863``); the source
tree is ``SX_ALIGN_REPO`` (default: the tree this file is in). Without git or
without that commit ``have_base()`` is false and the comparisons against
upstream are skipped.

Not a test file. Run ``python align_boot.py`` for a self-check.
"""

from __future__ import annotations

import ast
import contextlib
import enum
import functools
import hashlib
import importlib
import importlib.abc
import importlib.util
import logging
import os
import pickle
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO = Path(os.environ.get("SX_ALIGN_REPO") or Path(__file__).resolve().parents[2])
BASE_REV = os.environ.get("SX_ALIGN_BASE_REV", "d30469863")

# Module name -> path in the source tree.
REAL_MODULES = {
    "vllm.utils.math_utils": "vllm/utils/math_utils.py",
    "vllm.v1.kv_cache_interface": "vllm/v1/kv_cache_interface.py",
    "vllm.v1.request": "vllm/v1/request.py",
    "vllm.v1.outputs": "vllm/v1/outputs.py",
    "vllm.v1.core.block_pool": "vllm/v1/core/block_pool.py",
    "vllm.v1.core.encoder_cache_manager": "vllm/v1/core/encoder_cache_manager.py",
    "vllm.v1.core.kv_cache_coordinator": "vllm/v1/core/kv_cache_coordinator.py",
    "vllm.v1.core.kv_cache_manager": "vllm/v1/core/kv_cache_manager.py",
    "vllm.v1.core.kv_cache_metrics": "vllm/v1/core/kv_cache_metrics.py",
    "vllm.v1.core.kv_cache_utils": "vllm/v1/core/kv_cache_utils.py",
    "vllm.v1.core.single_type_kv_cache_manager": (
        "vllm/v1/core/single_type_kv_cache_manager.py"
    ),
    "vllm.v1.core.sched.async_scheduler": "vllm/v1/core/sched/async_scheduler.py",
    "vllm.v1.core.sched.interface": "vllm/v1/core/sched/interface.py",
    "vllm.v1.core.sched.output": "vllm/v1/core/sched/output.py",
    "vllm.v1.core.sched.request_queue": "vllm/v1/core/sched/request_queue.py",
    "vllm.v1.core.sched.scheduler": "vllm/v1/core/sched/scheduler.py",
    "vllm.v1.core.sched.utils": "vllm/v1/core/sched/utils.py",
}
SCHEDULER = "vllm.v1.core.sched.scheduler"
ALLOCATOR = "vllm.v1.core.single_type_kv_cache_manager"
# The modules the align multi-block series changes.
PATCHED_MODULES = (SCHEDULER, ALLOCATOR)
CONFIG_VLLM = "vllm/config/vllm.py"
_CONTRACTS = (
    "_is_sm70_dflash2_verifier_contract",
    "_is_sm70_qwen38_decode_compile_contract",
)


def _is_vllm(name: str) -> bool:
    return name == "vllm" or name.startswith("vllm.")


@functools.cache
def _git_show(rev: str, rel_path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{rel_path}"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.decode("utf-8")


def have_base() -> bool:
    """Whether the upstream base commit is readable through git."""
    return all(
        _git_show(BASE_REV, REAL_MODULES[name]) is not None for name in PATCHED_MODULES
    )


def read_source(rel_path: str, rev: str | None = None) -> str:
    """Source of a file of the tree, from the working tree or from `rev`."""
    if rev is None:
        return (REPO / rel_path).read_text(encoding="utf-8")
    source = _git_show(rev, rel_path)
    if source is None:
        raise RuntimeError(f"cannot read {rel_path} at {rev} from {REPO}")
    return source


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------
class _StubMeta(type):
    """Class whose unknown class attributes are further stub classes."""

    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _StubMeta(f"{cls.__name__}.{name}", (Stub,), {})
        setattr(cls, name, value)
        return value


class Stub(metaclass=_StubMeta):
    """Accepts any construction; instances only have what they were given."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __class_getitem__(cls, item: Any):
        return cls


class _StubModule(types.ModuleType):
    """Package whose unknown attributes are stub classes."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__: list[str] = []

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _StubMeta(name, (Stub,), {})
        setattr(self, name, value)
        return value


class _Logger(logging.Logger):
    def info_once(self, msg, *args, **kwargs) -> None:
        self.info(msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs) -> None:
        self.warning(msg, *args, **kwargs)

    def debug_once(self, msg, *args, **kwargs) -> None:
        self.debug(msg, *args, **kwargs)


class _Envs(types.ModuleType):
    """Every env knob reads as unset."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return 0 if name.endswith("_DEPTH") else None


def sha256(obj: Any) -> bytes:
    return hashlib.sha256(
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    ).digest()


class SamplingParams:
    """The fields of the real class that ``Request`` and ``check_stop`` read."""

    def __init__(self, max_tokens: int = 16, **kwargs: Any) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = 0
        self.eos_token_id = None
        self.stop_token_ids = None
        self.extra_args = None
        self.skip_reading_prefix_cache = None
        self.repetition_detection = None
        self.num_logprobs = None
        self.routed_experts_prompt_start = None
        self.ignore_eos = False
        self.__dict__.update(kwargs)

    def update_from_generation_config(self, generation_config, eos_token_id=None):
        if not self.ignore_eos:
            self.eos_token_id = eos_token_id


class ConstantList(list):
    """Read-only view in the real module; a live copy is enough here."""

    def __init__(self, x: list) -> None:
        super().__init__()
        self._x = x

    def _sync(self) -> list:
        return self._x

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, item):
        return self._x[item]

    def __iter__(self):
        return iter(self._x)

    def __eq__(self, other) -> bool:
        return list(self._x) == list(other)

    def __repr__(self) -> str:
        return f"ConstantList({self._x!r})"

    def copy(self) -> list:
        return list(self._x)


class PrefixCacheStats:
    def __init__(self) -> None:
        self.reset = False
        self.requests = 0
        self.queries = 0
        self.hits = 0

    def record(self, num_tokens: int, num_hits: int, preempted: bool = False) -> None:
        self.requests += 1
        self.queries += num_tokens
        self.hits += num_hits


class PrefillStats:
    def __init__(self) -> None:
        self.num_prompt_tokens = 0
        self.num_local_cached_tokens = 0
        self.num_external_cached_tokens = 0

    def set(self, **kwargs: int) -> None:
        self.__dict__.update(kwargs)


class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


class _FakeConnector:
    """Enough of a KV connector for the scheduler to be constructed with one."""

    def bind_gpu_block_pool(self, block_pool: Any) -> None:
        pass


class KVConnectorFactory:
    @staticmethod
    def create_connector(**kwargs: Any) -> _FakeConnector:
        return _FakeConnector()


class MambaAttentionBackendEnum(enum.Enum):
    MAMBA = "mamba"
    MAMBA2 = "mamba2"
    SHORT_CONV = "short_conv"
    LINEAR = "linear"
    GDN_ATTN = "gdn_attn"


def _module(name: str, **attrs: Any) -> _StubModule:
    module = _StubModule(name)
    module.__dict__.update(attrs)
    return module


def _config_vllm_module(torch: Any) -> types.ModuleType:
    """``vllm.config.vllm``: the real contract predicates, fake hardware."""
    tree = ast.parse(read_source(CONFIG_VLLM))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _CONTRACTS
    ]
    assert len(nodes) == len(_CONTRACTS), "contract predicates moved"
    # A plain module: a name it lacks fails to import, as in the real one.
    module = types.ModuleType("vllm.config.vllm")
    module.__dict__.update(torch=torch, Any=Any, Mapping=Mapping, SM70=True)
    exec(  # noqa: S102 - source tree code, as `import` would run it
        compile(ast.Module(body=nodes, type_ignores=[]), CONFIG_VLLM, "exec"),
        module.__dict__,
    )

    def _any_participating_device_is_capability(cfg, capability) -> bool:
        return bool(module.SM70) and tuple(capability) == (7, 0)

    module._any_participating_device_is_capability = (
        _any_participating_device_is_capability
    )
    return module


def _explicit_stubs() -> dict[str, types.ModuleType]:
    import torch

    loggers: dict[str, _Logger] = {}

    def init_logger(name: str) -> _Logger:
        if name not in loggers:
            loggers[name] = _Logger(name, level=logging.INFO)
            loggers[name].addHandler(logging.NullHandler())
        return loggers[name]

    def record_function_or_nullcontext(name: str):
        return contextlib.nullcontext()

    structured_output_request = _module("vllm.v1.structured_output.request")
    structured_output_request.StructuredOutputRequest.from_sampling_params = (
        staticmethod(lambda sampling_params: None)
    )
    return {
        module.__name__: module
        for module in (
            _module("vllm", __version__="stub"),
            _Envs("vllm.envs"),
            _module("vllm.logger", init_logger=init_logger, loggers=loggers),
            _module("vllm.config", VllmConfig=Stub),
            _config_vllm_module(torch),
            _module(
                "vllm.utils",
                length_from_prompt_token_ids_or_embeds=(
                    lambda token_ids, embeds=None: len(token_ids)
                ),
            ),
            _module(
                "vllm.utils.hashing",
                sha256=sha256,
                sha256_cbor=sha256,
                xxhash_cbor=sha256,
            ),
            _module(
                "vllm.utils.mem_utils", format_gib=lambda b: f"{b / 2**30:.2f}"
            ),
            _module(
                "vllm.utils.torch_utils",
                get_dtype_size=lambda dtype: torch.empty(0, dtype=dtype).element_size(),
                nvfp4_kv_cache_full_dim=lambda *args, **kwargs: 0,
            ),
            _module("vllm.sampling_params", SamplingParams=SamplingParams),
            _module(
                "vllm.v1.utils",
                ConstantList=ConstantList,
                record_function_or_nullcontext=record_function_or_nullcontext,
                tensor_data=lambda tensor: tensor,
            ),
            _module("vllm.v1.engine", FinishReason=FinishReason),
            _module(
                "vllm.distributed.kv_transfer.kv_connector.factory",
                KVConnectorFactory=KVConnectorFactory,
            ),
            _module(
                "vllm.v1.metrics.stats",
                PrefixCacheStats=PrefixCacheStats,
                PrefillStats=PrefillStats,
            ),
            _module(
                "vllm.v1.attention.backends.registry",
                MambaAttentionBackendEnum=MambaAttentionBackendEnum,
            ),
            structured_output_request,
        )
    }


# --------------------------------------------------------------------------
# Import machinery
# --------------------------------------------------------------------------
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, upstream: frozenset[str]) -> None:
        self.upstream = upstream
        self.stubs = _explicit_stubs()

    def find_spec(self, fullname, path=None, target=None):
        if not _is_vllm(fullname):
            return None
        return importlib.util.spec_from_loader(
            fullname, self, is_package=fullname not in REAL_MODULES
        )

    def create_module(self, spec):
        name = spec.name
        if name in REAL_MODULES:
            return None
        return self.stubs.get(name) or _StubModule(name)

    def exec_module(self, module) -> None:
        name = module.__name__
        if name not in REAL_MODULES:
            return
        rel_path = REAL_MODULES[name]
        rev = BASE_REV if name in self.upstream else None
        label = f"{rev}:{rel_path}" if rev else str(REPO / rel_path)
        module.__file__ = label
        exec(  # noqa: S102 - source tree code, as `import` would run it
            compile(read_source(rel_path, rev), label, "exec"), module.__dict__
        )


class Universe:
    """One isolated copy of the scheduler and KV-cache modules."""

    def __init__(self, modules: dict[str, types.ModuleType], upstream) -> None:
        self.modules = modules
        self.upstream = upstream
        self.scheduler = modules[SCHEDULER]
        self.async_scheduler = modules["vllm.v1.core.sched.async_scheduler"]
        self.allocator = modules[ALLOCATOR]
        self.kv_cache_manager = modules["vllm.v1.core.kv_cache_manager"]
        self.kv_cache_coordinator = modules["vllm.v1.core.kv_cache_coordinator"]
        self.kv_cache_utils = modules["vllm.v1.core.kv_cache_utils"]
        self.kv_cache_interface = modules["vllm.v1.kv_cache_interface"]
        self.block_pool = modules["vllm.v1.core.block_pool"]
        self.request = modules["vllm.v1.request"]
        self.outputs = modules["vllm.v1.outputs"]
        self.config_vllm = modules["vllm.config.vllm"]
        # `init_none_hash` draws a random seed; a fixed one makes block hashes
        # comparable between universes and runs.
        self.kv_cache_utils.NONE_HASH = sha256("align_boot")

    @property
    def is_upstream(self) -> bool:
        return set(PATCHED_MODULES) <= set(self.upstream)

    def logger(self, module: str = SCHEDULER) -> logging.Logger:
        return self.modules["vllm.logger"].loggers[module]

    @contextlib.contextmanager
    def activate(self):
        """Make ``import vllm...`` resolve to this universe."""
        saved = _pop_vllm_modules()
        sys.modules.update(self.modules)
        finder = self.modules["vllm"].__sx_finder__
        sys.meta_path.insert(0, finder)
        try:
            yield self
        finally:
            sys.meta_path.remove(finder)
            self.modules.update(_pop_vllm_modules())
            sys.modules.update(saved)


def _pop_vllm_modules() -> dict[str, types.ModuleType]:
    return {name: sys.modules.pop(name) for name in list(sys.modules) if _is_vllm(name)}


@functools.cache
def _load(upstream: frozenset[str]) -> Universe:
    saved = _pop_vllm_modules()
    finder = _Finder(upstream)
    sys.meta_path.insert(0, finder)
    try:
        for name in (*REAL_MODULES, *finder.stubs):
            importlib.import_module(name)
        sys.modules["vllm"].__sx_finder__ = finder
        modules = _pop_vllm_modules()
    finally:
        sys.meta_path.remove(finder)
        _pop_vllm_modules()
        sys.modules.update(saved)
    return Universe(modules, upstream)


def load(upstream: Iterable[str] = ()) -> Universe:
    """The modules of the working tree, `upstream` ones from the base commit."""
    upstream = frozenset(upstream)
    unknown = upstream - set(REAL_MODULES)
    assert not unknown, f"not loaded from source: {sorted(unknown)}"
    return _load(upstream)


if __name__ == "__main__":
    universe = load()
    print("working tree:", REPO)
    for name in REAL_MODULES:
        print("  real", name, "<-", universe.modules[name].__file__)
    stubbed = sorted(set(universe.modules) - set(REAL_MODULES))
    print(f"  {len(stubbed)} stub modules: {', '.join(stubbed)}")
    print("base commit", BASE_REV, "readable:", have_base())
    if have_base():
        base = load(upstream=PATCHED_MODULES)
        assert base.scheduler.Scheduler is not universe.scheduler.Scheduler
        print("  upstream scheduler <-", base.scheduler.__file__)
    print("ok")
