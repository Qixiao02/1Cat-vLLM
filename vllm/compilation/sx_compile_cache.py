# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SX_OPT_COMPILE_CACHE: switchable reuse of vLLM's torch.compile cache.

The SM70 Flash-V100 compile graph (``VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH``,
the Flash-Next lane) has always run with ``VLLM_DISABLE_COMPILE_CACHE=1`` and
in-memory AOT compile: a cold start pays two torch.compile runs (prefill/mixed
backbone and the decode backbone, ~185 s) on every restart, because reloading
cached artifacts once produced greedy token drift. Upstream (1Cat v1.5.1) now
ships the cache on by default; this module is the fork's gated version of that.

    SX_OPT_COMPILE_CACHE   (default "0")
      "0" / unset / "off"    today's behaviour, byte for byte: the config forces
                             VLLM_DISABLE_COMPILE_CACHE=1, nothing in this
                             module changes a cache key or a code path.
      "1" / "on" / "subgraph"
                             no forced opt-out; the compiled graphs of every
                             piecewise submodule are reloaded from the cache,
                             but the FX graph is rebuilt in the new process
                             (VLLM_USE_AOT_COMPILE=0 unless set explicitly).
                             This is the strategy upstream qualified on 27B
                             (cold/warm outputs identical); Dynamo still runs.
      "aot"                  no forced opt-out and VLLM_USE_AOT_COMPILE=1: the
                             whole AOT artifact (FX graph + compiled pieces) is
                             reloaded. Fastest restart; upstream's 27B parity
                             gate FAILED for this path (autotune configs differ
                             between a cold compile and a reload).
    An explicit VLLM_DISABLE_COMPILE_CACHE / VLLM_USE_AOT_COMPILE in the
    environment always wins over what the switch would choose.

With the switch on, the compile cache key additionally carries a *build
identity* (``extra_compile_factors``): torch/triton/CUDA versions, device name,
the content of every ``.py`` file of the vllm package (and of the other
compiled-graph-relevant packages), a fingerprint of the native libraries, the
torch-backport level and the non-VLLM_ env switches that reach Inductor/Triton.
Without it a cache from another build is only caught for the Python files
Dynamo happened to trace. ``VllmConfig.compute_hash`` also gets the checkpoint
identity (file names, sizes, mtimes) because NVFP4 linears bake per-layer global
scales (Python floats) into the traced graph.

Nothing here imports vllm at module level so the gating logic can be tested on
a machine without a vLLM install (see ``sx_tests/compile-cache``).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

ENV_SWITCH = "SX_OPT_COMPILE_CACHE"

MODE_OFF = "off"
MODE_SUBGRAPH = "subgraph"
MODE_AOT = "aot"

_OFF_VALUES = frozenset({"", "0", "off", "false", "no", "none"})
_SUBGRAPH_VALUES = frozenset({"1", "on", "true", "yes", "subgraph"})
_AOT_VALUES = frozenset({"aot"})

# Packages whose Python (and native) content can change a traced graph. vllm is
# always hashed; the others only when they are installed.
_IDENTITY_PACKAGES = ("vllm", "flash_attn_v100", "flash_qla", "flashinfer_sm70")

# Non-VLLM_/SX_OPT_ environment switches that reach Inductor, Triton or a
# kernel selector inside the traced graph. The VLLM_/SX_OPT_ names are already
# hashed by envs.compile_factors().
_ENV_PREFIX_FACTORS = ("TORCHINDUCTOR_", "TORCHDYNAMO_", "TORCH_COMPILE", "TRITON_")
_ENV_NAME_FACTORS = (
    "FLASH_QLA_SM70_USE_ORIGINAL_TILELANG",
    "USE_DEFAULT_FLA_NORM",
    "Q_SCALE_CONSTANT",
    "V_SCALE_CONSTANT",
    "CUBLAS_WORKSPACE_CONFIG",
)
# Directories and thread counts do not change what is compiled.
_ENV_IGNORED = frozenset(
    {
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "TRITON_CACHE_DIR",
        "TRITON_HOME",
        "TRITON_DUMP_DIR",
        "TRITON_OVERRIDE_DIR",
        "TRITON_PRINT_AUTOTUNING",
        "TRITON_CACHE_MANAGER",
    }
)

_HEAD_BYTES = 1 << 20
_TAIL_BYTES = 1 << 16


def _logger() -> Any:
    try:
        from vllm.logger import init_logger

        return init_logger(__name__)
    except Exception:  # noqa: BLE001 - no vllm install (unit tests)
        return logging.getLogger(__name__)


def _once(log: Any, level: str, msg: str, *args: Any) -> None:
    fn = getattr(log, f"{level}_once", None)
    if fn is not None:
        fn(msg, *args)
    else:
        getattr(log, level)(msg, *args)


def normalize_mode(raw: str | None) -> str | None:
    """Canonical mode of a raw switch value; None for an unknown value."""
    value = (raw or "").strip().lower()
    if value in _OFF_VALUES:
        return MODE_OFF
    if value in _SUBGRAPH_VALUES:
        return MODE_SUBGRAPH
    if value in _AOT_VALUES:
        return MODE_AOT
    return None


def compile_cache_mode(environ: Mapping[str, str] | None = None) -> str:
    """off | subgraph | aot. An unknown value is "off" (and warned about)."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_SWITCH)
    mode = normalize_mode(raw)
    if mode is None:
        _once(
            _logger(),
            "warning",
            "Ignoring unknown %s=%r (use 0, 1/subgraph or aot); the compile "
            "cache policy stays at the default (off).",
            ENV_SWITCH,
            raw,
        )
        return MODE_OFF
    return mode


def compile_cache_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return compile_cache_mode(environ) != MODE_OFF


def aot_default(mode: str) -> str | None:
    """Value VLLM_USE_AOT_COMPILE takes when the user did not set it."""
    if mode == MODE_SUBGRAPH:
        return "0"
    if mode == MODE_AOT:
        return "1"
    return None


@dataclasses.dataclass(frozen=True)
class CachePolicy:
    """What the switch decided for one engine; ``messages`` are (level, text)."""

    mode: str
    cache_disabled: bool
    aot: bool | None
    sets: tuple[tuple[str, str], ...] = ()
    messages: tuple[tuple[str, str], ...] = ()


def plan_policy(environ: Mapping[str, str] | None = None) -> CachePolicy:
    """Decide the cache/AOT environment for a non-"off" mode.

    Pure function of ``environ``; ``apply_policy`` writes the result. For mode
    "off" nothing is planned (the caller keeps the original branches).
    """
    env = os.environ if environ is None else environ
    mode = compile_cache_mode(env)
    if mode == MODE_OFF:
        return CachePolicy(mode=mode, cache_disabled=True, aot=None)

    sets: list[tuple[str, str]] = []
    messages: list[tuple[str, str]] = []

    explicit_disable = env.get("VLLM_DISABLE_COMPILE_CACHE")
    cache_disabled = explicit_disable is not None and explicit_disable.strip() == "1"
    if cache_disabled:
        messages.append(
            (
                "warning",
                f"{ENV_SWITCH}={env.get(ENV_SWITCH)} is overridden by the explicit "
                "VLLM_DISABLE_COMPILE_CACHE=1: the torch.compile cache stays "
                "disabled.",
            )
        )
    else:
        messages.append(
            (
                "info",
                f"{ENV_SWITCH}={mode}: leaving VLLM_DISABLE_COMPILE_CACHE unset "
                "for the SM70 Flash-V100 compile graph; compiled artifacts are "
                "reused across restarts. The cache key carries the build "
                "identity, so another build never reuses them.",
            )
        )

    explicit_aot = env.get("VLLM_USE_AOT_COMPILE")
    default_aot = aot_default(mode)
    if explicit_aot is None:
        assert default_aot is not None
        sets.append(("VLLM_USE_AOT_COMPILE", default_aot))
        aot = default_aot == "1"
        if aot:
            messages.append(
                (
                    "info",
                    f"{ENV_SWITCH}=aot: VLLM_USE_AOT_COMPILE=1, the whole AOT "
                    "artifact is reloaded on restart. Check cold/warm output "
                    "parity (sx_tests/compile-cache/cache_parity.py) before "
                    "relying on it.",
                )
            )
        else:
            messages.append(
                (
                    "info",
                    f"{ENV_SWITCH}={mode}: reusing compiled subgraphs without "
                    "the AOT FX-graph reload (VLLM_USE_AOT_COMPILE=0); Dynamo "
                    "still traces the graph on every start.",
                )
            )
    else:
        aot = explicit_aot.strip() == "1"
        if aot and mode == MODE_SUBGRAPH:
            messages.append(
                (
                    "warning",
                    f"{ENV_SWITCH}={env.get(ENV_SWITCH)} with explicit "
                    "VLLM_USE_AOT_COMPILE=1 selects the AOT FX-graph reload, "
                    "which is the path that failed upstream's cold/warm parity "
                    "gate; use SX_OPT_COMPILE_CACHE=aot to say so explicitly.",
                )
            )
        elif not aot and mode == MODE_AOT:
            messages.append(
                (
                    "warning",
                    f"{ENV_SWITCH}=aot with explicit VLLM_USE_AOT_COMPILE=0: "
                    "only compiled subgraphs are reused (no AOT reload).",
                )
            )
    return CachePolicy(
        mode=mode,
        cache_disabled=cache_disabled,
        aot=aot,
        sets=tuple(sets),
        messages=tuple(messages),
    )


def apply_policy(
    policy: CachePolicy,
    environ: MutableMapping[str, str] | None = None,
    log: Any = None,
) -> None:
    """Write the planned defaults (never over an explicit value) and log."""
    env = os.environ if environ is None else environ
    log = log or _logger()
    for key, value in policy.sets:
        env.setdefault(key, value)
    for level, text in policy.messages:
        _once(log, level, "%s", text)


# ---------------------------------------------------------------------------
# build identity
# ---------------------------------------------------------------------------


def _iter_files(root: str, suffixes: tuple[str, ...]) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(suffixes):
                yield os.path.join(dirpath, name)


def tree_digest(root: str, suffixes: tuple[str, ...] = (".py",)) -> str:
    """Digest of the relative paths and contents of every matching file."""
    digest = hashlib.sha256()
    root = os.path.abspath(root)
    for path in _iter_files(root, suffixes):
        digest.update(os.path.relpath(path, root).replace(os.sep, "/").encode())
        digest.update(b"\0")
        try:
            with open(path, "rb") as f:
                digest.update(hashlib.sha256(f.read()).digest())
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()


def native_fingerprint(root: str) -> str:
    """Name, size and head/tail content of every shared library under root."""
    digest = hashlib.sha256()
    root = os.path.abspath(root)
    for path in _iter_files(root, (".so", ".pyd")):
        digest.update(os.path.relpath(path, root).replace(os.sep, "/").encode())
        try:
            size = os.path.getsize(path)
            digest.update(str(size).encode())
            with open(path, "rb") as f:
                digest.update(f.read(_HEAD_BYTES))
                if size > _HEAD_BYTES + _TAIL_BYTES:
                    f.seek(size - _TAIL_BYTES)
                    digest.update(f.read(_TAIL_BYTES))
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()


def _package_root(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return locations[0]
    if spec.origin and os.path.isfile(spec.origin):
        return os.path.dirname(spec.origin)
    return None


def torch_backport_level() -> str:
    """"aot-triton-side-table" when tools/torch_patches 0001 is applied."""
    try:
        spec = importlib.util.find_spec("torch._dynamo.aot_compile_types")
    except (ImportError, ValueError):
        return "unknown"
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return "unknown"
    try:
        with open(spec.origin, "rb") as f:
            patched = b"_serialize_triton_kernel" in f.read()
    except OSError:
        return "unknown"
    return "aot-triton-side-table" if patched else "none"


def _package_version(dist: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(dist)
    except Exception:  # noqa: BLE001
        return "unknown"


def _device_name() -> str:
    # NVML through the platform object: querying torch.cuda here would create
    # a CUDA context in the API-server process and flip fork to spawn.
    try:
        from vllm.platforms import current_platform

        return str(current_platform.get_device_name(0))
    except Exception:  # noqa: BLE001
        return "unknown"


def env_identity(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for name, value in env.items():
        if name in _ENV_IGNORED:
            continue
        if name.startswith(_ENV_PREFIX_FACTORS) or name in _ENV_NAME_FACTORS:
            out[name] = value
    return dict(sorted(out.items()))


def gather_identity_fields(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "python": "%d.%d" % sys.version_info[:2],
        "torch_backport": torch_backport_level(),
        "device": _device_name(),
        "env": env_identity(environ),
        "triton": _package_version("triton"),
    }
    try:
        import torch

        fields["torch"] = torch.__version__
        fields["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        fields["torch"] = fields["cuda"] = "unknown"
    packages: dict[str, dict[str, str]] = {}
    for name in _IDENTITY_PACKAGES:
        root = _package_root(name)
        if root is None:
            continue
        packages[name] = {
            "py": tree_digest(root, (".py",)),
            "native": native_fingerprint(root),
        }
    fields["packages"] = packages
    return fields


def identity_digest(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode()
    ).hexdigest()


@functools.lru_cache(maxsize=1)
def build_identity() -> str:
    """Digest of everything but the model that decides a compiled artifact.

    Computed once per process (about 0.3 s for the vllm tree). A different
    process of the same engine computes the same value, so the ranks share a
    cache directory exactly when they would share artifacts.
    """
    fields = gather_identity_fields()
    digest = identity_digest(fields)
    _once(
        _logger(),
        "info",
        "sx-compile-cache build identity %s (torch %s, cuda %s, triton %s, "
        "device %s, torch backport: %s)",
        digest[:16],
        fields.get("torch"),
        fields.get("cuda"),
        fields.get("triton"),
        fields.get("device"),
        fields.get("torch_backport"),
    )
    return digest


def checkpoint_identity(model: str | None) -> str:
    """Names, sizes and mtimes of the checkpoint files of a local model dir.

    NVFP4 linears bake per-layer global scales (Python floats read from the
    weights) into the traced graph, so artifacts compiled for another set of
    weights must not be reused even when the architecture is identical. A
    Hub id or a missing path contributes only its name.
    """
    if not model or not os.path.isdir(model):
        return "path:%s" % (model or "")
    digest = hashlib.sha256()
    for name in sorted(os.listdir(model)):
        if not name.endswith((".safetensors", ".json", ".bin", ".pt", ".model", ".txt")):
            continue
        path = os.path.join(model, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not os.path.isfile(path):
            continue
        digest.update(("%s %d %d\n" % (name, st.st_size, st.st_mtime_ns)).encode())
        # Small metadata files (config, quant config, safetensors index) are
        # hashed by content: a config edit must never hit a stale artifact.
        if name.endswith(".json") and st.st_size <= (4 << 20):
            try:
                with open(path, "rb") as f:
                    digest.update(hashlib.sha256(f.read()).digest())
            except OSError:
                pass
    return digest.hexdigest()


# Constants the model code derives from runtime state while loading (not from
# the config or the environment) and that end up as literals in the traced
# graph, e.g. the device/host row split of the PLE table when the host budget is
# automatic. They are part of the cache key: a graph traced with one value must
# not be reloaded for another.
_BAKED_CONSTANTS: dict[str, str] = {}


def register_baked_constant(name: str, value: object) -> None:
    _BAKED_CONSTANTS[name] = repr(value)


def baked_constants() -> dict[str, str]:
    return dict(sorted(_BAKED_CONSTANTS.items()))


def clear_baked_constants() -> None:
    _BAKED_CONSTANTS.clear()


def extra_compile_factors(mode: str | None = None) -> dict[str, object]:
    """Factors envs.compile_factors() appends when the switch is on."""
    mode = compile_cache_mode() if mode is None else mode
    if mode == MODE_OFF:
        return {}
    return {
        ENV_SWITCH: mode,
        "SX_COMPILE_CACHE_BUILD": build_identity(),
        "SX_COMPILE_CACHE_BAKED": baked_constants(),
    }


def counters_line(counters: Any) -> str:
    """One greppable line with the compile-cache counters of this process."""
    names = (
        "num_models_seen",
        "num_graphs_seen",
        "num_backend_compilations",
        "num_cache_entries_updated",
        "num_compiled_artifacts_saved",
        "num_compiled_artifacts_loaded",
        "num_aot_compiles",
        "num_aot_artifacts_saved",
        "num_aot_artifacts_loaded",
    )
    return " ".join("%s=%d" % (n, getattr(counters, n, -1)) for n in names)
