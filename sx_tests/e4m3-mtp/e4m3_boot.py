# SPDX-License-Identifier: Apache-2.0
"""Load the E4M3 KV + MTP code paths without a vLLM install.

``import vllm`` needs the whole serving stack. What the E4M3 + MTP port
touches is host logic that only needs torch, so ``install()`` registers, under
the real module names:

    vllm.envs                                  the real file (stdlib only)
    vllm.models.qwen4_exp.nvidia.ops.qsa       the real file; Triton, the
                                               platform and the logger are
                                               stubs (no kernel is launched)
    vllm.models.qwen4_exp.nvidia.qsa           cut with ``ast``: the E4M3
                                               gate and the two K/V scale
                                               methods of the QSA owner
    vllm.models.qwen4_exp.nvidia.model         cut: the scale-load helpers
    vllm.models.qwen4_exp.nvidia.mtp           cut: weight-name remap and
                                               Qwen4ExpMTP.load_weights
    vllm.config.vllm                           cut: the SM70 Qwen3.8 lane
                                               contracts and defaults

Where vLLM is installed nothing is replaced and the real modules are used.
Not a test file. It is also a pytest plugin (``-p e4m3_boot``), so upstream
test files that import those modules run unchanged; see
``test_e4m3_mtp_port.py`` for the command lines.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NVIDIA = "vllm.models.qwen4_exp.nvidia"
_PRELUDE = (
    "from __future__ import annotations\n"
    "import math, os\n"
    "from collections.abc import Iterable\n"
    "from typing import Any\n"
    "import torch\n"
    "from torch import nn\n"
)
_OPS: types.ModuleType | None = None


def _path(*parts: str) -> str:
    return os.path.join(REPO, *parts)


def _cut(
    path: str,
    names: tuple[str, ...],
    members: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Source of the top-level ``names`` and of the listed class members."""
    source = open(path, encoding="utf-8").read()
    members = members or {}
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) in names:
                    found[target.id] = ast.get_source_segment(source, node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
        elif isinstance(node, ast.ClassDef) and node.name in members:
            body = [
                ast.get_source_segment(source, item, padded=True)
                for item in node.body
                if getattr(item, "name", None) in members[node.name]
                or any(
                    getattr(target, "id", None) in members[node.name]
                    for target in getattr(item, "targets", ())
                )
            ]
            assert len(body) == len(members[node.name]), (path, node.name)
            found[node.name] = f"class {node.name}(nn.Module):\n" + "\n\n".join(body)
    missing = [name for name in (*names, *members) if name not in found]
    assert not missing, f"{path}: not found: {missing}"
    return "\n\n\n".join(found.values()) + "\n"


def _register(name: str, module: types.ModuleType) -> types.ModuleType:
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    if parent:
        if parent not in sys.modules:
            _module(parent)
        setattr(sys.modules[parent], leaf, module)
    return module


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    module.__package__ = name.rpartition(".")[0]
    module.__dict__.update(attrs)
    return _register(name, module)


def _load_file(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = _register(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


def _cut_module(name: str, path: str, source: str, **attrs: object):
    module = _module(name, **attrs)
    exec(compile(_PRELUDE + source, path, "exec"), module.__dict__)  # noqa: S102
    return module


class _Logger(logging.Logger):
    def info_once(self, *args, **kwargs) -> None:
        pass

    warning_once = debug_once = info_once


def install() -> types.ModuleType:
    """Return the ops/qsa module: real if vLLM imports, otherwise the stubs."""
    global _OPS
    if _OPS is not None:
        return _OPS
    try:
        import vllm.models.qwen4_exp.nvidia.ops.qsa as real  # noqa: PLC0415

        _OPS = real
        return real
    except Exception:  # noqa: BLE001 - any import failure means "not installed"
        pass
    import re

    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]
    sys.modules.setdefault("regex", re)

    _module("vllm")
    envs = _load_file("vllm.envs", _path("vllm", "envs.py"))
    platform = types.SimpleNamespace(
        is_device_capability=lambda capability, *a, **k: capability in (70, (7, 0)),
        has_device_capability=lambda capability, *a, **k: False,
        is_cuda=lambda: False,
    )
    shared = dict(envs=envs, logger=_Logger("sx.e4m3"), current_platform=platform)
    _module("vllm.logger", init_logger=_Logger)
    _module("vllm.platforms", current_platform=platform)
    _module(
        "vllm.triton_utils",
        HAS_TRITON=False,
        tl=types.SimpleNamespace(constexpr=object),
        triton=types.SimpleNamespace(
            jit=lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f),
            cdiv=lambda a, b: -(-a // b),
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
    )
    _module(
        "vllm.models.deepseek_v4.common.ops.fp8_software",
        fp8_e4m3fn_bits_to_fp32_bitcast=None,
    )
    nvidia = _path("vllm", "models", "qwen4_exp", "nvidia")
    ops = _load_file(f"{NVIDIA}.ops.qsa", os.path.join(nvidia, "ops", "qsa.py"))

    attention = _path("vllm", "model_executor", "layers", "attention", "attention.py")
    helpers = _cut_module(
        "sx_e4m3_attention",
        attention,
        _cut(attention, ("set_default_quant_scales",)),
        **shared,
    )
    path = os.path.join(nvidia, "qsa.py")
    qsa = _cut_module(
        f"{NVIDIA}.qsa",
        path,
        _cut(
            path,
            ("_verify_e4m3_kv_requirements",),
            {
                "Qwen4ExpQSAAttention": (
                    "adopt_default_kv_scales",
                    "validate_loaded_kv_scales",
                )
            },
        ),
        set_default_quant_scales=helpers.set_default_quant_scales,
        **shared,
    )
    path = os.path.join(nvidia, "model.py")
    model = _cut_module(
        f"{NVIDIA}.model",
        path,
        _cut(
            path,
            (
                "_remap_qsa_cache_scale_name",
                "_QWEN4_EXP_IGNORED_MISSING_SUFFIXES",
                "_validate_qsa_e4m3_scale_load",
                "_finalize_qsa_e4m3_scale_load",
            ),
        ),
        is_offload_process=lambda: False,
        Qwen4ExpQSAAttention=qsa.Qwen4ExpQSAAttention,
        **shared,
    )
    _module(
        f"{NVIDIA}.mtp_fp8_checkpoint",
        prepare_mtp_fp8_checkpoint=lambda weights, *args, **kwargs: weights,
    )
    path = os.path.join(nvidia, "mtp.py")
    _cut_module(
        f"{NVIDIA}.mtp",
        path,
        _cut(
            path,
            ("_remap_mtp_weight_name", "_validate_mtp_expert_weights_loaded"),
            {"Qwen4ExpMTP": ("allow_patterns_overrides", "load_weights")},
        ),
        AutoWeightsLoader=None,
        _QWEN4_EXP_IGNORED_MISSING_SUFFIXES=model._QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
        **shared,
    )
    path = _path("vllm", "config", "vllm.py")
    _cut_module(
        "vllm.config.vllm",
        path,
        _cut(
            path,
            (
                "_SX_MTP_LANE_MAX_K",
                "_is_sm70_qwen38_nomtp_dual_compile_contract",
                "_sx_env_on",
                "_sx_mtp_lane_enabled",
                "_is_sm70_qwen38_mtp_lane_contract",
                "_is_sm70_qwen38_lane_contract",
                "_sm70_qwen38_lane_qualified",
                "_sm70_rmsnorm_gated_exact_available",
                "_apply_sm70_qwen38_nomtp_defaults",
            ),
        ),
        **shared,
    )
    _OPS = ops
    return ops


def pytest_configure(config) -> None:
    """``-p e4m3_boot``: make the modules importable before test collection."""
    install()
