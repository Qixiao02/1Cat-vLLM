# SPDX-License-Identifier: Apache-2.0
"""Load the code paths of the "mtp-port-moe" group without a vLLM install.

``import vllm`` needs the whole serving stack (and a CUDA build). What this
group changes is host-side dispatch logic around three native/Triton paths,
so ``install()`` registers, under the real module names:

    vllm.envs                                    the real file (stdlib only)
    vllm._sm70_ops                               the real file (every has_*
                                                 probe is False on CPU)
    vllm.config.vllm                             cut with ``ast``: the SM70
                                                 Qwen3.8 lane contracts and a
                                                 settable current config
    vllm.model_executor.layers.quantization.nvfp4_sm70_moe
    vllm.model_executor.layers.fused_moe.fused_moe
    vllm.model_executor.layers.fused_moe.router.fused_topk_router
                                                 the real files; their other
                                                 vLLM imports, Triton and the
                                                 platform are permissive stubs
                                                 (no kernel is ever launched)

Where vLLM is installed nothing is replaced and the real modules are used.
Not a test file. It is also a pytest plugin (``-p port_boot``), so upstream
test files that import those modules run unchanged on CPU (their GPU cases
skip); see ``test_port_cpu.py`` for the command lines.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOE_PKG = "vllm.model_executor.layers.fused_moe"
QUANT_PKG = "vllm.model_executor.layers.quantization"
_PRELUDE = (
    "from __future__ import annotations\n"
    "import os\n"
    "from typing import Any\n"
    "import torch\n"
)
_INSTALLED: dict[str, types.ModuleType] = {}


def _path(*parts: str) -> str:
    return os.path.join(REPO, *parts)


class _Stub:
    """Callable / attribute / item placeholder for imports nobody calls."""

    def __init__(self, name: str = "stub"):
        self._name = name

    def __call__(self, *args, **kwargs):
        return _Stub(f"{self._name}()")

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Stub(f"{self._name}.{name}")
        object.__setattr__(self, name, value)
        return value

    def __getitem__(self, item):
        return _Stub(f"{self._name}[]")

    def __or__(self, other):  # annotations such as ``Stub | None``
        return self

    __ror__ = __or__

    def __repr__(self) -> str:
        return f"<stub {self._name}>"


class _PermissiveModule(types.ModuleType):
    """Missing attributes become cached stubs (stable identity)."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Stub(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def _register(name: str, module: types.ModuleType) -> types.ModuleType:
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    if parent:
        if parent not in sys.modules:
            _module(parent)
        setattr(sys.modules[parent], leaf, module)
    return module


def _module(name: str, permissive: bool = False, **attrs: object):
    cls = _PermissiveModule if permissive else types.ModuleType
    module = cls(name)
    module.__path__ = []  # type: ignore[attr-defined]
    module.__package__ = name.rpartition(".")[0]
    module.__dict__.update(attrs)
    return _register(name, module)


def _load_file(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = []  # type: ignore[attr-defined]
    _register(name, module)
    spec.loader.exec_module(module)
    return module


def _cut(path: str, names: tuple[str, ...]) -> str:
    """Source of the listed top-level assignments and functions."""
    source = open(path, encoding="utf-8").read()
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) in names:
                    found[target.id] = ast.get_source_segment(source, node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
    missing = [name for name in names if name not in found]
    assert not missing, f"{path}: not found: {missing}"
    return "\n\n\n".join(found[name] for name in names) + "\n"


class _Logger(logging.Logger):
    def info_once(self, *args, **kwargs) -> None:
        pass

    warning_once = debug_once = info_once


def _triton_jit(fn=None, **kwargs):
    return fn if fn is not None else (lambda f: f)


class Platform:
    """current_platform stub; tests flip ``capability``."""

    capability = None  # e.g. 70

    def is_device_capability(self, capability, *args, **kwargs) -> bool:
        if self.capability is None:
            return False
        if isinstance(capability, tuple):
            capability = capability[0] * 10 + capability[1]
        return capability == self.capability

    def has_device_capability(self, capability, *args, **kwargs) -> bool:
        if self.capability is None:
            return False
        if isinstance(capability, tuple):
            capability = capability[0] * 10 + capability[1]
        return self.capability >= capability

    def is_cuda(self) -> bool:
        return self.capability is not None

    def is_rocm(self) -> bool:
        return False

    def import_kernels(self) -> None:
        pass


PLATFORM = Platform()


class _BaseRouter:
    def __init__(self, top_k, global_num_experts, eplb_state=None,
                 indices_type_getter=None):
        self.top_k = top_k
        self.global_num_experts = global_num_experts


class _ModelOptNvFp4FusedMoE:
    pass


class CurrentConfig:
    """Settable ``get_current_vllm_config_or_none`` of the cut config module."""

    value = None


def _current_config():
    return CurrentConfig.value


def install() -> dict[str, types.ModuleType]:
    """Return {short name: module}, real if vLLM imports, otherwise stubs."""
    if _INSTALLED:
        return _INSTALLED
    try:
        from vllm.model_executor.layers.fused_moe import (  # noqa: PLC0415
            fused_moe as real_fused_moe,
        )
        from vllm.model_executor.layers.fused_moe.router import (  # noqa: PLC0415
            fused_topk_router as real_router,
        )
        from vllm.model_executor.layers.quantization import (  # noqa: PLC0415
            nvfp4_sm70_moe as real_moe,
        )

        _INSTALLED.update(
            moe=real_moe, router=real_router, fused_moe=real_fused_moe
        )
        return _INSTALLED
    except Exception:  # noqa: BLE001 - any import failure means "not installed"
        pass
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]

    _module("vllm")
    envs = _load_file("vllm.envs", _path("vllm", "envs.py"))
    _module("vllm.logger", init_logger=_Logger)
    _module("vllm.platforms", current_platform=PLATFORM)
    tl = _PermissiveModule("triton.language")
    _module(
        "vllm.triton_utils",
        HAS_TRITON=False,
        tl=tl,
        triton=types.SimpleNamespace(
            jit=_triton_jit,
            cdiv=lambda a, b: -(-a // b),
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
    )
    _module("vllm.utils", permissive=True)
    _module("vllm.utils.torch_utils", direct_register_custom_op=lambda *a, **k: None)
    _module("vllm._custom_ops", permissive=True)
    _module("vllm._aiter_ops", permissive=True)
    _module("vllm.distributed.eplb.eplb_state", permissive=True)
    _module("vllm.forward_context", permissive=True)
    sm70_ops = _load_file("vllm._sm70_ops", _path("vllm", "_sm70_ops.py"))

    path = _path("vllm", "config", "vllm.py")
    config = _module("vllm.config.vllm", envs=envs)
    source = _cut(
        path,
        (
            "_SX_MTP_LANE_MAX_K",
            "_is_sm70_qwen38_nomtp_dual_compile_contract",
            "_sx_env_on",
            "_sx_mtp_lane_enabled",
            "_is_sm70_qwen38_mtp_lane_contract",
        ),
    )
    exec(compile(_PRELUDE + source, path, "exec"), config.__dict__)  # noqa: S102
    config.get_current_vllm_config_or_none = _current_config

    _module("vllm.model_executor", permissive=True)
    _module("vllm.model_executor.layers", permissive=True)
    _module(MOE_PKG, permissive=True)
    _module(f"{MOE_PKG}.modular_kernel", permissive=True)
    _module(f"{MOE_PKG}.activation", permissive=True)
    _module(f"{MOE_PKG}.config", permissive=True)
    _module(f"{MOE_PKG}.moe_align_block_size", permissive=True)
    _module(f"{MOE_PKG}.utils", permissive=True)
    _module(f"{MOE_PKG}.router", permissive=True)
    _module(f"{MOE_PKG}.router.base_router", BaseRouter=_BaseRouter)
    moe_dir = _path("vllm", "model_executor", "layers", "fused_moe")
    router = _load_file(
        f"{MOE_PKG}.router.fused_topk_router",
        os.path.join(moe_dir, "router", "fused_topk_router.py"),
    )
    fused_moe = _load_file(
        f"{MOE_PKG}.fused_moe", os.path.join(moe_dir, "fused_moe.py")
    )

    _module(QUANT_PKG, permissive=True)
    _module(
        f"{QUANT_PKG}.modelopt",
        permissive=True,
        ModelOptNvFp4FusedMoE=_ModelOptNvFp4FusedMoE,
    )
    _module(
        f"{QUANT_PKG}.sm70_turbomind",
        permissive=True,
        NVFP4_GROUP_SIZE=16,
        is_exact_sm70_cuda=lambda *a, **k: True,
    )
    moe = _load_file(
        f"{QUANT_PKG}.nvfp4_sm70_moe",
        _path("vllm", "model_executor", "layers", "quantization", "nvfp4_sm70_moe.py"),
    )
    _INSTALLED.update(
        moe=moe,
        router=router,
        fused_moe=fused_moe,
        envs=envs,
        sm70_ops=sm70_ops,
        config=config,
    )
    return _INSTALLED


def pytest_configure(config) -> None:
    """``-p port_boot``: make the modules importable before test collection."""
    install()
