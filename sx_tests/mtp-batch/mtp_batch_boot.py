# SPDX-License-Identifier: Apache-2.0
"""Load sm70_fp16_gemv / sm70_fp16_hc for CPU gating tests without a vLLM install.

The two route modules only need torch, vllm.envs, vllm.logger,
vllm.triton_utils (its placeholder works without triton),
vllm.utils.torch_utils and vllm.compilation.sm70_decode_graph, which all
import on a bare host. Their heavier imports are replaced by small stand-ins:

    vllm.platforms                      current_platform (CPU dispatch key,
                                        is_device_capability -> False)
    vllm.config                         get_current_vllm_config(_or_none)
    vllm.config.vllm                    the real SX MTP-lane contract
                                        functions, cut out of the source with
                                        ``ast`` (no other config code)
    vllm.model_executor.layers.linear   LinearBase / UnquantizedLinearMethod
    vllm.distributed.parallel_state     get_tp_group() -> ``TP_GROUP``
    vllm._custom_ops                    supports_sm70_qwen38_hc_batch()
                                        -> ``HC_BATCH_OP``

The route modules themselves are the real files of this tree, loaded under
their package names (the package ``__init__`` files are not executed).
Where vLLM is installed (the serving image) nothing is replaced and the real
modules are used. Not a test file; see ``test_mtp_batch_cpu.py``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NVIDIA = os.path.join(REPO, "vllm", "models", "qwen4_exp", "nvidia")
CONFIG_VLLM = os.path.join(REPO, "vllm", "config", "vllm.py")
CONTRACT_NAMES = (
    "_SX_MTP_LANE_MAX_K",
    "_sx_env_on",
    "_sx_mtp_lane_enabled",
    "_is_sm70_qwen38_nomtp_dual_compile_contract",
    "_is_sm70_qwen38_mtp_lane_contract",
)

STUBBED = False
# Mutable stand-in state (stub mode only).
TP_GROUP: list[object] = [None]
HC_BATCH_OP = [True]
CURRENT_CONFIG: list[object] = [None]


def _contract_source() -> str:
    source = open(CONFIG_VLLM, encoding="utf-8").read()
    parts = ["import os", "from typing import Any", "import torch"]
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in CONTRACT_NAMES:
            name = node.name
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in CONTRACT_NAMES
        ):
            name = node.targets[0].id
        else:
            continue
        parts.append(ast.get_source_segment(source, node))
        found.add(name)
    missing = set(CONTRACT_NAMES) - found
    assert not missing, f"contract pieces not found in vllm/config/vllm.py: {missing}"
    return "\n\n".join(parts) + "\n"


def _package(name: str, path: str | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [path] if path else []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


class _UnquantizedLinearMethod:
    def process_weights_after_loading(self, layer) -> None:
        del layer

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _LinearBase(torch.nn.Module):
    pass


def _get_current_vllm_config():
    config = CURRENT_CONFIG[0]
    if config is None:
        raise AssertionError("no current vLLM config (stub)")
    return config


def _install_stubs() -> None:
    # Modules that imported during the failed attempt stay (re-importing
    # vllm.utils.torch_utils would register its opaque types twice); the
    # names below are (re)placed by stand-ins.
    importlib.import_module("vllm")
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        dispatch_key="CPU",
        is_device_capability=lambda capability: False,
        is_cuda=lambda: False,
        is_arch_support_pdl=lambda: False,
    )
    sys.modules["vllm.platforms"] = platforms

    config = _package("vllm.config")
    config.get_current_vllm_config = _get_current_vllm_config
    config.get_current_vllm_config_or_none = lambda: CURRENT_CONFIG[0]
    config_vllm = types.ModuleType("vllm.config.vllm")
    code = compile(_contract_source(), CONFIG_VLLM, "exec")
    exec(code, config_vllm.__dict__)  # noqa: S102
    sys.modules["vllm.config.vllm"] = config_vllm
    config.vllm = config_vllm

    _package("vllm.model_executor")
    _package("vllm.model_executor.layers")
    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearBase = _LinearBase
    linear.UnquantizedLinearMethod = _UnquantizedLinearMethod
    sys.modules["vllm.model_executor.layers.linear"] = linear

    _package("vllm.distributed")
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tp_group = lambda: TP_GROUP[0]
    # vllm.logger's *_once scopes.
    parallel_state.is_global_first_rank = lambda: True
    parallel_state.is_local_first_rank = lambda: True
    sys.modules["vllm.distributed.parallel_state"] = parallel_state

    custom_ops = types.ModuleType("vllm._custom_ops")
    custom_ops.supports_sm70_qwen38_hc_batch = lambda: HC_BATCH_OP[0]
    sys.modules["vllm._custom_ops"] = custom_ops
    sys.modules["vllm"]._custom_ops = custom_ops

    _package("vllm.models", os.path.join(REPO, "vllm", "models"))
    _package("vllm.models.qwen4_exp", os.path.join(REPO, "vllm", "models", "qwen4_exp"))
    _package("vllm.models.qwen4_exp.nvidia", NVIDIA)


def install() -> tuple[types.ModuleType, types.ModuleType]:
    """Return (sm70_fp16_gemv, sm70_fp16_hc): real if vLLM imports, else stubbed."""
    global STUBBED
    if importlib.util.find_spec("vllm") is None:
        sys.path.append(REPO)  # bare host: this tree's sources
    name = "vllm.models.qwen4_exp.nvidia."
    if name + "sm70_fp16_hc" in sys.modules:
        return sys.modules[name + "sm70_fp16_gemv"], sys.modules[name + "sm70_fp16_hc"]
    try:
        gemv = importlib.import_module(name + "sm70_fp16_gemv")
        hc = importlib.import_module(name + "sm70_fp16_hc")
        return gemv, hc
    except Exception:  # noqa: BLE001 - any import failure means "not installed"
        pass
    _install_stubs()
    STUBBED = True
    gemv = importlib.import_module(name + "sm70_fp16_gemv")
    hc = importlib.import_module(name + "sm70_fp16_hc")
    return gemv, hc


def text_config(**overrides) -> SimpleNamespace:
    values = dict(
        hidden_size=2560,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        num_attention_heads=24,
        num_key_value_heads=2,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def model_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        multimodal_config=SimpleNamespace(language_model_only=True),
        dtype=torch.float16,
        quantization="modelopt_fp4",
        hf_text_config=text_config(),
    )


def mtp_spec(k: int = 4, method: str = "mtp") -> SimpleNamespace:
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        parallel_drafting=False,
        rejection_sample_method="standard",
        target_model_config=model_config(),
        draft_model_config=None,
    )
    spec.use_qwen4_exp_mtp = lambda: method == "mtp"
    spec.num_speculative_state_tokens = lambda: k
    return spec


def lane_config(k: int | None = 4, method: str = "mtp", tp: int = 4) -> SimpleNamespace:
    """A VllmConfig stand-in: the MTP lane (k given) or the no-MTP lane (None)."""
    return SimpleNamespace(
        model_config=model_config(),
        speculative_config=None if k is None else mtp_spec(k, method),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
            enable_expert_parallel=False,
            use_ubatching=False,
        ),
    )
