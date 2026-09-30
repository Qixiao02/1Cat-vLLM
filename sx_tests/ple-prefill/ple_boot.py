# SPDX-License-Identifier: Apache-2.0
"""Load the PLE short-conv prefill code without a vLLM install.

``import vllm`` needs the whole serving stack. The two prefill methods of
``Qwen4ExpPLELayer`` only need torch, so this module cuts them (and the
packed-rows bound with its grouping helper) out of the source file with
``ast`` and registers them under the real module names:

    vllm.models.qwen4_exp.nvidia.ple_layer   Qwen4ExpPLELayer with the two
                                             methods, ``_sx_ple_prefill_groups``
                                             and ``_SX_PLE_PREFILL_MAX_PACKED_ROWS``
    vllm.v1.attention.backends.utils         NULL_BLOCK_ID

Where vLLM is installed nothing is replaced and the real modules are used.
Not a test file; see ``test_ple_prefill_grouped.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLE = os.path.join(REPO, "vllm", "models", "qwen4_exp", "nvidia", "ple_layer.py")
UTILS = os.path.join(REPO, "vllm", "v1", "attention", "backends", "utils.py")
UPSTREAM_TEST = os.path.join(
    REPO, "tests", "models", "qwen4_exp", "test_ple_short_conv_prefill.py"
)
METHODS = ("_short_conv_dilated_prefill_batched", "_short_conv_dilated_prefill_packed")


def _stub_source() -> str:
    source = open(PLE, encoding="utf-8").read()
    tree = ast.parse(source)
    parts = [
        "import os",
        "from collections.abc import Sequence",
        "import torch",
        "from torch import nn",
        "from torch.nn import functional as F",
        "from vllm.v1.attention.backends.utils import NULL_BLOCK_ID",
        "PleShortConvAttentionMetadata = object",
    ]
    for node in tree.body:
        segment = ast.get_source_segment(source, node)
        if isinstance(node, ast.Try) and "_SX_PLE_PREFILL_MAX_PACKED_ROWS" in segment:
            parts.append(segment)
        elif isinstance(node, ast.FunctionDef) and node.name == "_sx_ple_prefill_groups":
            parts.append(segment)
        elif isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpPLELayer":
            methods = [
                ast.get_source_segment(source, item, padded=True)
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in METHODS
            ]
            assert len(methods) == len(METHODS), "prefill methods not found"
            parts.append("class Qwen4ExpPLELayer(nn.Module):\n" + "\n\n".join(methods))
    return "\n\n".join(parts) + "\n"


def install() -> types.ModuleType:
    """Return the ple_layer module, real if vLLM imports, otherwise the cut-out."""
    try:
        import vllm.models.qwen4_exp.nvidia.ple_layer as real  # noqa: PLC0415

        return real
    except Exception:  # noqa: BLE001 - any import failure means "not installed"
        pass
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]
    null_block = int(
        re.search(r"(?m)^NULL_BLOCK_ID = (\d+)$", open(UTILS, encoding="utf-8").read())[1]
    )
    names = (
        "vllm",
        "vllm.models",
        "vllm.models.qwen4_exp",
        "vllm.models.qwen4_exp.nvidia",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.utils",
        "vllm.models.qwen4_exp.nvidia.ple_layer",
    )
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    sys.modules["vllm.v1.attention.backends.utils"].NULL_BLOCK_ID = null_block
    ple = sys.modules["vllm.models.qwen4_exp.nvidia.ple_layer"]
    exec(compile(_stub_source(), PLE, "exec"), ple.__dict__)  # noqa: S102
    return ple


def upstream_test_module() -> types.ModuleType:
    """The upstream test file as a module (its oracle and case builder)."""
    install()
    spec = importlib.util.spec_from_file_location("ple_upstream_test", UPSTREAM_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
