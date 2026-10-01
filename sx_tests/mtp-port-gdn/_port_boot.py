# SPDX-License-Identifier: Apache-2.0
"""Cut functions out of the fork's sources so the port can be checked on CPU.

``import vllm`` needs the whole serving stack (and CUDA/Triton for the real
kernels). The gates and host-side logic ported from upstream main@d30469863
only need torch, so the tests here parse the real source files with ``ast``,
take the named top-level functions / assignments (or methods of a class) and
execute them in a namespace whose remaining globals are supplied by the test
(stubs, fakes or the real helpers). Decorators are dropped: ``@triton.jit``
kernels are replaced by Python emulations in the tests that need them.

Not a test file; see ``test_*.py`` next to it.
"""

from __future__ import annotations

import ast
import os
import textwrap
from typing import Any

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG = os.path.join(REPO, "vllm", "config", "vllm.py")
GDN_LAYER = os.path.join(
    REPO, "vllm", "model_executor", "layers", "mamba", "gdn", "qwen_gdn_linear_attn.py"
)
GDN_ATTN = os.path.join(REPO, "vllm", "v1", "attention", "backends", "gdn_attn.py")
ATTN_UTILS = os.path.join(REPO, "vllm", "v1", "worker", "gpu", "attn_utils.py")
BACKEND_UTILS = os.path.join(REPO, "vllm", "v1", "attention", "backends", "utils.py")
MAMBA_HYBRID = os.path.join(
    REPO, "vllm", "v1", "worker", "gpu", "model_states", "mamba_hybrid.py"
)
PLE = os.path.join(REPO, "vllm", "models", "qwen4_exp", "nvidia", "ple_layer.py")
CMAKE = os.path.join(REPO, "CMakeLists.txt")
PLE_CU = os.path.join(REPO, "csrc", "sm70_turbomind", "ops", "qwen38_ple_spec_sm70.cu")


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _top_level_segments(path: str, names: tuple[str, ...]) -> dict[str, str]:
    source = read(path)
    found: dict[str, str] = {}
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            # Keep class decorators (``@dataclass``); functions drop theirs.
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            found[node.name] = "".join(lines[first - 1 : node.end_lineno])
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.get_source_segment(source, node)
    return found


def cut(path: str, names: tuple[str, ...], namespace: dict[str, Any]) -> dict[str, Any]:
    """Exec the named top-level definitions of ``path`` into ``namespace``."""
    found = _top_level_segments(path, names)
    missing = [name for name in names if name not in found]
    assert not missing, f"{os.path.basename(path)}: not found {missing}"
    code = "\n\n".join(found[name] for name in names) + "\n"
    exec(compile(code, path, "exec"), namespace)  # noqa: S102
    return namespace


def method_source(path: str, class_name: str, method: str) -> str:
    source = read(path)
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return textwrap.dedent(
                        ast.get_source_segment(source, item, padded=True)
                    )
    raise AssertionError(f"{class_name}.{method} not found in {path}")


def cut_method(
    path: str, class_name: str, method: str, namespace: dict[str, Any]
) -> Any:
    """Return ``class_name.method`` as a plain function bound to ``namespace``."""
    code = method_source(path, class_name, method)
    exec(compile(code, path, "exec"), namespace)  # noqa: S102
    return namespace[method]


def global_names(path: str, class_name: str | None, function: str) -> set[str]:
    """Names a function (or method) reads as globals (for stub completeness)."""
    if class_name is None:
        code = _top_level_segments(path, (function,))[function]
    else:
        code = method_source(path, class_name, function)
    tree = ast.parse(code)
    assigned: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            else:
                loaded.add(node.id)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, ast.FunctionDef):
            assigned.add(node.name)
    import builtins

    return {name for name in loaded - assigned if not hasattr(builtins, name)}
