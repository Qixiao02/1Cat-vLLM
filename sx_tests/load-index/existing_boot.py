# SPDX-License-Identifier: Apache-2.0
"""Run repo tests that import vllm.model_executor.models.utils without a vLLM install.

``import vllm.model_executor.models.utils`` needs transformers, the whole serving
stack and a platform probe. This pytest plugin registers, under the real module
names, the pieces tests/models/test_utils.py uses:

    vllm.model_executor.models.utils   AutoWeightsLoader, WeightsMapper,
                                       maybe_fuse_shared_experts, maybe_prefix
                                       (cut from the real file by ``ast``) and a
                                       ``_merge_multimodal_embeddings`` that
                                       fails if called (its only test is
                                       CUDA-only and skipped)
    vllm.platforms                     current_platform: CPU, not CUDA

Use (``--noconftest`` keeps tests/conftest.py, which imports vLLM, out):

    PYTHONPATH=sx_tests/load-index python -m pytest -q --noconftest \
        -p existing_boot tests/models/test_utils.py

Where vLLM is installed none of this is needed and the plain command works.
Not a test file.
"""

from __future__ import annotations

import sys
import types

import pipeline_harness


def install() -> None:
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.models"):
        package(name)
    ns = pipeline_harness.utils_namespace()
    utils = sys.modules.setdefault(
        "vllm.model_executor.models.utils",
        types.ModuleType("vllm.model_executor.models.utils"),
    )
    for key in ("AutoWeightsLoader", "WeightsMapper", "maybe_fuse_shared_experts",
                "maybe_prefix"):
        setattr(utils, key, ns[key])

    def _merge_multimodal_embeddings(*args, **kwargs):
        raise RuntimeError("not available without a vLLM install")

    utils._merge_multimodal_embeddings = _merge_multimodal_embeddings
    platforms = package("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(
        device_type="cpu", is_cuda=lambda: False
    )


def pytest_configure(config) -> None:
    install()
