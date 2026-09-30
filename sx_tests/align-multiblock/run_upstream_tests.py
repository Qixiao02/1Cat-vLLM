# SPDX-License-Identifier: Apache-2.0
"""Run upstream's own CPU tests of the touched code through the stub loader.

    python sx_tests/align-multiblock/run_upstream_tests.py [pytest arguments]

Where vLLM is installed these files run as they are:

    python -m pytest tests/v1/core/test_mamba_align_chunk_split.py \
        tests/v1/core/test_mamba_sparse_retention.py \
        tests/v1/core/test_single_type_kv_cache_manager.py

This script is for a machine where ``import vllm`` is not possible. It makes
``vllm.*`` resolve to ``align_boot``'s universe (the scheduler and KV-cache
modules of the working tree, everything else stubbed) and hands the
unmodified upstream test files to pytest, without ``tests/conftest.py``.

Two things cannot come from the tree that way and are replaced:
* ``tests/v1/core/utils.py`` builds a real ``VllmConfig`` for a Hugging Face
  model. ``create_scheduler`` and ``create_requests`` below build the same
  scheduler and requests from stand-in configs.
* ``vllm.config.CacheConfig`` is a stub, so
  ``test_mamba_sparse_retention.py::test_upstream_config_semantics`` (the
  field's default and hash exclusion) is deselected.

Files:
* ``test_mamba_align_chunk_split.py``: ``_mamba_block_aligned_split`` on an
  object without any ``_sx_*`` attribute, and ``Scheduler.__init__``;
* ``test_mamba_sparse_retention.py``: ``MambaManager`` allocation, freeing
  and sparse retention, ``get_replay_boundaries``;
* ``test_single_type_kv_cache_manager.py``: the single-type managers;
* ``test_prefix_caching.py``: the two ``mamba_align`` cases (null blocks in
  ``cache_blocks``, relocated speculative blocks in a multi-block chunk).
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align_boot  # noqa: E402
import align_harness  # noqa: E402

TESTS = [
    "tests/v1/core/test_mamba_align_chunk_split.py",
    "tests/v1/core/test_mamba_sparse_retention.py",
    "tests/v1/core/test_single_type_kv_cache_manager.py",
    "tests/v1/core/test_prefix_caching.py::test_prefill_hybrid_model_mamba_align",
    "tests/v1/core/test_prefix_caching.py"
    "::test_mamba_align_offload_handoff_tracks_relocated_mtp_boundary",
]
DESELECT = [
    "tests/v1/core/test_mamba_sparse_retention.py::test_upstream_config_semantics",
]


def _utils_shim(universe: align_boot.Universe) -> types.ModuleType:
    """Stand-in for ``tests/v1/core/utils.py``."""
    import torch

    kvi = universe.kv_cache_interface

    def create_scheduler(max_num_seqs: int = 16, max_num_batched_tokens: int = 8192,
                         enable_prefix_caching: bool = False,
                         long_prefill_token_threshold: int = 0,
                         num_blocks: int = 10000, block_size: int = 16,
                         max_model_len: int | None = None,
                         async_scheduling: bool = False, kv_cache_spec=None):
        vllm_config = align_harness.make_vllm_config(
            torch=torch, model="facebook/opt-125m", block_size=block_size,
            num_blocks=num_blocks, budget=max_num_batched_tokens,
            threshold=long_prefill_token_threshold, max_num_seqs=max_num_seqs,
            max_model_len=max_model_len or max_num_batched_tokens, spec=None,
            retention=0, tp=1, runner_v2=False,
            queue_depth=2 if async_scheduling else 1, connector=False,
            cache_mode="none",
        )
        vllm_config.cache_config.enable_prefix_caching = enable_prefix_caching
        if kv_cache_spec is None:
            kv_cache_spec = kvi.FullAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=1,
                dtype=torch.float32,
            )
        kv_cache_config = kvi.KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[kvi.KVCacheGroupSpec(["layer"], kv_cache_spec)],
        )
        scheduler_cls = (
            universe.async_scheduler.AsyncScheduler
            if async_scheduling
            else universe.scheduler.Scheduler
        )
        return scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=block_size,
            structured_output_manager=align_harness._FakeStructuredOutputManager(),
            mm_registry=align_harness._FakeMMRegistry(),
        )

    def create_requests(num_requests: int, num_tokens: int = 10,
                        max_tokens: int = 16, same_prompt: bool = False,
                        block_size: int = 16, req_ids: list[str] | None = None):
        block_hasher = universe.kv_cache_utils.get_request_block_hasher(
            block_size, align_boot.sha256
        )
        req_ids = req_ids or [f"{i}" for i in range(num_requests)]
        return [
            universe.request.Request(
                request_id=req_ids[i],
                prompt_token_ids=[0 if same_prompt else i] * num_tokens,
                sampling_params=align_boot.SamplingParams(max_tokens=max_tokens),
                pooling_params=None,
                block_hasher=block_hasher,
            )
            for i in range(num_requests)
        ]

    shim = types.ModuleType("tests.v1.core.utils")
    shim.create_scheduler = create_scheduler
    shim.create_requests = create_requests
    shim.mock_kv = lambda **kwargs: SimpleNamespace(**kwargs)
    return shim


def main(argv: list[str]) -> int:
    universe = align_boot.load()
    os.chdir(align_boot.REPO)
    sys.path.insert(0, str(align_boot.REPO))
    with universe.activate():
        sys.modules["tests.v1.core.utils"] = _utils_shim(universe)
        try:
            return pytest.main(
                [
                    "--noconftest",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    *(f"--deselect={name}" for name in DESELECT),
                    *TESTS,
                    *argv,
                ]
            )
        finally:
            sys.modules.pop("tests.v1.core.utils", None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
