# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the sx_tests/qsa GPU checks (single V100, SM70).

``new_ops()`` is the live ``vllm.models.qwen4_exp.nvidia.ops.qsa`` (with the
SX_OPT_QSA_* changes) and ``old_ops()`` is a byte-identical copy of the git
baseline 71c1822 module (see _baseline_ops_qsa.py), so every comparison is
new-vs-old implementation on the same inputs.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import statistics
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _prefer_installed_vllm() -> None:
    """Keep ``import vllm`` on the installed (overlaid) package.

    ``python -m pytest`` run from the 1Cat source-tree root puts that root on
    sys.path, which would shadow the image's compiled vllm with the
    uncompiled sources. Drop it unless it actually carries compiled
    extensions.
    """
    repo_root = HERE.parents[1]
    if any((repo_root / "vllm").glob("_C*.so")):
        return
    for entry in list(sys.path):
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            continue
        if resolved == repo_root:
            sys.path.remove(entry)


_prefer_installed_vllm()

# Deployment geometry (Qwen3.8-Flash-Next NVFP4, TP4, FP16 KV).
Q_HEADS = 6
HEAD_DIM = 256
SELECTION = 2051  # token_topk 2048 + compress_ratio 4 - 1
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
INDEX_HEADS = 4
INDEX_DIM = 128
SCHED_BLOCK = 784  # mamba-align scheduler block
COMPRESSED_PAGE = SCHED_BLOCK // COMPRESS_RATIO  # 196 compressed rows / page
MAX_MODEL_LEN = 131072
INDEX_TABLE_WIDTH = math.ceil(MAX_MODEL_LEN / SCHED_BLOCK)  # 168


def is_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def require_sm70():
    import pytest

    if not is_sm70():
        pytest.skip("requires a V100 / SM70 GPU")


_OLD = None


def new_ops():
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    return qsa


def old_ops():
    global _OLD
    if _OLD is None:
        new_ops()  # load the live module (and any sidecar op) first
        spec = importlib.util.spec_from_file_location(
            "sx_baseline_ops_qsa", HERE / "_baseline_ops_qsa.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["sx_baseline_ops_qsa"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _OLD = module
    return _OLD


@contextlib.contextmanager
def patched(module, **values) -> Iterator[None]:
    """Temporarily override module-level switches (e.g. _SX_OPT_QSA_*)."""
    saved = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.float16, torch.bfloat16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def median_ms(fn: Callable[[], object], iters: int = 50, warmup: int = 10) -> float:
    """Median per-call CUDA-event time of ``fn`` (eager)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def paired_graph_ms(
    graphs: dict[str, torch.cuda.CUDAGraph], iters: int = 50, inner: int = 20
) -> dict[str, float]:
    """Alternate A/B graph replays; median over ``iters`` samples of ``inner``."""
    for graph in graphs.values():
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    names = list(graphs)
    for turn in range(iters):
        order = names if turn % 2 == 0 else list(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                graphs[name].replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) / inner)
    return {name: statistics.median(values) for name, values in samples.items()}


def report(title: str, values: dict) -> None:
    items = ", ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in values.items()
    )
    print(f"[sx-qsa-bench] {title}: {items}", flush=True)


def env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------
def sample_row_selection(
    generator: torch.Generator,
    visible_tokens: int,
    base_blocks: torch.Tensor | None = None,
    resample_fraction: float = 0.0,
) -> torch.Tensor:
    """Return a [2051] int32 token selection shaped like the QSA indexer output.

    512 compressed blocks (x4 tokens) out of the complete groups, then the
    causal tail of the open group, then -1 padding.
    """
    row = torch.full((SELECTION,), -1, dtype=torch.int32)
    complete = visible_tokens // COMPRESS_RATIO
    if complete > 0:
        target = min(512, complete)
        if base_blocks is not None:
            # Adjacent prefill rows share most of their selected blocks: keep
            # the request's base selection, drop a few, top up with new ones.
            blocks = base_blocks[base_blocks < complete]
            if resample_fraction > 0 and blocks.numel():
                keep = torch.rand(blocks.numel(), generator=generator) >= (
                    resample_fraction
                )
                blocks = blocks[keep]
            blocks = blocks[:target]
            if blocks.numel() < target:
                candidates = torch.randperm(complete, generator=generator)
                candidates = candidates[~torch.isin(candidates, blocks)]
                blocks = torch.cat((blocks, candidates[: target - blocks.numel()]))
            blocks = blocks.sort().values
        else:
            blocks = torch.randperm(complete, generator=generator)[:512].sort().values
        full = (blocks[:, None] * COMPRESS_RATIO + torch.arange(COMPRESS_RATIO)).flatten()
        row[: full.numel()] = full.to(torch.int32)
        tail_start = complete * COMPRESS_RATIO
        tail = torch.arange(tail_start, visible_tokens, dtype=torch.int32)
        row[full.numel() : full.numel() + tail.numel()] = tail
    else:
        tail = torch.arange(0, visible_tokens, dtype=torch.int32)
        row[: tail.numel()] = tail
    return row


def make_block_table(
    generator: torch.Generator,
    contexts: list[int],
    page_size: int,
    width: int,
    spare_pages: int = 3,
) -> tuple[torch.Tensor, int]:
    """Random physical pages for each request; unused entries are -1."""
    needed = [math.ceil(max(context, 1) / page_size) for context in contexts]
    total = sum(needed) + spare_pages
    physical = torch.randperm(total, generator=generator).to(torch.int32)
    table = torch.full((len(contexts), width), -1, dtype=torch.int32)
    cursor = 0
    for request, count in enumerate(needed):
        table[request, :count] = physical[cursor : cursor + count]
        cursor += count
    return table, total
