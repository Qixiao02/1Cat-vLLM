# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the batch-2 "small-native" tests (design_4 MR5 + MR6).

Not a test module. The sibling test_*.py files import it first (pytest puts
this directory on sys.path; script mode runs from it). Requires the rebuilt
extensions and the patched .py files installed into the vllm package that
``import vllm`` resolves (e.g. /opt/venv/lib/python3.12/site-packages/vllm in
the 1.8.0-dev1 image, with the rebuilt _C.abi3.so and
_C_stable_libtorch.abi3.so mounted over the image copies).
"""

from __future__ import annotations

import math
import os
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

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

import torch  # noqa: E402

# Deployment geometry (Qwen3.8-Flash-Next NVFP4, TP4).
HIDDEN = 2560
TOPK = 512  # QSA block top-k = token_topk 2048 / compress_ratio 4
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
INDEX_HEADS = 4
INDEX_DIM = 128
SCHED_BLOCK = 784  # mamba-align scheduler block = indexer page
COMPRESSED_PAGE = SCHED_BLOCK // COMPRESS_RATIO  # 196 compressed rows / page
MAX_MODEL_LEN = 131072
INDEX_TABLE_WIDTH = math.ceil(MAX_MODEL_LEN / SCHED_BLOCK)  # 168
SCORE_COLUMNS = INDEX_TABLE_WIDTH * COMPRESSED_PAGE  # 32928 decode logits width

FULL_M = (1, 2, 3, 4, 8, 16, 17, 20, 24, 32)
QUICK_M = (1, 2, 4, 8, 16, 24, 32)
BENCH_M = (1, 2, 4, 8, 16, 24, 32)


def quick() -> bool:
    return os.environ.get("SX_TEST_QUICK", "0") not in ("", "0")


def no_bench() -> bool:
    return os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0")


def m_values() -> tuple[int, ...]:
    return QUICK_M if quick() else FULL_M


def sm70_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def bits16(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def bit_equal16(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits16(a), bits16(b))


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> str:
    """Human-readable location of the first differing element."""
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    if a.dtype == torch.float16:
        diff = bits16(a) != bits16(b)
    else:
        diff = a != b
    idx = diff.nonzero()
    if idx.numel() == 0:
        return "equal"
    where = tuple(int(v) for v in idx[0])
    return f"{int(diff.sum())} differ, first at {where}: {a[where]} vs {b[where]}"


def capture(fn: Callable[[], object], warmup: int = 3) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def bench_graphs(
    variants: dict[str, Callable[[int], object]],
    calls: int,
    rounds: int = 8,
    per_round: int = 8,
    warmup: int = 5,
) -> dict[str, float]:
    """Median microseconds per call of each variant.

    Each variant is captured once as a CUDA graph of ``calls`` launches (call
    i uses buffer set i, so per-layer operands are distinct as in a decode
    step). Samples alternate between variants round by round (A/B/A...),
    ``rounds * per_round`` replays each.
    """
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, fn in variants.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for i in range(calls):
                fn(i)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(calls):
                fn(i)
        graphs[name] = graph
    for graph in graphs.values():
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    names = list(graphs)
    for rnd in range(rounds):
        for name in names if rnd % 2 == 0 else names[::-1]:
            graph = graphs[name]
            for _ in range(per_round):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0 / calls)
    del graphs
    torch.cuda.synchronize()
    return {name: statistics.median(values) for name, values in samples.items()}


def print_table(title: str, header: list[str], rows: list[list[object]]) -> None:
    widths = [len(h) for h in header]
    text_rows = [[str(c) for c in row] for row in rows]
    for row in text_rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    print(f"\n== {title}")
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for row in text_rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
