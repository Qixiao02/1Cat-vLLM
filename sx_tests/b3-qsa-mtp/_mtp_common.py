# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the batch-3a "qsa-mtp" tests (not a test module).

``new_ops()`` is the live ``vllm.models.qwen4_exp.nvidia.ops.qsa`` (with the
SX_OPT_QSA_MTP_* changes); ``dev2_ops()`` is a byte-identical copy of the
same module at tag sx-1.8.0-dev2 (b53c180, the vllm package of image
shixiang/1cat-vllm-v100:1.8.0-dev2-sm70main; sha256 2e4c55b5...3505ad), so
every comparison is new-vs-dev2 implementation on identical inputs.

Run inside image 1.8.0-dev2 with the three patched files bind-mounted over
/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/
{qsa.py,indexer_qsa.py,ops/qsa.py} and this directory mounted anywhere:
  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-qsa-mtp
GPU: ONE V100 (SM70). SX_TEST_QUICK=1 shrinks the grids, SX_TEST_NO_BENCH=1
skips the microbenchmarks.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import statistics
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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

# Deployment geometry (Qwen3.8-Flash-Next NVFP4, TP4-local, FP16 KV).
HIDDEN = 2560
Q_HEADS = 6
HEAD_DIM = 256
SELECTION = 2051  # token_topk 2048 + compress_ratio 4 - 1
TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO  # 512
INDEX_HEADS = 4
INDEX_DIM = 128
SCHED_BLOCK = 784  # mamba-align scheduler block = main KV page
COMPRESSED_PAGE = SCHED_BLOCK // COMPRESS_RATIO  # 196 compressed rows / page
MAX_MODEL_LEN = 131072
TABLE_WIDTH = math.ceil(MAX_MODEL_LEN / SCHED_BLOCK)  # 168

# Verify geometry: B requests x q = k + 1 rows (k in 1..4).
VERIFY_B = (1, 2, 4, 8, 12, 16, 24)
VERIFY_Q = (2, 3, 4, 5)
CONTEXTS = (2048, 8192)
QUICK_B = (1, 4, 8, 12, 24)
QUICK_Q = (2, 5)


def quick() -> bool:
    return os.environ.get("SX_TEST_QUICK", "0") not in ("", "0")


def no_bench() -> bool:
    return os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0")


def verify_grid() -> list[tuple[int, int]]:
    bs = QUICK_B if quick() else VERIFY_B
    qs = QUICK_Q if quick() else VERIFY_Q
    return [(b, q) for b in bs for q in qs]


def is_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def require_sm70() -> None:
    import pytest

    if not is_sm70():
        pytest.skip("requires a V100 / SM70 GPU")


def new_ops():
    from vllm.models.qwen4_exp.nvidia.ops import qsa

    return qsa


_DEV2 = None


@contextlib.contextmanager
def _tolerant_module_load() -> Iterator[None]:
    """Load a second copy of ops/qsa.py next to the live one.

    The copy must not load the optional sidecar library a second time or
    re-register the sidecar fake kernels the live module already registered.
    """
    saved_env = os.environ.pop("VLLM_SM70_QSA_TOPK_LIBRARY", None)
    original = torch.library.register_fake

    def register_fake(name, *args, **kwargs):
        def decorator(fn):
            try:
                return original(name, *args, **kwargs)(fn)
            except Exception:  # noqa: BLE001 - already registered by live copy
                return fn

        return decorator

    torch.library.register_fake = register_fake
    try:
        yield
    finally:
        torch.library.register_fake = original
        if saved_env is not None:
            os.environ["VLLM_SM70_QSA_TOPK_LIBRARY"] = saved_env


def dev2_ops():
    global _DEV2
    if _DEV2 is None:
        new_ops()  # the live module (and any sidecar op) first
        spec = importlib.util.spec_from_file_location(
            "sx_dev2_ops_qsa", HERE / "_dev2_ops_qsa.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["sx_dev2_ops_qsa"] = module
        assert spec.loader is not None
        with _tolerant_module_load():
            spec.loader.exec_module(module)
        _DEV2 = module
    return _DEV2


def mtp_lane(page4_graph_rows: int = 0, rows: int = 63):
    """MTP-lane options as the layer builds them (defaults: every cap 63)."""
    ops = new_ops()
    return ops.SxQsaMtpLane(rows, rows, rows, page4_graph_rows)


@contextlib.contextmanager
def patched(module, **values) -> Iterator[None]:
    """Temporarily override module-level names (switches, kernels)."""
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


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    if a.dtype == torch.float16:
        diff = a.view(torch.int16) != b.view(torch.int16)
    else:
        diff = a != b
    idx = diff.nonzero()
    if idx.numel() == 0:
        return "equal"
    where = tuple(int(v) for v in idx[0])
    return f"{int(diff.sum())} differ, first at {where}: {a[where]} vs {b[where]}"


class LaunchSpy:
    """Wrap a Triton kernel; record the keyword arguments of every launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches: list[dict] = []

    def __getitem__(self, grid):
        launcher = self.kernel[grid]

        def run(*args, **kwargs):
            self.launches.append(kwargs)
            return launcher(*args, **kwargs)

        return run


class CallSpy:
    """Wrap a callable (e.g. a torch op); count calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def capture(
    fn: Callable[[], object], warmup: int = 2, pool=None
) -> torch.cuda.CUDAGraph:
    """vLLM-like capture: eager warmup on a side stream, then capture on the
    private stream torch.cuda.graph() uses (a different stream id). ``pool``
    shares one graph memory pool across captures, as vLLM's global pool."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        fn()
    torch.cuda.synchronize()
    return graph


def paired_graph_ms(
    graphs: dict[str, torch.cuda.CUDAGraph], iters: int = 40, inner: int = 10
) -> dict[str, float]:
    """Alternate A/B graph replays; median ms per replay."""
    for graph in graphs.values():
        for _ in range(10):
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
    print(f"[sx-qsa-mtp] {title}: {items}", flush=True)


# ---------------------------------------------------------------------------
# Verify-batch builders
# ---------------------------------------------------------------------------
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


@dataclass
class VerifyLayout:
    """Row layout of a batch: per request, rows at causal positions.

    Request r owns rows [starts[r], starts[r+1]) at positions
    seq_lens[r] - rows_r + i (the QSA metadata builder's contract: the last
    verify row is the newest token). Padded rows (graph padding) have
    token_to_req = -1 / 0 and position -1.
    """

    request_rows: list[int]
    seq_lens: list[int]
    token_to_req: list[int]
    positions: list[int]
    starts: list[int]

    @property
    def rows(self) -> int:
        return len(self.positions)


def verify_layout(
    request_rows: list[int],
    seq_lens: list[int],
    padded_rows: int = 0,
    pad_request: int = 0,
) -> VerifyLayout:
    token_to_req: list[int] = []
    positions: list[int] = []
    starts = [0]
    for request, (rows, seq_len) in enumerate(zip(request_rows, seq_lens)):
        assert 1 <= rows <= seq_len
        token_to_req += [request] * rows
        positions += list(range(seq_len - rows, seq_len))
        starts.append(starts[-1] + rows)
    token_to_req += [pad_request] * padded_rows
    positions += [-1] * padded_rows
    return VerifyLayout(list(request_rows), list(seq_lens), token_to_req, positions, starts)


def jittered_contexts(generator: torch.Generator, count: int, context: int) -> list[int]:
    return [
        max(8, context - int(torch.randint(0, 97, (1,), generator=generator)))
        for _ in range(count)
    ]


@dataclass
class IndexerCase:
    layout: VerifyLayout
    q: torch.Tensor  # [rows, 4, 128] fp16
    cache: torch.Tensor  # [pages, 196, 1, 128] fp16
    table: torch.Tensor  # [reqs, 168] int32
    token_to_req: torch.Tensor  # [rows] int32
    positions: torch.Tensor  # [rows] int64
    seq_lens: torch.Tensor  # [reqs] int32 (device, exact)
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor  # exact host copy


def build_indexer_case(layout: VerifyLayout, seed: int = 0) -> IndexerCase:
    generator = torch.Generator().manual_seed(seed)
    table, num_pages = make_block_table(
        generator, layout.seq_lens, SCHED_BLOCK, TABLE_WIDTH
    )
    device = "cuda"
    cache_gen = torch.Generator(device=device).manual_seed(seed + 17)
    cache = torch.randn(
        num_pages,
        COMPRESSED_PAGE,
        1,
        INDEX_DIM,
        dtype=torch.float16,
        device=device,
        generator=cache_gen,
    )
    q = torch.randn(
        layout.rows,
        INDEX_HEADS,
        INDEX_DIM,
        dtype=torch.float16,
        device=device,
        generator=cache_gen,
    )
    seq_lens_cpu = torch.tensor(layout.seq_lens, dtype=torch.int32)
    return IndexerCase(
        layout=layout,
        q=q,
        cache=cache,
        table=table.to(device),
        token_to_req=torch.tensor(layout.token_to_req, dtype=torch.int32, device=device),
        positions=torch.tensor(layout.positions, dtype=torch.int64, device=device),
        seq_lens=seq_lens_cpu.to(device),
        query_start_loc_cpu=torch.tensor(layout.starts, dtype=torch.int32),
        seq_lens_cpu=seq_lens_cpu,
    )


def run_select(ops, case: IndexerCase, out: torch.Tensor | None = None, **kwargs):
    return ops.qsa_select_paged_tokens(
        case.q,
        case.cache,
        case.table,
        case.token_to_req,
        case.positions,
        case.seq_lens,
        TOKEN_TOPK,
        COMPRESS_RATIO,
        out,
        **kwargs,
    )


@dataclass
class AttentionCase:
    layout: VerifyLayout
    q: torch.Tensor  # [rows, 6, 256] fp16
    gate: torch.Tensor  # [rows, 6 * 256] fp16
    kv: torch.Tensor  # [blocks, 2, page, 1, 256] fp16 (interleaved ABI)
    k: torch.Tensor
    v: torch.Tensor
    indices: torch.Tensor  # [rows, 2051] int32
    table: torch.Tensor  # [reqs, width] int32
    token_to_req: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    page_size: int


def build_attention_case(
    layout: VerifyLayout,
    indices: torch.Tensor,
    seed: int = 0,
    page_size: int = SCHED_BLOCK,
) -> AttentionCase:
    generator = torch.Generator().manual_seed(seed + 5)
    width = math.ceil(MAX_MODEL_LEN / page_size)
    table, num_blocks = make_block_table(generator, layout.seq_lens, page_size, width)
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed + 29)
    kv = torch.randn(
        num_blocks,
        2,
        page_size,
        1,
        HEAD_DIM,
        dtype=torch.float16,
        device=device,
        generator=gen,
    )
    k, v = kv.unbind(1)
    q = torch.randn(
        layout.rows, Q_HEADS, HEAD_DIM, dtype=torch.float16, device=device, generator=gen
    )
    gate = torch.randn(
        layout.rows,
        Q_HEADS * HEAD_DIM,
        dtype=torch.float16,
        device=device,
        generator=gen,
    )
    return AttentionCase(
        layout=layout,
        q=q,
        gate=gate,
        kv=kv,
        k=k,
        v=v,
        indices=indices.contiguous(),
        table=table.to(device),
        token_to_req=torch.tensor(layout.token_to_req, dtype=torch.int32, device=device),
        positions=torch.tensor(layout.positions, dtype=torch.int64, device=device),
        seq_lens=torch.tensor(layout.seq_lens, dtype=torch.int32, device=device),
        page_size=page_size,
    )


def run_attention(
    ops, case: AttentionCase, out: torch.Tensor, gated: bool = True, **kwargs
):
    return ops.qsa_sparse_paged_attention(
        case.q,
        case.k,
        case.v,
        case.indices,
        case.table,
        case.token_to_req,
        out=out,
        output_gate=case.gate if gated else None,
        query_positions=case.positions,
        sequence_lengths=case.seq_lens,
        **kwargs,
    )


def verify_case(
    num_reqs: int, q_len: int, context: int, seed: int = 0
) -> tuple[IndexerCase, AttentionCase]:
    """A uniform verify batch: num_reqs requests x q_len causal rows, with
    selections produced by the dev2 indexer (the real selection layout)."""
    generator = torch.Generator().manual_seed(seed * 7919 + num_reqs * 31 + q_len)
    seq_lens = jittered_contexts(generator, num_reqs, context)
    layout = verify_layout([q_len] * num_reqs, seq_lens)
    index_case = build_indexer_case(layout, seed=seed * 131 + num_reqs * 7 + q_len)
    indices = run_select(dev2_ops(), index_case)
    attention_case = build_attention_case(layout, indices, seed=seed + num_reqs + q_len)
    return index_case, attention_case


def reference_attention(case: AttentionCase, gated: bool = True) -> torch.Tensor:
    """FP32 torch reference over each row's selected (causal) tokens."""
    rows = case.q.shape[0]
    out = torch.zeros(rows, Q_HEADS, HEAD_DIM, dtype=torch.float32, device="cuda")
    table = case.table
    for row in range(rows):
        request = case.layout.token_to_req[row]
        if request < 0 or request >= table.shape[0]:
            continue
        tokens = case.indices[row]
        tokens = tokens[tokens >= 0].long()
        if tokens.numel() == 0:
            continue
        pages = table[request, tokens // case.page_size].long()
        valid = pages >= 0
        tokens, pages = tokens[valid], pages[valid]
        offsets = tokens % case.page_size
        keys = case.k[pages, offsets, 0].float()  # [n, 256]
        values = case.v[pages, offsets, 0].float()
        query = case.q[row].float()  # [6, 256]
        scores = query @ keys.t() * HEAD_DIM**-0.5
        probs = torch.softmax(scores, dim=-1)
        out[row] = probs @ values
    out = out.to(torch.float16).float()
    if gated:
        out = out * torch.sigmoid(case.gate.view(rows, Q_HEADS, HEAD_DIM).float())
    return out
