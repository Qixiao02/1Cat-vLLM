# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for sx_tests/b2-moe-m32 (SX_OPT_MOE_GROUPED32 / _MASK).

The v2 grouped-decode ops live in csrc/sm70_turbomind/ops/
nvfp4_grouped_decode_sm70.cu. Inside the deployed image the old ops are in
vllm._C; the v2 ops come from either

  * a rebuilt vllm._C (namespace ``_C``), or
  * the sidecar (namespace ``_C_qwen38_grouped``): this .cu compiled alone with
    -DSX_NVFP4_GROUPED_SIDECAR=1 for sm_70, loaded from
    $SX_OPT_MOE_GROUPED32_LIBRARY or vllm/_sx_nvfp4_grouped32_C*.so.

``ensure_v2_ops()`` loads $SX_OPT_MOE_GROUPED32_LIBRARY if set, otherwise
builds the sidecar into $SX_B2_BUILD_DIR (default /tmp/sx_b2_moe_m32_build)
with build_sidecar.py (needs nvcc; CUDA_HOME=/usr/local/cuda in the image).
"""

from __future__ import annotations

import importlib.util
import os
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CU_SOURCE = REPO / "csrc" / "sm70_turbomind" / "ops" / "nvfp4_grouped_decode_sm70.cu"
NAMESPACE = "_C_qwen38_grouped"
W13_V2 = "nvfp4_grouped_w13_v2_sm70_out"
W2_V2 = "nvfp4_grouped_w2_v2_sm70_out"
MAX_ROUTES = "nvfp4_grouped_decode_v2_max_routes"
E, H, I, K = 512, 2560, 160, 10
LAYERS = 48

ON_SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _v2_namespace() -> str | None:
    for name in (NAMESPACE, "_C"):
        ns = getattr(torch.ops, name)
        if all(hasattr(ns, op) for op in (W13_V2, W2_V2, MAX_ROUTES)):
            return name
    return None


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "sx_b2_build_sidecar", HERE / "build_sidecar.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_v2_ops() -> str:
    """Load or build the v2 ops; return their torch.ops namespace name."""
    import vllm._C  # noqa: F401  (old grouped ops, _moe_C etc.)

    name = _v2_namespace()
    if name is not None:
        return name
    library = os.environ.get("SX_OPT_MOE_GROUPED32_LIBRARY")
    if not library or not os.path.exists(library):
        library = _load_build_module().build()
        # Spawned processes and nvfp4_sm70_moe._load_grouped_v2_ops reuse it.
        os.environ["SX_OPT_MOE_GROUPED32_LIBRARY"] = library
    if _v2_namespace() is None:
        torch.ops.load_library(library)
    name = _v2_namespace()
    assert name is not None, f"v2 ops missing after loading {library}"
    return name


def v2_ops():
    ns = getattr(torch.ops, ensure_v2_ops())
    return getattr(ns, W13_V2), getattr(ns, W2_V2), getattr(ns, MAX_ROUTES)


def legacy_ops():
    import vllm._C  # noqa: F401

    ns = torch.ops._C
    if not hasattr(ns, "nvfp4_grouped_w13_sm70_out"):
        return None
    return ns.nvfp4_grouped_w13_sm70_out, ns.nvfp4_grouped_w2_sm70_out


def synthetic_weights(seed: int = 123):
    """Same distribution as tests/kernels/quantization/test_sm70_moe_packed_w13.py."""
    torch.manual_seed(seed)
    return (
        torch.randint(0, 2**31 - 1, (E, H, 40), device="cuda", dtype=torch.int32),
        torch.rand(E, 160, 320, device="cuda", dtype=torch.float16) * 0.01,
        torch.randint(0, 2**31 - 1, (E, 160, 320), device="cuda", dtype=torch.int32),
        torch.rand(E, 10, H, device="cuda", dtype=torch.float16) * 0.01,
    )


KINDS = (
    "random",  # independent top-10 per row (distinct within a row)
    "clustered",  # 40-expert pool: heavy cross-row reuse
    "pool113",  # ~C24 production pool (design_1: ~101-107 unique at M24/32)
    "shared10",  # every row uses the same 10 experts (4 packs per expert @M32)
    "reversed",  # descending ids inside each row
    "invalid",  # -1 / 512 / huge ids mixed in (invalid bucket)
    "same",  # every route to one expert (up to 40 packs of one expert)
)


def routes(kind: str, m: int, seed: int, device: str = "cuda"):
    """(x [m,H] fp16, topk weights [m,K] fp32, ids [m,K] int32), contiguous."""
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn(m, E, generator=gen, device=device)
    if kind in ("clustered", "pool113"):
        pool = torch.randperm(E, generator=gen, device=device)[
            : 40 if kind == "clustered" else 113
        ]
        mask = torch.full((E,), -1e4, device=device)
        mask[pool] = 0.0
        scores = scores + mask
    vals, ids = torch.topk(scores, K, dim=-1)
    ids = ids.to(torch.int32)
    if kind == "shared10":
        base = torch.tensor(
            [3, 77, 150, 151, 222, 300, 301, 402, 480, 511],
            device=device,
            dtype=torch.int32,
        )
        ids = torch.stack([base.roll(r) for r in range(m)])
    elif kind == "reversed":
        ids, _ = torch.sort(ids, dim=-1, descending=True)
    elif kind == "invalid":
        flat = ids.view(-1)
        n = flat.numel()
        flat[torch.arange(0, n, 7, device=device)] = -1
        flat[torch.arange(3, n, 11, device=device)] = E
        flat[torch.arange(5, n, 13, device=device)] = 99999
    elif kind == "same":
        ids = torch.full((m, K), 42, device=device, dtype=torch.int32)
    weights = torch.softmax(vals.float(), dim=-1).contiguous()
    x = (torch.randn(m, H, generator=gen, device=device) * 0.1).half().contiguous()
    return x, weights, ids.contiguous()


def expected_plan(ids: list[int], limit: int) -> list[tuple[int, list[int]]]:
    """CPU model of plan_v2_kernel: (expert, routes) per group, in group order."""
    buckets: dict[int, list[int]] = {}
    for route in range(limit):
        expert = ids[route]
        if expert < 0 or expert >= E:
            expert = E
        buckets.setdefault(expert, []).append(route)
    order = sorted(b for b in buckets if b < E) + ([E] if E in buckets else [])
    groups = []
    for expert in order:
        members = buckets[expert]
        for start in range(0, len(members), 8):
            groups.append((expert, members[start : start + 8]))
    return groups


def check_plan(ids_flat: torch.Tensor, limit: int, rows, experts, sizes, total):
    groups = expected_plan(ids_flat.tolist(), limit)
    assert int(total.item()) == len(groups), (int(total.item()), len(groups))
    rr, ee, ss = rows.cpu(), experts.cpu(), sizes.cpu()
    for g, (expert, members) in enumerate(groups):
        assert int(ee[g]) == expert, (g, int(ee[g]), expert)
        assert int(ss[g]) == len(members), (g, int(ss[g]), len(members))
        assert rr.view(-1, 8)[g, : len(members)].tolist() == members, g
    return len(groups)


class Workspace:
    """Buffers for one grouped decode call at width m (capacity 32 rows)."""

    def __init__(self, m: int, capacity_routes: int = 320):
        n = m * K
        self.m = m
        self.inter = torch.empty(n, 160, device="cuda", dtype=torch.float16)
        self.routed = torch.empty(n, H, device="cuda", dtype=torch.float16)
        self.out = torch.empty(m, H, device="cuda", dtype=torch.float16)
        self.rows = torch.empty(capacity_routes, 8, device="cuda", dtype=torch.int32)
        self.experts = torch.empty(capacity_routes, device="cuda", dtype=torch.int32)
        self.sizes = torch.empty(capacity_routes, device="cuda", dtype=torch.int32)
        self.total = torch.empty(1, device="cuda", dtype=torch.int32)

    def poison(self):
        self.inter.fill_(float("nan"))
        self.routed.fill_(float("nan"))
        self.out.fill_(float("nan"))
        for t in (self.rows, self.experts, self.sizes, self.total):
            t.fill_(-9999)


def run_v2(ws: Workspace, weights, x, topk, ids, split=8, interleaved=True,
           valid=None):
    w13, w2, _ = v2_ops()
    w, s, w2w, s2 = weights
    w13(ws.inter, x, w, s, ids.view(-1), ws.rows, ws.experts, ws.sizes,
        ws.total, split, interleaved, valid)
    w2(ws.out, ws.routed, ws.inter, w2w, s2, topk, ws.rows, ws.experts,
       ws.sizes, ws.total, valid)


def run_legacy(ws: Workspace, weights, x, topk, ids, split=8, interleaved=True):
    w13, w2 = legacy_ops()
    w, s, w2w, s2 = weights
    w13(ws.inter, x, w, s, ids.view(-1), ws.rows, ws.experts, ws.sizes,
        ws.total, split, interleaved)
    w2(ws.out, ws.routed, ws.inter, w2w, s2, topk, ws.rows, ws.experts,
       ws.sizes, ws.total)


def graph_ms(fn, samples: int = 60, warmup: int = 5) -> float:
    """Median CUDA-event time of one replay of a graph that runs fn()."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return statistics.median(times)
