# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the batch-3 "draft" tests and the tile sweep.

Not a test module. The sibling files import it first (pytest puts this
directory on sys.path; script mode runs from it). Requires the changed .py
files (and the new configs/*.json) installed into, or bind-mounted over, the
vllm package that ``import vllm`` resolves, e.g.
/opt/venv/lib/python3.12/site-packages/vllm in image 1.8.0-dev2.

Geometry: Qwen3.8-Flash-Next MTP draft experts, TP4-local:
E=512 routed experts, hidden K=2560, I_local=160 (w13 [512, 320, 2560],
w2 [512, 2560, 160]), top-10, FP16.
"""

from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import statistics
import sys
from collections.abc import Callable, Iterator
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

E = 512
HIDDEN = 2560
I_GLOBAL = 640
TP = 4
I_LOCAL = I_GLOBAL // TP  # 160
TOPK = 10
SHAPE = (E, I_LOCAL, HIDDEN, TOPK)

# Every MTP draft width for k in {1,2,3,4} and B in {1,2,4,8,12,16,24}:
# draft decode M = B, draft prefill (step 0) M = B*(k+1).
DRAFT_BATCHES = (1, 2, 4, 8, 12, 16, 24)
DRAFT_KS = (1, 2, 3, 4)


def draft_widths() -> tuple[int, ...]:
    widths = set(DRAFT_BATCHES)
    for k in DRAFT_KS:
        widths.update(b * (k + 1) for b in DRAFT_BATCHES)
    return tuple(sorted(widths))


# 1,2,3,4,5,6,8,10,12,16,20,24,32,36,40,48,60,64,72,80,96,120
FULL_WIDTHS = draft_widths()
QUICK_WIDTHS = (1, 2, 5, 8, 10, 16, 20, 40, 80, 120)
# Split-graph draft decode sizes and table boundaries: 12/13 is the naive ->
# sorted expert-assignment switch (M*topk*4 <= E), 32/33 and 128/129 the
# default table edges (129+ keeps the 0.0.3 tile), 136 = an aligned
# (M % 8 == 0) draft prefill above the table, 160 = k=7 at B=20.
EXTRA_WIDTHS = (7, 9, 13, 15, 18, 30, 33, 45, 128, 129, 136, 160)


def quick() -> bool:
    return os.environ.get("SX_TEST_QUICK", "0") not in ("", "0")


def no_bench() -> bool:
    return os.environ.get("SX_TEST_NO_BENCH", "0") not in ("", "0")


def widths() -> tuple[int, ...]:
    if quick():
        return QUICK_WIDTHS
    return tuple(sorted(set(FULL_WIDTHS) | set(EXTRA_WIDTHS)))


def sm70_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def fm():
    from vllm.model_executor.layers.fused_moe import fused_moe

    return fused_moe


# ---------------------------------------------------------------------------
# Config selection contexts
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def tiles_env(armed: bool | None = None, **env: str | None) -> Iterator[None]:
    """Set SX_OPT_MTP_DRAFT_TILES* env vars / arming; restore afterwards."""
    mod = fm()
    saved_env = {key: os.environ.get(key) for key in env}
    saved_armed = mod._sx_mtp_draft_tiles_armed
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        mod.sx_mtp_draft_tiles_cache_clear()
        if armed is not None:
            mod._sx_mtp_draft_tiles_armed = bool(armed)
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        mod._sx_mtp_draft_tiles_armed = saved_armed
        mod.sx_mtp_draft_tiles_cache_clear()


def previous_default_ctx():
    """Image 1.8.0-dev2 behaviour: 1Cat tile at M1/M5, 0.0.3 tile elsewhere."""
    return tiles_env(armed=False)


def table_ctx(**env: str | None):
    """The SX draft tile table as the armed drafter selects it."""
    env.setdefault("SX_OPT_MTP_DRAFT_TILES", "1")
    return tiles_env(armed=True, **env)


def legacy_ctx():
    """The 0.0.3 tile BM16/BN32/BK64 for every width."""
    return fm().force_sm70_mtp_moe_legacy_config()


def tile_ctx(config: dict[str, int]):
    """Force one explicit Triton config (vLLM override_config)."""
    from vllm.model_executor.layers.fused_moe import override_config

    return override_config(dict(config))


def selected_config(m: int) -> dict[str, int]:
    """The config fused_experts would select at width m (current context)."""
    return fm().try_get_optimal_moe_config(
        (E, 2 * I_LOCAL, HIDDEN), (E, HIDDEN, I_LOCAL), TOPK, None, m
    )


def tile_str(cfg: dict[str, int]) -> str:
    return (
        f"BM{cfg.get('BLOCK_SIZE_M')}/BN{cfg.get('BLOCK_SIZE_N')}"
        f"/BK{cfg.get('BLOCK_SIZE_K')}/G{cfg.get('GROUP_SIZE_M', 1)}"
        f"/w{cfg.get('num_warps', '-')}/s{cfg.get('num_stages', '-')}"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_experts(
    seed: int, scale: float = 0.02, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random TP4-local draft experts (w13 [E, 320, 2560], w2 [E, 2560, 160])."""
    gen = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.empty((E, 2 * I_LOCAL, HIDDEN), dtype=torch.float16, device=device)
    w2 = torch.empty((E, HIDDEN, I_LOCAL), dtype=torch.float16, device=device)
    for e in range(0, E, 64):
        w1[e : e + 64].copy_(
            torch.randn(
                (min(64, E - e), 2 * I_LOCAL, HIDDEN), generator=gen, device=device
            )
            * scale
        )
        w2[e : e + 64].copy_(
            torch.randn((min(64, E - e), HIDDEN, I_LOCAL), generator=gen, device=device)
            * scale
        )
    return w1, w2


def _find_key(keys, suffix: str) -> str | None:
    hits = [k for k in keys if "mtp" in k and k.endswith(suffix)]
    return sorted(hits, key=len)[0] if hits else None


def load_real_experts(
    model_dir: str, tp_rank: int = 0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the checkpoint's MTP routed experts and slice TP rank ``tp_rank``.

    Accepts the fused layout (mtp.layers.0.mlp.experts.gate_up_proj
    [E, 2*I, H] + down_proj [E, H, I]) or per-expert gate/up/down weights.
    BF16 is converted to FP16 like the model does (dtype=float16).
    """
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    weight_map[key] = os.path.basename(path)

    def open_key(key: str):
        return safe_open(
            os.path.join(model_dir, weight_map[key]), framework="pt", device="cpu"
        )

    r0, r1 = tp_rank * I_LOCAL, (tp_rank + 1) * I_LOCAL
    gate_up = _find_key(weight_map, "mlp.experts.gate_up_proj")
    down = _find_key(weight_map, "mlp.experts.down_proj")
    if gate_up is not None and down is not None:
        with open_key(gate_up) as f:
            sl = f.get_slice(gate_up)
            n_experts, two_i, hidden = sl.get_shape()
            assert (n_experts, two_i, hidden) == (E, 2 * I_GLOBAL, HIDDEN), (
                gate_up,
                sl.get_shape(),
            )
            gate = sl[:, r0:r1, :]
            up = sl[:, I_GLOBAL + r0 : I_GLOBAL + r1, :]
        w1 = torch.cat([gate, up], dim=1).to(torch.float16)
        with open_key(down) as f:
            sl = f.get_slice(down)
            assert tuple(sl.get_shape()) == (E, HIDDEN, I_GLOBAL), sl.get_shape()
            w2 = sl[:, :, r0:r1].to(torch.float16)
        return w1.contiguous().to(device), w2.contiguous().to(device)

    w1 = torch.empty((E, 2 * I_LOCAL, HIDDEN), dtype=torch.float16)
    w2 = torch.empty((E, HIDDEN, I_LOCAL), dtype=torch.float16)
    for e in range(E):
        gk = _find_key(weight_map, f"mlp.experts.{e}.gate_proj.weight")
        uk = _find_key(weight_map, f"mlp.experts.{e}.up_proj.weight")
        dk = _find_key(weight_map, f"mlp.experts.{e}.down_proj.weight")
        if gk is None or uk is None or dk is None:
            raise FileNotFoundError(f"no MTP expert {e} weights under {model_dir}")
        with open_key(gk) as f:
            w1[e, :I_LOCAL] = f.get_slice(gk)[r0:r1, :].to(torch.float16)
        with open_key(uk) as f:
            w1[e, I_LOCAL:] = f.get_slice(uk)[r0:r1, :].to(torch.float16)
        with open_key(dk) as f:
            w2[e] = f.get_slice(dk)[:, r0:r1].to(torch.float16)
    return w1.to(device), w2.to(device)


def experts_for_test(seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    model_dir = os.environ.get("SX_TEST_MTP_WEIGHTS_DIR", "").strip()
    if model_dir:
        rank = int(os.environ.get("SX_TEST_MTP_TP_RANK", "0"))
        return load_real_experts(model_dir, rank)
    return make_experts(seed)


def make_routing(
    m: int,
    seed: int,
    overlap: float = 0.0,
    group: int = 1,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-10 routing: (topk_weights fp32 [m, 10], topk_ids int32 [m, 10]).

    Uniform distinct experts per row. With ``overlap`` > 0, each row after
    the first of a ``group`` (one request's k+1 draft-prefill rows) keeps each
    expert of the previous row with that probability, emulating correlated
    routing of consecutive tokens.
    """
    gen = torch.Generator().manual_seed(seed)
    ids = torch.empty((m, TOPK), dtype=torch.int64)
    for r in range(m):
        if overlap > 0.0 and group > 1 and r % group:
            prev = ids[r - 1]
            keep = prev[torch.rand(TOPK, generator=gen) < overlap]
            mask = torch.ones(E, dtype=torch.bool)
            mask[keep] = False
            pool = mask.nonzero().flatten()
            fill = pool[torch.randperm(pool.numel(), generator=gen)[: TOPK - keep.numel()]]
            ids[r] = torch.cat([keep, fill])
        else:
            ids[r] = torch.randperm(E, generator=gen)[:TOPK]
    weights = torch.softmax(torch.randn((m, TOPK), generator=gen), dim=-1)
    return weights.to(device=device, dtype=torch.float32), ids.to(
        device=device, dtype=torch.int32
    )


def make_hidden(m: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((m, HIDDEN), generator=gen, device="cuda") * scale).to(
        torch.float16
    )


def run_moe(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """The unquantized Triton fused MoE (same config selection, assignment and
    kernel as the drafter's TritonExperts)."""
    return fm().fused_experts(hidden, w1, w2, topk_weights, topk_ids)


def reference_fp32(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """FP32 oracle: sum_j w_j * W2[e_j] (silu(G[e_j] x) * U[e_j] x)."""
    x = hidden.float()
    out = torch.zeros((hidden.shape[0], HIDDEN), dtype=torch.float32, device=x.device)
    ids = topk_ids.long()
    for e in torch.unique(ids).tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        xe = x[rows]
        gu = xe @ w1[e].float().t()
        act = torch.nn.functional.silu(gu[:, :I_LOCAL]) * gu[:, I_LOCAL:]
        ye = act @ w2[e].float().t()
        out.index_add_(0, rows, ye * topk_weights[rows, slots].float()[:, None])
    return out


# ---------------------------------------------------------------------------
# Bits and graphs
# ---------------------------------------------------------------------------
def bits16(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def bit_equal16(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits16(a), bits16(b))


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def capture(fn: Callable[[], object], warmup: int = 2) -> torch.cuda.CUDAGraph:
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


def time_graph(
    graph: torch.cuda.CUDAGraph, calls: int, replays: int = 20, warmup: int = 3
) -> list[float]:
    """Microseconds per call for each replay of a graph holding ``calls`` calls."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / calls)
    return samples


class MoeBench:
    """CUDA-graph timing of one width with rotating routings / weight copies.

    ``routings`` distinct (hidden, topk) sets are chained in one graph, each
    reading weight copy ``i % len(experts)``, so consecutive calls touch
    different experts and the >1 GB weights are cold (V100 L2 is 6 MB).
    """

    def __init__(
        self,
        m: int,
        experts: list[tuple[torch.Tensor, torch.Tensor]],
        routings: int = 8,
        seed: int = 0,
        overlap: float = 0.0,
        group: int = 1,
    ) -> None:
        self.m = m
        self.experts = experts
        self.inputs = []
        for i in range(routings):
            weights, ids = make_routing(m, seed + 7919 * i, overlap, group)
            self.inputs.append((make_hidden(m, seed + 31 * i), weights, ids))

    def fn(self) -> None:
        for i, (hidden, weights, ids) in enumerate(self.inputs):
            w1, w2 = self.experts[i % len(self.experts)]
            run_moe(hidden, w1, w2, weights, ids)

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def graph(self, ctx_factory: Callable[[], contextlib.AbstractContextManager]):
        """Capture under the config-selection context (selection is baked)."""
        with ctx_factory():
            return capture(self.fn)


def bench_alternating(
    graphs: dict[str, torch.cuda.CUDAGraph],
    calls: int,
    rounds: int = 6,
    per_round: int = 6,
) -> dict[str, float]:
    """Median us/call; variants alternate A/B/A... round by round."""
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    names = list(graphs)
    for graph in graphs.values():
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    for rnd in range(rounds):
        for name in names if rnd % 2 == 0 else names[::-1]:
            samples[name].extend(time_graph(graphs[name], calls, per_round, 0))
    return {name: statistics.median(values) for name, values in samples.items()}


def print_table(title: str, header: list[str], rows: list[list[object]]) -> None:
    widths_ = [len(h) for h in header]
    text_rows = [[str(c) for c in row] for row in rows]
    for row in text_rows:
        widths_ = [max(w, len(c)) for w, c in zip(widths_, row)]
    print(f"\n== {title}")
    print("  ".join(h.rjust(w) for h, w in zip(header, widths_)))
    for row in text_rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths_)))


def expected_distinct_experts(m: int) -> float:
    """Uniform-routing expectation D(10M) = E (1 - (1 - 1/E)^(10M))."""
    return E * (1.0 - math.pow(1.0 - 1.0 / E, TOPK * m))
