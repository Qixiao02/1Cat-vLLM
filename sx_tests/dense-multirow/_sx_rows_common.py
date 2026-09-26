# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the dense-multirow (design_4 MR1/MR2/MR3/MR9a) tests.

Not a test module.  Imported by the sibling test_*.py files (pytest puts the
test directory on sys.path; script mode runs from it).  Requires the overlay
files to be installed into the vllm package that ``import vllm`` resolves
(e.g. /opt/venv/lib/python3.12/site-packages/vllm in the 1.7.2 image).
"""

from __future__ import annotations

import contextlib
import math
import os
import statistics
from collections.abc import Callable, Iterator

import torch

# (name, prefix, (N, K), calls per decode step, benchmark?)
GEMV_ROLES: list[tuple[str, str, tuple[int, int], int, bool]] = [
    ("router", "model.layers.3.mlp.gate", (512, 2560), 48, True),
    ("qsa_qkv", "model.layers.3.self_attn.qkv_proj", (3584, 2560), 12, True),
    ("qsa_o", "model.layers.3.self_attn.o_proj", (2560, 1536), 12, True),
    (
        "qsa_index",
        "model.layers.3.self_attn.indexer.index_qk_proj",
        (640, 2560),
        12,
        True,
    ),
    ("gdn_out", "model.layers.0.linear_attn.out_proj", (2560, 1536), 36, True),
    # Shadowed in production by the fused GDN input / fused HC routes; kept
    # for exactness coverage of every _ROLE_PLANS entry.
    ("gdn_qkvz", "model.layers.0.linear_attn.in_proj_qkvz", (4096, 2560), 36, False),
    ("gdn_ba", "model.layers.0.linear_attn.in_proj_ba", (24, 2560), 36, False),
    (
        "hc_down_gemv",
        "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject",
        (336, 10240),
        96,
        False,
    ),
]

FULL_M = (1, 2, 3, 4, 8, 16, 17, 24, 32)
QUICK_M = (1, 2, 4, 8, 16, 24)
BENCH_M = (2, 4, 8, 16, 24)
SCALES = (0.25, 1.0, 3.0)


def quick() -> bool:
    return os.environ.get("SX_TEST_QUICK", "0") not in ("", "0")


def m_values() -> tuple[int, ...]:
    return QUICK_M if quick() else FULL_M


def sm70_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def gemv_module():
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv

    return sm70_fp16_gemv


def hc_module():
    # Registers qwen4_exp_hc_silu / gate_mix / combine_norm custom ops too.
    import vllm.models.qwen4_exp.nvidia.ops.hc  # noqa: F401
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc

    return sm70_fp16_hc


@contextlib.contextmanager
def sx_env(**overrides: str | None) -> Iterator[None]:
    """Temporarily set SX_* / VLLM_* env vars and re-read the SX config."""
    gemv = gemv_module()
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        gemv._sx_rows_config.cache_clear()
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        gemv._sx_rows_config.cache_clear()


# The per-role table these tests were written against (the implementation's
# first defaults). The shipped default table is narrower after V100 validation
# (see _SX_ROWS_DEFAULT_MAX_M); mechanism/admission tests pin this table so they
# do not depend on the tuned defaults.
IMPL_TABLE = "gdn_in=8,gdn_out=8,qsa_qkv=8,qsa_o=8,qsa_index=8,router=24,hc=8"


@contextlib.contextmanager
def decode_graph_ctx(**overrides: str | None) -> Iterator[None]:
    """Emulate the FULL decode-graph capture context of the dual-compile lane."""
    from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation

    overrides.setdefault("SX_OPT_ROWS_TABLE", IMPL_TABLE)

    with (
        sx_env(VLLM_SM70_QWEN38_DUAL_COMPILE="1", **overrides),
        sm70_decode_graph_compilation(True),
    ):
        yield


def _special_row(kind: int, k: int, gen: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    if kind == 0:
        return torch.zeros(k, device=dev)
    if kind == 1:
        return torch.full((k,), -0.0, device=dev)
    sign = torch.where(
        torch.rand(k, generator=gen, device=dev) < 0.5,
        -1.0,
        1.0,
    )
    if kind == 2:
        # FP16 subnormals (multiples of 2^-24) plus signed zeros.
        mant = torch.randint(0, 1024, (k,), generator=gen, device=dev).float()
        return sign * mant * 2.0**-24
    if kind == 3:
        row = torch.randn(k, generator=gen, device=dev)
        row[::97] = 65504.0 * sign[::97]
        row[1::193] = -0.0
        return row
    # Tiny alternating values.
    return sign * 1e-3


def make_rows(
    m: int, k: int, scale: float, seed: int, special: str = "mixed"
) -> torch.Tensor:
    """(m, k) FP16 activations.

    special="mixed": randn*scale rows, every fifth row a special row;
    "all": every row special (zeros, -0, subnormals, +-65504, tiny);
    "none": randn*scale only.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device="cuda") * scale
    for r in range(m):
        if special == "all":
            x[r] = _special_row(r % 5, k, gen)
        elif special == "mixed" and r % 5 == 4:
            x[r] = _special_row((r // 5) % 5, k, gen)
    return x.to(torch.float16).contiguous()


def input_cases(m: int, k: int, seed: int) -> list[tuple[str, torch.Tensor]]:
    cases = [
        (f"randn*{scale}", make_rows(m, k, scale, seed + i))
        for i, scale in enumerate(SCALES)
    ]
    cases.append(("special", make_rows(m, k, 1.0, seed + 97, special="all")))
    return cases


def poison_rows(x: torch.Tensor, rows) -> None:
    """Overwrite ``rows`` of x with NaN/+Inf/-Inf garbage in place.

    Padded FULL-decode-graph rows (e.g. 17..23 real rows padded to 24) carry
    stale data from earlier steps; it must never leak into the real rows.
    """
    k = x.shape[1]
    garbage = torch.full((k,), float("nan"), device=x.device)
    garbage[1::3] = float("inf")
    garbage[2::3] = float("-inf")
    for r in rows:
        x[r].copy_(garbage.to(x.dtype))


class Mismatches:
    """Collect bitwise mismatches per variant (e.g. reduce form) and fail once.

    A failure of SX_OPT_ROWS_FUSED_REDUCE=1 must not hide whether the plain
    per-row form (=0) passes: the report names the clean forms so the
    validation run can pick the right switch without a second pass.
    """

    def __init__(self) -> None:
        self.failed: dict[str, list[str]] = {}
        self.checked: dict[str, int] = {}

    def check(self, key: str, ok: bool, context: str) -> None:
        self.checked[key] = self.checked.get(key, 0) + 1
        if not ok:
            self.failed.setdefault(key, []).append(context)

    def summary(self) -> str:
        lines = []
        for key in sorted(self.checked):
            bad = self.failed.get(key, [])
            line = f"  {key}: {len(bad)} failed / {self.checked[key]} checked"
            if bad:
                line += f"; first: {bad[0]}"
            lines.append(line)
        return "\n".join(lines)

    def assert_clean(self, title: str) -> None:
        print(f"\n== {title}: per-form bitwise results\n{self.summary()}")
        if self.failed:
            clean = [key for key in sorted(self.checked) if key not in self.failed]
            raise AssertionError(
                f"{title}: bitwise mismatches\n{self.summary()}\n"
                f"  clean forms: {', '.join(clean) or 'none'}"
            )


def make_weight(n: int, k: int, scale: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((n, k), generator=gen, device="cuda") * scale
    w = w.to(torch.float16)
    flat = w.view(-1)
    flat[::1009] = 0.0
    flat[5::2003] = -0.0
    flat[7::3001] = 2.0**-24
    return w.contiguous()


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def max_abs_finite(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float(), b.float()
    finite = torch.isfinite(af) & torch.isfinite(bf)
    if not bool(finite.any()):
        return 0.0
    return float((af - bf).abs()[finite].max())


def n_copies(bytes_per_copy: int, target_bytes: int = 48 << 20) -> int:
    """Distinct weight copies so the rotating working set defeats the 6 MB L2."""
    return max(2, min(32, math.ceil(target_bytes / bytes_per_copy)))


def bench_graphs(
    variants: dict[str, Callable[[int], object]],
    calls: int,
    rounds: int = 8,
    per_round: int = 8,
    warmup: int = 5,
) -> dict[str, float]:
    """Median microseconds per call of each variant.

    Each variant is captured once as a CUDA graph of ``calls`` launches
    (call i uses weight copy i % copies).  Samples alternate between variants
    round by round (A/B/A...), ``rounds * per_round`` (>= 50) replays each.
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


def admitted_max_m(
    per_m: dict[int, tuple[float, float]], margin: float = 0.97
) -> int:
    """Largest M such that rows <= margin * cuBLAS at every measured M <= it."""
    best = 1
    for m in sorted(per_m):
        baseline, candidate = per_m[m]
        if candidate <= margin * baseline:
            best = m
        else:
            break
    return best


def print_table(title: str, header: list[str], rows: list[list[object]]) -> None:
    widths = [len(h) for h in header]
    text_rows = [[str(c) for c in row] for row in rows]
    for row in text_rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    print(f"\n== {title}")
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for row in text_rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
