# SPDX-License-Identifier: Apache-2.0
"""design_4 [MR5]: decode-specialised QSA lexicographic top-k for every row.

GPU: ONE V100 (SM70). Needs the rebuilt _C_stable_libtorch.abi3.so (new op
_C::qsa_lexicographic_topk_decode_rows) and the patched
vllm/models/qwen4_exp/nvidia/ops/qsa.py installed into the imported vllm
(or VLLM_SM70_QSA_TOPK_LIBRARY pointing at a build of the patched
benchmarks/kernels/sm70_qsa_topk_sidecar.cu).

  /opt/venv/bin/python -m pytest -q -s sx_tests/b2-small-native/test_qsa_topk_rows.py
  (SX_TEST_QUICK=1 shrinks the M ladder and replay counts; SX_TEST_NO_BENCH=1
  skips the microbenchmark.)

Asserts (torch.equal on the int32 ids, i.e. identical membership AND order):
  * rows kernel (grid = M) == generic kernel (the old M > 1 launch) == the
    UNCHANGED M == 1 launch (decode kernel) run on each row alone == a torch
    reference (score key descending, index ascending, emitted in increasing
    index order), for M in {1,2,3,4,8,16,17,20,24,32}, the deployment decode
    width (32928 score columns), per-row mixed lengths
    {0, 1, 511, 512, 513, 600, 777, 1500, 2048, 2304, 2305, 4096, 8192,
    32928, >columns (clamped), -1} and score kinds (randn, heavy ties, all
    +-0, FP16-rounded, +-Inf/+-NaN/+-0 specials, >FP16 ramp, one coarse
    bucket), contiguous and padded row stride;
  * realistic decode lengths (2K/8K/32K/128K contexts);
  * CUDA-graph replay with changing scores and lengths and a poisoned output
    == eager generic (128 replays per M; 32 with SX_TEST_QUICK);
  * qsa_select_paged_tokens (paged Triton scorer -> top-k -> expand) with
    SX_OPT_QSA_TOPK_ROWS on == off bit for bit, the rows op engaged exactly
    for 2 <= M <= 32 (not for M == 1 / M == 33), eager and graph-captured,
    including padded (position -1) rows;
Reports a CUDA-graph microbenchmark (12 calls = the 12 QSA layers of one
decode step, distinct score buffers per call) of the old launch vs the rows
kernel at M in {1,2,4,8,16,24,32} for 2K/8K/32K/mixed contexts.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b2_common as C  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.sm70_available(), reason="requires SM70 GPU")

LENGTH_PATTERN = (
    0,
    1,
    511,
    512,
    513,
    600,
    777,
    1500,
    2048,
    2304,
    2305,
    4096,
    8192,
    C.SCORE_COLUMNS,
    C.SCORE_COLUMNS + 5000,
    -1,
)
KINDS = (
    "randn",
    "ties17",
    "zeros_signed",
    "fp16_randn",
    "special",
    "ramp",
    "narrow",
)
NEG_NAN_BITS = -4194304  # 0xffc00000 as int32


def qsa():
    from vllm.models.qwen4_exp.nvidia.ops import qsa as module

    return module


def rows_op():
    op = qsa()._sm70_qsa_lexicographic_topk_rows_op()
    if op is None:
        pytest.fail(
            "qsa_lexicographic_topk_decode_rows is not available: mount the "
            "rebuilt _C_stable_libtorch.abi3.so (or a patched QSA sidecar)"
        )
    return op


def old_op():
    return qsa()._sm70_qsa_lexicographic_topk_op()


def run_rows(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = torch.full((logits.shape[0], C.TOPK), -7, dtype=torch.int32, device="cuda")
    rows_op()(logits, lengths, out, C.TOPK)
    return out


def run_generic(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The generic kernel (the old launch for num_rows >= 2)."""
    m = logits.shape[0]
    if m >= 2:
        out = torch.full((m, C.TOPK), -7, dtype=torch.int32, device="cuda")
        old_op()(logits, lengths, out, C.TOPK)
        return out
    # num_rows == 1 would take the decode kernel: add a dummy second row so
    # the old launcher picks the generic kernel, and keep row 0.
    padded = torch.zeros((2, logits.shape[1]), dtype=logits.dtype, device="cuda")
    padded[0].copy_(logits[0])
    padded_lengths = torch.zeros(2, dtype=torch.int32, device="cuda")
    padded_lengths[0] = lengths[0]
    out = torch.full((2, C.TOPK), -7, dtype=torch.int32, device="cuda")
    old_op()(padded, padded_lengths, out, C.TOPK)
    return out[:1].clone()


def run_m1_per_row(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The unchanged M == 1 launch (decode kernel) on each row alone."""
    m = logits.shape[0]
    out = torch.full((m, C.TOPK), -7, dtype=torch.int32, device="cuda")
    for r in range(m):
        row_out = torch.full((1, C.TOPK), -7, dtype=torch.int32, device="cuda")
        old_op()(logits[r : r + 1], lengths[r : r + 1], row_out, C.TOPK)
        out[r].copy_(row_out[0])
    return out


def ordered_keys(values: torch.Tensor) -> torch.Tensor:
    """ordered_float_bits() of the CUDA selector as int64 (0 .. 2^32-1)."""
    v = values.clone()
    v[v == 0] = 0.0  # -0.0 -> +0.0 (NaN compares false and is kept)
    raw = v.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    negative = raw >= 0x80000000
    return torch.where(negative, (~raw) & 0xFFFFFFFF, raw | 0x80000000)


def reference(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    m, columns = logits.shape
    out = torch.full((m, C.TOPK), -1, dtype=torch.int32, device="cuda")
    for r, raw in enumerate(lengths.tolist()):
        length = min(max(raw, 0), columns)
        if length <= C.TOPK:
            out[r, :length] = torch.arange(length, dtype=torch.int32, device="cuda")
            continue
        keys = ordered_keys(logits[r, :length])
        order = torch.sort(keys, descending=True, stable=True).indices[: C.TOPK]
        out[r] = order.sort().values.to(torch.int32)
    return out


def make_row(kind: str, columns: int, gen: torch.Generator) -> torch.Tensor:
    dev = "cuda"
    if kind == "randn":
        return torch.randn(columns, generator=gen, device=dev)
    if kind == "ties17":
        return torch.randint(0, 17, (columns,), generator=gen, device=dev).float()
    if kind == "zeros_signed":
        v = torch.zeros(columns, device=dev)
        v[1::2] = -0.0
        return v
    if kind == "fp16_randn":
        return (torch.randn(columns, generator=gen, device=dev) * 4).half().float()
    if kind == "special":
        v = torch.randn(columns, generator=gen, device=dev)
        v[::97] = float("inf")
        v[5::89] = float("-inf")
        v[3::131] = float("nan")
        # Negative quiet NaN, assigned as a tensor so the sign bit survives.
        v[7::151] = torch.tensor(
            [NEG_NAN_BITS], dtype=torch.int32, device=dev
        ).view(torch.float32)
        v[11::53] = -0.0
        v[13::59] = 0.0
        return v
    if kind == "ramp":
        # Strictly descending unique keys, every value above the FP16 range.
        return torch.arange(columns, 0, -1, dtype=torch.float32, device=dev) + 806736.0
    if kind == "narrow":
        # Every score shares the top key byte: the whole row lands in one
        # coarse bucket (maximal candidate compaction).
        return 1.0 + torch.rand(columns, generator=gen, device=dev)
    raise ValueError(kind)


def make_case(
    m: int, seed: int, columns: int = C.SCORE_COLUMNS, pad: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    buf = torch.empty((m, columns + pad), dtype=torch.float32, device="cuda")
    buf.fill_(float("nan"))  # garbage beyond the row width (padded stride)
    logits = buf[:, :columns]
    lengths = []
    for r in range(m):
        logits[r].copy_(make_row(KINDS[(r + seed) % len(KINDS)], columns, gen))
        lengths.append(LENGTH_PATTERN[(5 * r + seed) % len(LENGTH_PATTERN)])
    return logits, torch.tensor(lengths, dtype=torch.int32, device="cuda")


def assert_all_equal(logits, lengths, context: str, with_reference: bool = True):
    rows = run_rows(logits, lengths)
    generic = run_generic(logits, lengths)
    m1 = run_m1_per_row(logits, lengths)
    torch.cuda.synchronize()
    assert torch.equal(rows, generic), f"{context}: rows vs generic: " + (
        C.first_mismatch(rows, generic)
    )
    assert torch.equal(rows, m1), f"{context}: rows vs M1 per row: " + (
        C.first_mismatch(rows, m1)
    )
    if with_reference:
        ref = reference(logits, lengths)
        assert torch.equal(rows, ref), f"{context}: rows vs reference: " + (
            C.first_mismatch(rows, ref)
        )
    return rows


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", C.m_values())
@torch.inference_mode()
def test_rows_equal_generic_m1_and_reference(m: int) -> None:
    seeds = (0, 3) if C.quick() else (0, 1, 2, 3, 4, 5, 6)
    for seed in seeds:
        logits, lengths = make_case(m, seed)
        assert_all_equal(logits, lengths, f"M={m} seed={seed}")


@pytest.mark.parametrize("m", (1, 2, 8, 24, 32))
@torch.inference_mode()
def test_rows_padded_stride(m: int) -> None:
    for seed in (0, 1):
        logits, lengths = make_case(m, seed, pad=37)
        assert logits.stride(0) == C.SCORE_COLUMNS + 37
        assert_all_equal(logits, lengths, f"padded stride M={m} seed={seed}")


@pytest.mark.parametrize("m", (2, 4, 8, 16, 24, 32))
@pytest.mark.parametrize("context", (2100, 2600, 8200, 9300, 33000, 131000))
@torch.inference_mode()
def test_rows_realistic_decode(m: int, context: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(context + m)
    logits = torch.randn((m, C.SCORE_COLUMNS), generator=gen, device="cuda")
    # Decode rows of different requests: visible blocks ~ context / 4.
    jitter = torch.randint(0, 64, (m,), generator=gen, device="cuda")
    lengths = (context // C.COMPRESS_RATIO + jitter).to(torch.int32)
    assert_all_equal(logits, lengths, f"decode M={m} ctx={context}")


@torch.inference_mode()
def test_rows_op_contract() -> None:
    op = rows_op()
    empty = torch.empty((0, C.SCORE_COLUMNS), dtype=torch.float32, device="cuda")
    op(
        empty,
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty((0, C.TOPK), dtype=torch.int32, device="cuda"),
        C.TOPK,
    )
    logits, lengths = make_case(2, 0)
    out = torch.empty((2, C.TOPK), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        op(logits, lengths, out, 1024)
    with pytest.raises(RuntimeError):
        op(logits.half(), lengths, out, C.TOPK)
    with pytest.raises(RuntimeError):
        op(logits, lengths[:1], out, C.TOPK)


# ---------------------------------------------------------------------------
# CUDA graph
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", (2, 4, 8, 16, 24, 32))
@torch.inference_mode()
def test_rows_graph_replay_changing_inputs(m: int) -> None:
    replays = 32 if C.quick() else 128
    logits = torch.zeros((m, C.SCORE_COLUMNS), dtype=torch.float32, device="cuda")
    lengths = torch.zeros(m, dtype=torch.int32, device="cuda")
    out = torch.empty((m, C.TOPK), dtype=torch.int32, device="cuda")
    op = rows_op()
    graph = C.capture(lambda: op(logits, lengths, out, C.TOPK))
    gen = torch.Generator(device="cpu").manual_seed(1234 + m)
    for i in range(replays):
        new_logits, _ = make_case(m, 100 + i)
        new_lengths = [
            LENGTH_PATTERN[int(torch.randint(0, len(LENGTH_PATTERN), (1,), generator=gen))]
            if i % 2 == 0
            else int(torch.randint(0, 9000, (1,), generator=gen))
            for _ in range(m)
        ]
        logits.copy_(new_logits)
        lengths.copy_(torch.tensor(new_lengths, dtype=torch.int32))
        out.fill_(-7)
        graph.replay()
        expected = run_generic(logits, lengths)
        torch.cuda.synchronize()
        assert torch.equal(out, expected), f"replay {i} M={m}: " + (
            C.first_mismatch(out, expected)
        )


# ---------------------------------------------------------------------------
# Dispatch through qsa_select_paged_tokens
# ---------------------------------------------------------------------------
def build_decode_case(contexts: list[int], padded_rows: int = 0, seed: int = 0):
    """One decode row per request (the FULL decode-graph layout)."""
    generator = torch.Generator().manual_seed(seed)
    needed = [-(-max(c, 1) // C.SCHED_BLOCK) for c in contexts]
    total_pages = sum(needed) + 3
    physical = torch.randperm(total_pages, generator=generator).to(torch.int32)
    table = torch.full((len(contexts), C.INDEX_TABLE_WIDTH), -1, dtype=torch.int32)
    cursor = 0
    for request, count in enumerate(needed):
        table[request, :count] = physical[cursor : cursor + count]
        cursor += count
    token_to_req = list(range(len(contexts))) + [0] * padded_rows
    positions = [c - 1 for c in contexts] + [-1] * padded_rows
    rows = len(positions)
    cache = torch.randn(
        total_pages,
        C.COMPRESSED_PAGE,
        1,
        C.INDEX_DIM,
        dtype=torch.float16,
        device="cuda",
    )
    q = torch.randn(rows, C.INDEX_HEADS, C.INDEX_DIM, dtype=torch.float16, device="cuda")
    return dict(
        q=q,
        k_cache=cache,
        page_table=table.cuda(),
        token_to_req=torch.tensor(token_to_req, dtype=torch.int32, device="cuda"),
        query_positions=torch.tensor(positions, dtype=torch.int64, device="cuda"),
        sequence_lengths=torch.tensor(contexts, dtype=torch.int32, device="cuda"),
    )


class _Counting:
    def __init__(self, op):
        self.op = op
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.op(*args)


def _select(module, case, out):
    return module.qsa_select_paged_tokens(
        case["q"],
        case["k_cache"],
        case["page_table"],
        case["token_to_req"],
        case["query_positions"],
        case["sequence_lengths"],
        C.TOKEN_TOPK,
        C.COMPRESS_RATIO,
        out,
    )


CONTEXT_CYCLE = (2100, 2600, 8200, 9300, 33000, 1, 5, 2048, 70000, 131000)


@pytest.mark.parametrize(
    "m,padded", ((1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (17, 7), (24, 0), (32, 0), (33, 0))
)
@torch.inference_mode()
def test_select_paged_tokens_dispatch(m: int, padded: int, monkeypatch) -> None:
    module = qsa()
    real_rows = m - padded
    contexts = [CONTEXT_CYCLE[i % len(CONTEXT_CYCLE)] for i in range(real_rows)]
    case = build_decode_case(contexts, padded_rows=padded, seed=m)
    width = C.TOKEN_TOPK + C.COMPRESS_RATIO - 1

    counting = _Counting(rows_op())
    monkeypatch.setattr(module, "_sm70_qsa_lexicographic_topk_rows_op", lambda: counting)

    monkeypatch.setattr(module, "_SX_OPT_QSA_TOPK_ROWS", False)
    out_old = torch.full((m, width), -9, dtype=torch.int32, device="cuda")
    _select(module, case, out_old)
    assert counting.calls == 0

    monkeypatch.setattr(module, "_SX_OPT_QSA_TOPK_ROWS", True)
    out_new = torch.full((m, width), -9, dtype=torch.int32, device="cuda")
    _select(module, case, out_new)
    torch.cuda.synchronize()
    expected_calls = 1 if 2 <= m <= 32 else 0
    assert counting.calls == expected_calls, (m, counting.calls)
    assert torch.equal(out_new, out_old), f"M={m}: " + C.first_mismatch(out_new, out_old)


@pytest.mark.parametrize("m", (4, 24))
@torch.inference_mode()
def test_select_paged_tokens_graph(m: int, monkeypatch) -> None:
    module = qsa()
    contexts = [CONTEXT_CYCLE[i % len(CONTEXT_CYCLE)] for i in range(m)]
    case = build_decode_case(contexts, seed=77 + m)
    width = C.TOKEN_TOPK + C.COMPRESS_RATIO - 1

    monkeypatch.setattr(module, "_SX_OPT_QSA_TOPK_ROWS", False)
    expected = torch.empty((m, width), dtype=torch.int32, device="cuda")
    _select(module, case, expected)

    monkeypatch.setattr(module, "_SX_OPT_QSA_TOPK_ROWS", True)
    out = torch.empty((m, width), dtype=torch.int32, device="cuda")
    graph = C.capture(lambda: _select(module, case, out))
    for i in range(8):
        case["q"].copy_(torch.randn_like(case["q"]))
        monkeypatch.setattr(module, "_SX_OPT_QSA_TOPK_ROWS", False)
        _select(module, case, expected)
        out.fill_(-9)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected), f"graph replay {i} M={m}: " + (
            C.first_mismatch(out, expected)
        )


# ---------------------------------------------------------------------------
# Microbenchmark
# ---------------------------------------------------------------------------
BENCH_CONTEXTS = {
    # ~2K prompts during decode: 512..575 visible blocks (the common case).
    "2k": (2048, 2300),
    "8k": (8192, 8800),
    "32k": (32768, 33500),
    "mixed": None,
}


def _bench_lengths(m: int, name: str, gen: torch.Generator) -> torch.Tensor:
    if BENCH_CONTEXTS[name] is None:
        cycle = (2100, 8300, 2400, 33000, 2200, 9000)
        ctx = [cycle[i % len(cycle)] for i in range(m)]
    else:
        lo, hi = BENCH_CONTEXTS[name]
        ctx = torch.randint(lo, hi, (m,), generator=gen).tolist()
    return torch.tensor([c // C.COMPRESS_RATIO for c in ctx], dtype=torch.int32)


@pytest.mark.skipif(C.no_bench(), reason="SX_TEST_NO_BENCH")
@torch.inference_mode()
def test_bench_topk_rows() -> None:
    calls = 12  # QSA layers per decode step
    rows_table = []
    gen = torch.Generator().manual_seed(0)
    for name in BENCH_CONTEXTS:
        for m in C.BENCH_M:
            logits = [
                torch.randn((m, C.SCORE_COLUMNS), device="cuda") for _ in range(calls)
            ]
            lengths = [_bench_lengths(m, name, gen).cuda() for _ in range(calls)]
            outs_a = [torch.empty((m, C.TOPK), dtype=torch.int32, device="cuda") for _ in range(calls)]
            outs_b = [torch.empty((m, C.TOPK), dtype=torch.int32, device="cuda") for _ in range(calls)]
            old, new = old_op(), rows_op()
            result = C.bench_graphs(
                {
                    "old": lambda i: old(logits[i], lengths[i], outs_a[i], C.TOPK),
                    "rows": lambda i: new(logits[i], lengths[i], outs_b[i], C.TOPK),
                },
                calls=calls,
            )
            torch.cuda.synchronize()
            for i in range(calls):
                assert torch.equal(outs_a[i], outs_b[i]), (name, m, i)
            saving_ms = (result["old"] - result["rows"]) * calls / 1000.0
            rows_table.append(
                [
                    name,
                    m,
                    "decode" if m == 1 else "generic",
                    f"{result['old']:.2f}",
                    f"{result['rows']:.2f}",
                    f"{result['old'] / result['rows']:.2f}x",
                    f"{saving_ms:.3f}",
                ]
            )
            del logits, lengths, outs_a, outs_b
    C.print_table(
        "QSA block top-k, us/call (graph of 12 calls, median; old = existing launch)",
        ["ctx", "M", "old kernel", "old us", "rows us", "speedup", "ms/step saved"],
        rows_table,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
