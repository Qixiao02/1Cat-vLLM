# SPDX-License-Identifier: Apache-2.0
"""design_5 [cpu-2]: grouped mamba-align pre-copy / post-process grids.

Needs 1 GPU (V100) with ~6 GB free. Run inside the deployed image after the
overlay is installed, from a directory where ``import vllm`` resolves to the
installed package:

    cd /tmp && /opt/venv/bin/python -m pytest -q -s \
        <repo>/sx_tests/align-chunking/test_mamba_align_copy_grid.py
    cd /tmp && /opt/venv/bin/python \
        <repo>/sx_tests/align-chunking/test_mamba_align_copy_grid.py   # bench only

Layout mirrors the TP4-local Qwen3.8 align cache: 36 GDN layers in 3 groups
(conv state [3, 2048] fp16 and temporal state [8, 128, 128] fp32 inside one
802816-byte padded page per block) plus the PLE short-conv state [9, 10240]
fp16 in a 4th group: 73 states, 4 block tables, 784-token state blocks.

Asserts: for M in {1,2,3,4,8,16,17,24,32} requests with a random mix of fresh
requests, non-crossing decode rows, single- and multi-block crossings (the
multi-block chunks of SX_OPT_ALIGN_MULTIBLOCK), prefix-hit seeds, padded
idx_mapping rows and speculative token_bias, the grouped kernels produce
byte-identical state tensors (torch.equal on the integer reinterpretation of
every state view, since random bytes as fp16/fp32 contain NaNs, and on the raw
pages) and identical num_accepted outputs compared with the old per-state
grid. Both grids are also checked against a torch reference of the copy rules
(so two equally wrong or no-op kernels cannot pass). Prints a microbenchmark
(CUDA events, median of 60 x 20 launches, both eager wall time and GPU-only
time; the kernels run eagerly on the main stream in production, outside CUDA
graphs). The grouped grid is the default only on SM70 devices.
Expected: all pass; grouped grid faster when no request crosses a block.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

B = 784
PAGE = 802816
GDN_LAYERS = 36
GDN_GROUPS = 3
NUM_COLS = 3
MAX_M = 32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


def _mod():
    from vllm.v1.worker.gpu import mamba_align

    return mamba_align


def _state_views(pages: list[torch.Tensor], num_blocks: int):
    """(state view, group) list over the raw pages, in ctx state order."""
    states = []
    for layer in range(GDN_LAYERS):
        buf = pages[layer]
        conv = torch.as_strided(buf.view(torch.float16), (num_blocks, 3, 2048),
                                (PAGE // 2, 2048, 1), 0)
        temporal = torch.as_strided(buf.view(torch.float32),
                                    (num_blocks, 8, 128, 128),
                                    (PAGE // 4, 128 * 128, 128, 1), 12288 // 4)
        group = layer // (GDN_LAYERS // GDN_GROUPS)
        states += [(conv, group), (temporal, group)]
    ple = torch.as_strided(pages[GDN_LAYERS].view(torch.float16),
                           (num_blocks, 9, 10240), (PAGE // 2, 10240, 1), 0)
    states.append((ple, GDN_GROUPS))
    return states


def _make_states(num_blocks: int, device, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pages = [
        torch.randint(0, 256, (num_blocks * PAGE,), dtype=torch.uint8,
                      device=device, generator=gen)
        for _ in range(GDN_LAYERS + 1)
    ]
    return _state_views(pages, num_blocks), pages


def _make_ctx(states, block_tables, device):
    conv = [t.dim() == 3 and t.dtype == torch.float16 for t, _ in states]
    return SimpleNamespace(
        is_initialized=True,
        num_states=len(states),
        block_size=B,
        block_table_ptrs=torch.tensor([bt.data_ptr() for bt in block_tables],
                                      dtype=torch.int64, device=device),
        block_table_stride_req=block_tables[0].stride(0),
        state_base_addrs=torch.tensor([t.data_ptr() for t, _ in states],
                                      dtype=torch.int64, device=device),
        state_block_strides=torch.tensor(
            [t.stride(0) * t.element_size() for t, _ in states],
            dtype=torch.int64, device=device),
        state_elem_sizes=torch.tensor([t.element_size() for t, _ in states],
                                      dtype=torch.int32, device=device),
        state_inner_sizes=torch.tensor(
            [t.stride(1) if c else t[0].numel() for (t, _), c in zip(states, conv)],
            dtype=torch.int64, device=device),
        state_conv_widths=torch.tensor(
            [t.size(1) if c else 0 for (t, _), c in zip(states, conv)],
            dtype=torch.int32, device=device),
        state_group_indices=torch.tensor([g for _, g in states],
                                         dtype=torch.int32, device=device),
        num_accepted_tokens_out=torch.zeros(MAX_M + 4, dtype=torch.int32,
                                            device=device),
    )


class _Fixture:
    def __init__(self, device="cuda", seed=0):
        self.device = torch.device(device)
        self.num_blocks = MAX_M * NUM_COLS + 1
        rng = torch.Generator().manual_seed(seed)
        self.block_tables = []
        for _ in range(GDN_GROUPS + 1):
            ids = torch.randperm(self.num_blocks - 1, generator=rng)[
                : MAX_M * NUM_COLS] + 1
            self.block_tables.append(
                ids.to(torch.int32).reshape(MAX_M, NUM_COLS).to(self.device)
            )
        self.states_a, self.pages_a = _make_states(self.num_blocks, self.device, seed)
        self.states_b, self.pages_b = _make_states(self.num_blocks, self.device, seed)
        self.ctx_a = _make_ctx(self.states_a, self.block_tables, self.device)
        self.ctx_b = _make_ctx(self.states_b, self.block_tables, self.device)

    def reset(self):
        for a, b in zip(self.pages_a, self.pages_b):
            b.copy_(a)

    def assert_equal(self):
        torch.cuda.synchronize()
        # Random bytes viewed as fp16/fp32 contain NaNs and NaN != NaN, so a
        # float torch.equal would fail even for identical bytes. Compare the
        # integer reinterpretation (same element size) and the raw pages.
        for (a, _), (b, _) in zip(self.states_a, self.states_b):
            assert torch.equal(_bits(a), _bits(b))
        for a, b in zip(self.pages_a, self.pages_b):
            assert torch.equal(a, b)


_INT_VIEW = {torch.float16: torch.int16, torch.float32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(_INT_VIEW.get(t.dtype, t.dtype))


@pytest.fixture(scope="module")
def fx():
    return _Fixture()


def _precopy_case(m: int, rng: random.Random, spec: bool, device):
    num_slots = MAX_M + 4
    state_idx = torch.full((num_slots,), -1, dtype=torch.int32)
    src_col = torch.full((num_slots,), -1, dtype=torch.int32)
    token_bias = torch.zeros(num_slots, dtype=torch.int32)
    slots = rng.sample(range(num_slots), m)  # persistent request-slot order
    idx_mapping = []
    for batch_idx in range(m):
        slot = slots[batch_idx]
        kind = rng.choice(["fresh", "decode", "cross1", "crossN", "hit", "pad"])
        if kind == "pad" and batch_idx > 0:
            idx_mapping.append(-1)
            continue
        idx_mapping.append(slot)
        if kind == "fresh":
            src, dst = -1, rng.randrange(NUM_COLS)
        elif kind == "decode":
            src = dst = rng.randrange(NUM_COLS)
        elif kind == "cross1":
            src = rng.randrange(NUM_COLS - 1)
            dst = src + 1
        elif kind == "crossN":
            src, dst = 0, NUM_COLS - 1
        else:  # prefix hit seed: hit column -> later column
            src = rng.randrange(NUM_COLS - 1)
            dst = rng.randrange(src + 1, NUM_COLS)
        bias = 0
        if spec and src >= 0 and src != dst:
            bias = rng.randrange(0, NUM_COLS - src)
        state_idx[slot], src_col[slot], token_bias[slot] = dst, src, bias
    return (state_idx.to(device), src_col.to(device), token_bias.to(device),
            torch.tensor(idx_mapping, dtype=torch.int32, device=device))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 17, 24, 32])
@pytest.mark.parametrize("spec", [False, True])
def test_precopy_grouped_is_bitwise(fx, m: int, spec: bool) -> None:
    mod = _mod()
    rng = random.Random(1000 * m + spec)
    for _ in range(3):
        fx.reset()
        state_idx, src_col, token_bias, idx_mapping = _precopy_case(
            m, rng, spec, fx.device)
        mod.run_mamba_align_precopy(fx.ctx_a, m, state_idx, src_col, token_bias,
                                    idx_mapping, grouped=False)
        mod.run_mamba_align_precopy(fx.ctx_b, m, state_idx, src_col, token_bias,
                                    idx_mapping, grouped=True)
        fx.assert_equal()


def _postprocess_case(m: int, rng: random.Random, spec: bool, device):
    num_slots = MAX_M + 4
    accepted = torch.ones(num_slots, dtype=torch.int32)
    state_idx = torch.zeros(num_slots, dtype=torch.int32)
    new_computed = torch.zeros(num_slots, dtype=torch.int32)
    slots = rng.sample(range(num_slots), m)
    idx_mapping = []
    for batch_idx in range(m):
        slot = slots[batch_idx]
        if batch_idx > 0 and rng.random() < 0.1:
            idx_mapping.append(-1)
            continue
        idx_mapping.append(slot)
        n_acc = rng.randrange(1, 4) if spec else 1
        k = rng.randrange(1, NUM_COLS)  # boundary k*B -> dst column k-1
        if rng.random() < 0.3:
            # running state strictly past the boundary: early exit, no copy
            new = k * B + rng.randrange(n_acc, B // 2)
            src = (new - 1) // B
        else:
            # running state at k*B - bias; bias > 0 with src == dst is the
            # overlapping conv memmove. Keep src + bias inside the row: src is
            # drawn from [k - 1, NUM_COLS - bias), so bias <= NUM_COLS - k.
            bias = rng.randrange(0, min(n_acc, NUM_COLS - k + 1))
            new = k * B - bias + n_acc - 1
            src = rng.randrange(k - 1, NUM_COLS - bias)
        accepted[slot] = n_acc
        state_idx[slot] = src
        new_computed[slot] = new
    return (accepted.to(device), state_idx.to(device), new_computed.to(device),
            torch.tensor(idx_mapping, dtype=torch.int32, device=device))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 17, 24, 32])
@pytest.mark.parametrize("spec", [False, True])
def test_postprocess_grouped_is_bitwise(fx, m: int, spec: bool) -> None:
    mod = _mod()
    rng = random.Random(7000 * m + spec)
    for _ in range(3):
        fx.reset()
        accepted, state_idx, new_computed, idx_mapping = _postprocess_case(
            m, rng, spec, fx.device)
        acc_a, acc_b = accepted.clone(), accepted.clone()
        mod.run_mamba_align_postprocess(fx.ctx_a, m, acc_a, state_idx,
                                        new_computed, idx_mapping, grouped=False)
        mod.run_mamba_align_postprocess(fx.ctx_b, m, acc_b, state_idx,
                                        new_computed, idx_mapping, grouped=True)
        fx.assert_equal()
        assert torch.equal(acc_a, acc_b)


# ------------------------------------------------------------------------------
# Reference semantics: old-vs-grouped equality alone would also pass if both
# kernels copied nothing (or the same wrong bytes), so check each grid against
# a torch reference of the V1 align copy rules. The expectation is built in
# place on the fixture's second page set (no extra ~3 GB clone): distinct
# (row, column) cells own distinct blocks, so the only aliasing is a copy within
# one block (conv memmove, temporal src + bias == dst), handled by .clone().
# ------------------------------------------------------------------------------
def _reference_copy(states, block_tables_cpu, row: int, src: int, dst: int,
                    bias: int) -> None:
    for state, group in states:
        s = _bits(state)
        table = block_tables_cpu[group][row]
        dst_block = int(table[dst])
        if s.dim() == 3:  # conv / PLE state: dst[: w - bias] = src[bias:]
            width = s.size(1)
            s[dst_block, : width - bias] = s[int(table[src]), bias:].clone()
        else:  # temporal state: whole block from the accepted column
            s[dst_block] = s[int(table[src + bias])].clone()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("spec", [False, True])
def test_precopy_matches_reference(fx, grouped: bool, spec: bool) -> None:
    mod = _mod()
    rng = random.Random(4242 + 2 * spec + grouped)
    bt_cpu = [bt.cpu() for bt in fx.block_tables]
    m = 12
    total_copies = 0
    for _ in range(4):
        fx.reset()  # pages_b := pages_a; pages_b becomes the expectation
        state_idx, src_col, token_bias, idx_mapping = _precopy_case(
            m, rng, spec, fx.device)
        si, sc, tb = (t.cpu().tolist() for t in (state_idx, src_col, token_bias))
        for row, slot in enumerate(idx_mapping.cpu().tolist()):
            if slot < 0 or sc[slot] < 0 or sc[slot] == si[slot]:
                continue
            _reference_copy(fx.states_b, bt_cpu, row, sc[slot], si[slot], tb[slot])
            total_copies += 1
        mod.run_mamba_align_precopy(fx.ctx_a, m, state_idx, src_col, token_bias,
                                    idx_mapping, grouped=grouped)
        fx.assert_equal()
    assert total_copies > 0, "no crossing row was generated: test is vacuous"


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("spec", [False, True])
def test_postprocess_matches_reference(fx, grouped: bool, spec: bool) -> None:
    mod = _mod()
    rng = random.Random(9090 + 2 * spec + grouped)
    bt_cpu = [bt.cpu() for bt in fx.block_tables]
    m = 12
    total_copies = 0
    for _ in range(4):
        fx.reset()  # pages_b := pages_a; pages_b becomes the expectation
        accepted, state_idx, new_computed, idx_mapping = _postprocess_case(
            m, rng, spec, fx.device)
        acc = accepted.cpu().tolist()
        acc_want = list(acc)
        si, nc = state_idx.cpu().tolist(), new_computed.cpu().tolist()
        for row, slot in enumerate(idx_mapping.cpu().tolist()):
            if slot < 0:
                continue
            running = nc[slot] - acc[slot] + 1
            aligned = nc[slot] // B * B
            if aligned < running:
                continue
            bias, dst, src = aligned - running, aligned // B - 1, si[slot]
            if src == dst:
                acc_want[slot] = 1
                if bias == 0:
                    continue
            _reference_copy(fx.states_b, bt_cpu, row, src, dst, bias)
            total_copies += 1
        acc_got = accepted.clone()
        mod.run_mamba_align_postprocess(fx.ctx_a, m, acc_got, state_idx,
                                        new_computed, idx_mapping, grouped=grouped)
        fx.assert_equal()
        assert acc_got.cpu().tolist() == acc_want
    assert total_copies > 0, "no copying row was generated: test is vacuous"


def test_default_grid_is_grouped() -> None:
    """Default: grouped on SM70 unless SX_OPT_MAMBA_ALIGN_COPY_GRID=0."""
    import os

    mod = _mod()
    env_on = os.environ.get("SX_OPT_MAMBA_ALIGN_COPY_GRID", "1") != "0"
    sm70 = torch.cuda.get_device_capability() == (7, 0)
    assert mod._USE_GROUPED_COPY_GRID == env_on
    assert mod._use_grouped_grid(None) == (env_on and sm70)
    assert mod._use_grouped_grid(True) is True
    assert mod._use_grouped_grid(False) is False


# ------------------------------------------------------------------------------
# microbenchmark
# ------------------------------------------------------------------------------
def _time(fn, iters=60, inner=20, gpu_only=False) -> float:
    """Median us per fn() call.

    gpu_only=False: events around eager launches (includes the Python/Triton
    launch cost, which is the same for both grids).
    gpu_only=True: a ~10 ms device sleep is queued first so every launch is
    enqueued before the GPU reaches the start event; the event delta is then
    the GPU execution time of the kernels alone.
    """
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        if gpu_only:
            torch.cuda._sleep(15_000_000)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000.0 / inner)
    times.sort()
    return times[len(times) // 2]


def run_benchmark(fx: _Fixture | None = None) -> list[tuple]:
    mod = _mod()
    fx = fx or _Fixture()
    dev = fx.device
    rows = []
    for m in (1, 2, 4, 8, 16, 24):
        # decode: no row crosses (the common step); one_cross: one row crosses
        # a block (~3% of C24 decode steps, and each prefill chunk after the
        # first); all_cross: every row crosses (worst case for the grouped grid,
        # which spreads one row's copy over 8 x 16 CTAs instead of 73 x 16).
        for scenario in ("decode", "one_cross", "all_cross"):
            slots = list(range(m))
            state_idx = torch.ones(MAX_M + 4, dtype=torch.int32, device=dev)
            src_col = torch.ones(MAX_M + 4, dtype=torch.int32, device=dev)
            if scenario == "one_cross":
                src_col[0] = 0  # one request crosses one block
            elif scenario == "all_cross":
                src_col[:m] = 0
            token_bias = torch.zeros(MAX_M + 4, dtype=torch.int32, device=dev)
            idx = torch.tensor(slots, dtype=torch.int32, device=dev)
            accepted = torch.ones(MAX_M + 4, dtype=torch.int32, device=dev)
            new_computed = torch.full((MAX_M + 4,), B + 100, dtype=torch.int32,
                                      device=dev)
            res = []
            for grouped in (False, True):
                ctx = fx.ctx_b

                def step(grouped=grouped, ctx=ctx):
                    mod.run_mamba_align_precopy(ctx, m, state_idx, src_col,
                                                token_bias, idx, grouped=grouped)
                    mod.run_mamba_align_postprocess(ctx, m, accepted, state_idx,
                                                    new_computed, idx,
                                                    grouped=grouped)

                res.append((_time(step), _time(step, gpu_only=True)))
            rows.append((m, scenario, res[0], res[1]))
    print("\nmamba-align pre+post copy per step (us, median; 'gpu' excludes "
          "launch overhead):")
    print(f"{'M':>3} {'scenario':>10} {'old eager':>10} {'new eager':>10} "
          f"{'old gpu':>9} {'new gpu':>9} {'gpu speedup':>11}")
    for m, sc, (old_e, old_g), (new_e, new_g) in rows:
        print(f"{m:>3} {sc:>10} {old_e:>10.1f} {new_e:>10.1f} {old_g:>9.1f} "
              f"{new_g:>9.1f} {old_g / new_g:>10.2f}x")
    return rows


def test_benchmark_prints(fx) -> None:
    rows = run_benchmark(fx)
    assert rows


if __name__ == "__main__":
    import vllm

    print("vllm package:", vllm.__file__, "GPU:", torch.cuda.get_device_name())
    run_benchmark()
