# SPDX-License-Identifier: Apache-2.0
"""SX_OPT sync-free SM70 top-k/top-p: equivalence tests + microbenchmark.

Needs ONE V100 (SM70) in the deployed image, with the overlay installed.
Either the patched vllm is copied into /opt/venv, or the source tree is
first on PYTHONPATH.

    /opt/venv/bin/python -m pytest -q sx_tests/sampler/test_syncfree_topk_topp.py
    /opt/venv/bin/python sx_tests/sampler/test_syncfree_topk_topp.py  # tests + bench

The old implementation is sx_tests/sampler/baseline_topk_topp_triton.py,
a verbatim copy of vllm/v1/sample/ops/topk_topp_triton.py at git 71c1822.
It is routed exactly like the legacy ``apply_top_k_top_p``: B>=2 goes to
Qrita plus the host-synchronised reference fallback, and B==1 goes to
``apply_top_k_top_p_pytorch`` (the full-sort reference).

What is asserted, V=248320 FP32 logits made from FP16 values divided by the
temperature (natural FP16 ties), B in {1,2,3,4,8,16,17,24,32}:

1. Every row the new path handles itself (compact / extended / noop) is
   bitwise equal to the full-vocabulary reference ``apply_top_k_top_p_pytorch``.
   Allowed exceptions: "boundary" rows, where an fp64 recomputation puts a
   cumulative probability within 1e-5 of 1-p (FP32 rounding decides both
   the reference and the legacy Qrita pivot there); and p=1 rows, where the
   reference masks entries whose FP32 probability underflows to 0. Those
   can never be sampled, and the legacy Qrita route keeps them too.
2. Every "hard" row (finished by the unchanged Qrita kernel) is bitwise
   equal to the Qrita kernel output without the reference fallback.
3. New vs old (legacy route): every mismatching row must be explained by
   (1) boundary, (2) a hard row that the legacy route re-masked with the
   reference, or (3) a legacy row that itself deviates from the reference.
   For the deployment parameters (T=0.3, top_k=20, top_p=0.95, plus greedy
   rows) the unexplained count must be 0, and the counts are printed.
4. Tokens from the V2 Gumbel sampler (``gumbel_sample``, same seeds and
   positions) are equal for every row whose masked logits are equal. This
   is also checked through the real MRV2 ``SamplingStates`` call path.
5. No host synchronisation: routed calls run under
   torch.cuda.set_sync_debug_mode("error"). The legacy route raises there
   (shown as a control).
6. With SX_OPT_TOPK_TOPP_SYNCFREE=0 the patched module equals the baseline
   bitwise.
7. (review) Top-k-only batches (p=None: the P=None / TOPP_ENABLED=False
   kernel variants, k int32 and int64, including the MTP warm-up call with
   all-zero logits) equal the sort reference bitwise on every non-hard row,
   and run without host synchronisation.
8. (review) Row-strided logits and per-row parameters that are not dense
   (``param.expand(n)``, stride-2 slices) give the dense result bitwise and
   leave memory outside the logits window untouched.
9. (review) The exact tie fixtures of tests/v1/sample/test_topk_topp_tied_cutoffs.py,
   forced through the new path at V=32768 and V=248320.

Also run the legacy 8-warp regression with the new route off, otherwise it
compares the new route with itself:
    SX_OPT_TOPK_TOPP_SYNCFREE=0 /opt/venv/bin/python -m pytest -q \
        tests/v1/sample/test_topk_topp_sampler.py -k 8_warps
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
V = 248320
NEG_INF = float("-inf")
BOUNDARY_TOL = 1e-5
BATCHES = (1, 2, 3, 4, 8, 16, 17, 24, 32)

CUDA = torch.cuda.is_available()
SM70 = CUDA and torch.cuda.get_device_capability() == (7, 0)
pytestmark = pytest.mark.skipif(not SM70, reason="needs one SM70 (V100) GPU")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@contextlib.contextmanager
def env(**kv):
    old = {key: os.environ.get(key) for key in kv}
    try:
        for key, val in kv.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(val)
        yield
    finally:
        for key, val in old.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


_BASE = None


def base():
    global _BASE
    if _BASE is None:
        spec = importlib.util.spec_from_file_location(
            "sx_baseline_topk_topp_triton", HERE / "baseline_topk_topp_triton.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BASE = mod
    return _BASE


def mods():
    from vllm.v1.sample.ops import topk_topp_sampler as ts
    from vllm.v1.sample.ops import topk_topp_triton as tt

    assert hasattr(tt, "apply_top_k_top_p_sm70_syncfree"), (
        "SX_OPT overlay not installed: the imported vllm has no sync-free path "
        f"({tt.__file__})"
    )
    return tt, ts


def legacy_apply(x, k, p):
    """The legacy apply_top_k_top_p routing, using the baseline module."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    b, v = x.shape
    if b >= 2 and v >= 32768:
        return base().apply_top_k_top_p_triton(x, k, p)
    return apply_top_k_top_p_pytorch(x, k, p)


def qrita_native(x, k, p):
    """Baseline Qrita kernel without the reference fallback -> (out, flags)."""
    from vllm.utils.platform_utils import num_compute_units

    b_mod = base()
    x = x.clone()
    b, v = x.shape
    num_sm = num_compute_units(x.device.index)
    num_programs = min(num_sm, b)
    buffer = x.new_empty((num_programs, v))
    cdf = x.new_tensor(b_mod._NORMAL_CDF_TO_SIGMA_TABLE)
    pct = x.new_tensor(b_mod._PERCENTILE_TO_STD_TABLE)
    flags = torch.empty(b, dtype=torch.bool, device=x.device)
    kwargs = {}
    if b_mod._use_sm70_topk_topp_8_warps(x.device, b, v, k is not None, p is not None):
        kwargs["num_warps"] = 8
    b_mod._topk_topp_kernel[(num_programs,)](
        x,
        x.stride(0),
        buffer,
        pct,
        cdf,
        k.to(torch.int32) if k is not None else x,
        p.to(torch.float32) if p is not None else x,
        BATCH_SIZE=b,
        MASK_VALUE=NEG_INF,
        VOCAB_SIZE=v,
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=k is not None,
        TOPP_ENABLED=p is not None,
        REFERENCE_ROWS=flags,
        **kwargs,
    )
    return x, flags


def row_classes(x, k, p):
    """Row classes the new path assigns (0 noop, 1 handled, 2 hard)."""
    tt, _ = mods()
    cfg = tt._sx_config()
    thresh, k_hard, p_hard = tt._sx_compact_select(
        x.clone(),
        k.to(torch.int32),
        p.to(torch.float32) if p is not None else None,
        cfg.kt,
        None,
    )
    hard = k_hard < x.shape[1]
    if p is not None:
        hard |= p_hard < 1.0
    handled = thresh > NEG_INF
    return hard.cpu(), handled.cpu()


def realistic_logits(b, seed, temps):
    """FP16 LM-head-like logits -> FP32 -> / temperature, as MRV2 samples."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(b, V, generator=g, device="cuda") * 2.0
    n_top = 256
    ids = torch.randint(0, V, (b, n_top), generator=g, device="cuda")
    boost = torch.rand(b, n_top, generator=g, device="cuda").pow(3) * 16.0 + 2.0
    x.scatter_add_(1, ids, boost)
    x = x.half().float()
    t = torch.as_tensor(temps, dtype=torch.float32, device="cuda").view(-1, 1)
    t = torch.where(t == 0, torch.ones_like(t), t)  # T=0 rows are not scaled
    return x / t


def apply_fixture(x, r, kind, rng, k_row):
    """In-place adversarial row fixtures (grammar masks, ties, degenerate)."""
    if kind == "plain":
        return
    row = x[r]
    if kind.startswith("grammar"):
        n = int(kind[len("grammar") :])
        keep = torch.as_tensor(rng.choice(V, size=n, replace=False), device="cuda")
        # grammar-allowed tokens: some from the head, some random
        head = row.topk(8).indices
        keep = torch.cat([keep[: max(0, n - 3)], head[: min(3, n)]])
        vals = row[keep].clone()
        row.fill_(NEG_INF)
        row[keep] = vals
    elif kind.startswith("ktie"):
        t = int(kind[len("ktie") :])
        kk = k_row if 0 < k_row < V else 20
        vals, ids = row.topk(kk + 64)
        row[ids[kk - 1 : kk - 1 + t]] = vals[kk - 1]
    elif kind.startswith("kspan"):
        t = int(kind[len("kspan") :])
        kk = k_row if 0 < k_row < V else 20
        vals, ids = row.topk(kk + 64)
        lo = max(0, kk - 1 - t // 2)
        row[ids[lo : lo + t]] = vals[kk - 1]
    elif kind.startswith("ptie"):
        n_tie = int(kind[len("ptie") :])
        ids = torch.as_tensor(rng.choice(V, size=n_tie + 2, replace=False), device="cuda")
        row.fill_(-20.0)
        row[ids[:2]] = 2.0
        row[ids[2:]] = 1.0
    elif kind == "uniform":
        row.fill_(1.0)
    elif kind == "allninf":
        row.fill_(NEG_INF)
    elif kind == "zeros":
        vals, ids = row.topk(30)
        row -= vals[19].item()
        row[ids[15:25]] = 0.0
        row[ids[15:25:2]] = -0.0
    elif kind == "flat":
        row.copy_((torch.randn(V, device="cuda") * 0.05).half().float())
    else:
        raise ValueError(kind)


def make_case(b, seed, params, kinds=None):
    """params: list of (T, top_k (V=disabled), top_p); cycled over rows."""
    rng = np.random.default_rng(seed)
    rows = [params[(i + seed) % len(params)] for i in range(b)]
    temps = [r[0] for r in rows]
    x = realistic_logits(b, seed, temps)
    k = torch.tensor([r[1] for r in rows], dtype=torch.int32, device="cuda")
    p = torch.tensor([r[2] for r in rows], dtype=torch.float32, device="cuda")
    if kinds:
        for r in range(b):
            apply_fixture(x, r, kinds[(r + seed) % len(kinds)], rng, int(k[r]))
    return x, k, p, torch.tensor(temps, dtype=torch.float32, device="cuda")


def boundary_distance(orig_row, k, p):
    """fp64 min |cumsum - (1-p)| of the reference computation (inf if p>=1)."""
    if p >= 1.0:
        return math.inf
    s = torch.sort(orig_row.double(), stable=True).values
    if 0 < k < V:
        s = torch.where(s < s[V - k], torch.full_like(s, NEG_INF), s)
    if not torch.isfinite(s).any():
        return math.inf
    probs = torch.softmax(s, 0)
    cum = torch.cumsum(probs, 0)
    nz = probs > 0
    return float((cum[nz] - (1.0 - p)).abs().min())


def zero_prob_only_diff(orig_row, a, b):
    """True if a and b differ only where the FP32 probability is exactly 0."""
    diff = ~((a == b) | (torch.isneginf(a) & torch.isneginf(b)))
    if not diff.any():
        return True
    e = torch.exp(orig_row - orig_row.max())
    return bool((e[diff] == 0).all())


def gumbel_tokens(masked, temps, seeds, pos):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    idx = torch.arange(masked.shape[0], dtype=torch.int32, device="cuda")
    return gumbel_sample(
        masked,
        idx,
        temps,
        seeds,
        pos,
        apply_temperature=False,
        is_drafting=False,
    )


def compare_case(x, k, p, temps, strict_old: bool):
    """Run old/new/ref/qrita on one batch, check all invariants, return stats."""
    tt, ts = mods()
    b = x.shape[0]
    with env(SX_OPT_TOPK_TOPP_SYNCFREE="1"):
        assert tt.sm70_syncfree_topk_topp_eligible(x, k, p)
        new = ts.apply_top_k_top_p(x.clone(), k, p)
    old = legacy_apply(x.clone(), k, p)
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    ref = apply_top_k_top_p_pytorch(x.clone(), k, p)
    qn, qflags = qrita_native(x, k, p)
    hard, handled = row_classes(x, k, p)
    qflags = qflags.cpu()

    stats = dict(rows=b, hard=int(hard.sum()), handled=int(handled.sum()))
    stats.update(
        new_eq_ref=0,
        boundary=0,
        p1_underflow=0,
        hard_eq_qrita=0,
        new_eq_old=0,
        old_diff_boundary=0,
        old_diff_hard_flagged=0,
        old_diff_legacy_dev=0,
        old_diff_unexplained=0,
        tokens_equal=0,
        token_diff_rows=0,
    )
    seeds = torch.randint(-(2**62), 2**62, (b,), dtype=torch.int64, device="cuda")
    pos = torch.randint(0, 131072, (b,), dtype=torch.int64, device="cuda")
    tok_new = gumbel_tokens(new, temps, seeds, pos).cpu()
    tok_old = gumbel_tokens(old, temps, seeds, pos).cpu()

    for r in range(b):
        kr, pr = int(k[r]), float(p[r])
        bdist = None

        def is_boundary():
            nonlocal bdist
            if bdist is None:
                bdist = boundary_distance(x[r], kr, pr)
            return bdist < BOUNDARY_TOL

        # (1)/(2) the new path against its contract
        if bool(hard[r]):
            assert torch.equal(new[r], qn[r]), f"hard row {r} != Qrita native"
            stats["hard_eq_qrita"] += 1
        elif torch.equal(new[r], ref[r]):
            stats["new_eq_ref"] += 1
        elif pr >= 1.0 and zero_prob_only_diff(x[r], new[r], ref[r]):
            stats["p1_underflow"] += 1
        else:
            assert is_boundary(), (
                f"row {r} (k={kr}, p={pr}) differs from the reference away from "
                f"the top-p boundary: kept new={int(torch.isfinite(new[r]).sum())} "
                f"ref={int(torch.isfinite(ref[r]).sum())}"
            )
            stats["boundary"] += 1

        # (3) new vs old
        if torch.equal(new[r], old[r]):
            stats["new_eq_old"] += 1
        elif pr >= 1.0 and zero_prob_only_diff(x[r], new[r], old[r]):
            stats["new_eq_old"] += 1  # only never-samplable entries differ
        elif is_boundary():
            stats["old_diff_boundary"] += 1
        elif bool(hard[r]) and bool(qflags[r]):
            stats["old_diff_hard_flagged"] += 1
        elif not torch.equal(old[r], ref[r]) and not (
            pr >= 1.0 and zero_prob_only_diff(x[r], old[r], ref[r])
        ):
            stats["old_diff_legacy_dev"] += 1
        else:
            stats["old_diff_unexplained"] += 1

        # (4) tokens
        same_logits = torch.equal(new[r], old[r])
        if tok_new[r] == tok_old[r]:
            stats["tokens_equal"] += 1
        else:
            stats["token_diff_rows"] += 1
            assert not same_logits, f"row {r}: equal masked logits, different token"

    assert stats["old_diff_unexplained"] == 0, stats
    if strict_old:
        assert stats["old_diff_legacy_dev"] == 0, stats
        assert stats["old_diff_hard_flagged"] == 0, stats
    return stats


def _sum_stats(acc, s):
    for key, val in s.items():
        acc[key] = acc.get(key, 0) + val
    return acc


# --------------------------------------------------------------------------
# parameter sets
# --------------------------------------------------------------------------

DEPLOY = [(0.3, 20, 0.95)] * 7 + [(0.0, V, 1.0)]  # + a greedy row
MIXED = [
    (0.3, 20, 0.95),
    (0.7, 20, 0.95),
    (1.0, 20, 0.8),
    (0.3, 20, 1.0),
    (0.3, 1, 0.95),
    (0.3, 5, 0.5),
    (0.5, 50, 0.9),
    (0.3, 64, 0.95),
    (0.3, 100, 0.95),
    (0.0, V, 1.0),
    (0.3, V, 0.95),  # top-p only (extended when the nucleus fits)
    (0.7, V, 0.9),
]
HARD = [
    (0.3, 20, 0.95),
    (1.0, V, 0.95),  # flat-ish top-p only -> may be hard
    (0.3, 200, 0.95),  # k > KT
    (0.3, 1000, 0.9),
    (0.3, 200, 1.0),
    (2.0, V, 0.99),
]
FIXTURES = [
    "plain",
    "grammar1",
    "grammar5",
    "grammar19",
    "grammar20",
    "grammar21",
    "grammar500",
    "ktie2",
    "ktie10",
    "ktie40",
    "kspan10",
    "kspan30",
    "ptie18",
    "ptie30",
    "zeros",
    "allninf",
    "uniform",
    "flat",
]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("b", BATCHES)
def test_deployment_params_exact(b):
    """T=0.3/top_k=20/top_p=0.95 plus greedy rows: new == old, same tokens."""
    total = {}
    for seed in range(3):
        x, k, p, temps = make_case(b, seed, DEPLOY)
        s = compare_case(x, k, p, temps, strict_old=True)
        assert s["hard"] == 0, s
        _sum_stats(total, s)
    print(f"[deploy B={b}] {total}")


@pytest.mark.parametrize("b", (2, 8, 16, 24))
def test_deployment_params_grammar_and_ties(b):
    """Deployment params on adversarial rows (grammar masks, k/p ties, zeros)."""
    kinds = [
        f
        for f in FIXTURES
        if f.startswith(("plain", "grammar", "ktie", "kspan", "ptie", "zeros"))
    ]
    total = {}
    for seed in range(4):
        x, k, p, temps = make_case(b, seed, DEPLOY, kinds)
        _sum_stats(total, compare_case(x, k, p, temps, strict_old=False))
    print(f"[deploy+fixtures B={b}] {total}")
    assert total["old_diff_unexplained"] == 0


@pytest.mark.parametrize("b", (1, 2, 8, 16, 24, 32))
def test_mixed_per_row_params(b):
    total = {}
    for seed in range(3):
        x, k, p, temps = make_case(b, seed, MIXED, FIXTURES)
        _sum_stats(total, compare_case(x, k, p, temps, strict_old=False))
    print(f"[mixed B={b}] {total}")


@pytest.mark.parametrize("b", (2, 8, 24))
def test_hard_rows(b):
    total = {}
    for seed in range(3):
        x, k, p, temps = make_case(b, seed, HARD, ["plain", "flat", "uniform", "ktie10"])
        _sum_stats(total, compare_case(x, k, p, temps, strict_old=False))
    print(f"[hard B={b}] {total}")
    assert total["hard"] > 0


@pytest.mark.parametrize("kt", (32, 64, 256, 1024))
def test_kt_variants(kt):
    with env(SX_OPT_TOPK_TOPP_COMPACT_KT=kt):
        total = {}
        for b in (1, 8, 24):
            x, k, p, temps = make_case(b, kt + b, MIXED, FIXTURES)
            _sum_stats(total, compare_case(x, k, p, temps, strict_old=False))
    print(f"[KT={kt}] {total}")


def test_top_p_only_batch_keeps_legacy_route():
    tt, ts = mods()
    x, _, p, _ = make_case(8, 0, DEPLOY)
    assert not tt.sm70_syncfree_topk_topp_eligible(x, None, p)
    out = ts.apply_top_k_top_p(x.clone(), None, p)
    assert torch.equal(out, legacy_apply(x.clone(), None, p))


def test_switch_off_is_baseline_bitwise():
    tt, ts = mods()
    for b in (1, 2, 8, 24):
        x, k, p, _ = make_case(b, 11, MIXED, FIXTURES)
        with env(SX_OPT_TOPK_TOPP_SYNCFREE="0"):
            assert not tt.sm70_syncfree_topk_topp_eligible(x, k, p)
            out = ts.apply_top_k_top_p(x.clone(), k, p)
            out_triton = tt.apply_top_k_top_p_triton(x.clone(), k, p)
        assert torch.equal(out, legacy_apply(x.clone(), k, p))
        assert torch.equal(out_triton, base().apply_top_k_top_p_triton(x.clone(), k, p))
    with env(SX_OPT_TOPK_TOPP_SYNCFREE_B1="0"):
        x, k, p, _ = make_case(1, 12, DEPLOY)
        assert not tt.sm70_syncfree_topk_topp_eligible(x, k, p)


def test_sampling_states_call_path():
    """The real MRV2 call path: SamplingStates -> apply_top_k_top_p -> Gumbel."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    rows = [
        (0.3, 20, 0.95),
        (0.3, 20, 0.95),
        (0.0, -1, 1.0),
        (0.7, 40, 0.9),
        (0.3, -1, 0.95),
        (1.0, 20, 1.0),
    ] * 4
    b = len(rows)
    st = SamplingStates(max_num_reqs=64, vocab_size=V)
    for i, (t, kk, pp) in enumerate(rows):
        st.add_request(
            i, SamplingParams(temperature=t, top_k=kk, top_p=pp, seed=1000 + i)
        )
    st.apply_staged_writes()
    idx_np = np.arange(b, dtype=np.int32)
    idx = torch.from_numpy(idx_np).cuda()
    g = torch.Generator(device="cuda").manual_seed(5)
    raw = (torch.randn(b, V, generator=g, device="cuda") * 2.0).half()
    raw[:, :64] += torch.linspace(12, 2, 64, device="cuda").half()
    x = raw.float()
    st.apply_temperature(x, idx, idx_np)
    with env(SX_OPT_TOPK_TOPP_SYNCFREE="1"):
        new = st.apply_top_k_top_p(x.clone(), idx, idx_np)
    with env(SX_OPT_TOPK_TOPP_SYNCFREE="0"):
        old = st.apply_top_k_top_p(x.clone(), idx, idx_np)
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    pos = torch.randint(0, 131072, (b,), dtype=torch.int64, device="cuda")
    tok = {}
    for name, masked in (("new", new), ("old", old)):
        tok[name] = gumbel_sample(
            masked,
            idx,
            st.temperature.gpu,
            st.seeds.gpu,
            pos,
            apply_temperature=False,
            is_drafting=False,
        ).cpu()
    same_rows = [torch.equal(new[r], old[r]) for r in range(b)]
    print(f"[SamplingStates] rows equal {sum(same_rows)}/{b}")
    for r in range(b):
        if same_rows[r]:
            assert tok["new"][r] == tok["old"][r]
    assert torch.equal(tok["new"], tok["old"]), (tok["new"], tok["old"])


@pytest.mark.parametrize("b", (1, 24))
def test_no_host_sync(b):
    tt, ts = mods()
    x, k, p, _ = make_case(b, 3, DEPLOY)
    ts.apply_top_k_top_p(x.clone(), k, p)  # compile / warm caches
    xs = [x.clone() for _ in range(3)]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for xi in xs:
            ts.apply_top_k_top_p(xi, k, p)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if b >= 2:
        # Control: the legacy route synchronises and raises in this mode.
        with env(SX_OPT_TOPK_TOPP_SYNCFREE="0"):
            ts.apply_top_k_top_p(x.clone(), k, p)
            xc = x.clone()
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                with pytest.raises(RuntimeError):
                    ts.apply_top_k_top_p(xc, k, p)
            finally:
                torch.cuda.set_sync_debug_mode("default")


def test_stats_counters():
    tt, ts = mods()
    with env(SX_OPT_TOPK_TOPP_STATS="1", SX_OPT_TOPK_TOPP_STATS_EVERY="1000000"):
        before = tt.get_sm70_syncfree_topk_topp_stats()
        x, k, p, _ = make_case(24, 4, DEPLOY)
        ts.apply_top_k_top_p(x.clone(), k, p)
        after = tt.get_sm70_syncfree_topk_topp_stats()
    delta = {key: after[key] - before[key] for key in after}
    print(f"[stats] {delta}")
    assert delta["compact"] + delta["noop"] == 24 and delta["hard"] == 0


# --------------------------------------------------------------------------
# review additions: top-k-only batches (P=None kernel specialisation),
# strided logits / expanded per-row params, the exact tied-cutoff fixtures of
# tests/v1/sample/test_topk_topp_tied_cutoffs.py routed through the new path.
# --------------------------------------------------------------------------

TOPK_ONLY_KS = (1, 5, 20, 50, 64, 100, 127, 128, 129, 200, V)


def _check_topk_only(x, k):
    """p=None: every non-hard row must equal the sort reference bitwise (no
    top-p, so there is no boundary or p=1 exception); hard rows (k > KT) must
    equal the Qrita kernel output. Returns the number of hard rows."""
    tt, ts = mods()
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    with env(SX_OPT_TOPK_TOPP_SYNCFREE="1"):
        assert tt.sm70_syncfree_topk_topp_eligible(x, k, None)
        new = ts.apply_top_k_top_p(x.clone(), k, None)
    ref = apply_top_k_top_p_pytorch(x.clone(), k, None)
    qn, _ = qrita_native(x, k, None)
    hard, _ = row_classes(x, k, None)
    for r in range(x.shape[0]):
        if bool(hard[r]):
            assert torch.equal(new[r], qn[r]), f"hard row {r} != Qrita native"
        else:
            assert torch.equal(new[r], ref[r]), (
                f"top-k-only row {r} (k={int(k[r])}) differs from the reference: "
                f"kept new={int(torch.isfinite(new[r]).sum())} "
                f"ref={int(torch.isfinite(ref[r]).sum())}"
            )
    return int(hard.sum())


@pytest.mark.parametrize("k_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("b", (1, 2, 8, 24))
def test_topk_only_batches(b, k_dtype):
    """p=None compiles and runs the P=None / TOPP_ENABLED=False variants of
    the compact kernel and of the Qrita finishing pass (V1 sampler, MTP draft
    warm-up and spec-decode callers pass top-k without top-p)."""
    params = [(0.3, kk, 1.0) for kk in TOPK_ONLY_KS]
    kinds = ["plain", "grammar5", "grammar21", "ktie10", "ktie40", "kspan30",
             "zeros", "uniform", "allninf", "flat"]
    n_hard = 0
    for seed in range(3):
        x, k, _, _ = make_case(b, seed, params, kinds)
        n_hard += _check_topk_only(x, k.to(k_dtype))
    print(f"[top-k only B={b} {k_dtype}] hard rows={n_hard}")


@pytest.mark.parametrize("b", (1, 8))
def test_large_k_on_grammar_rows_is_exact(b):
    """top_k > KT on a row with fewer than KT finite logits (grammar mask):
    every finite logit is a candidate, so the row is handled exactly (not
    hard) and matches the reference. With >= KT finite logits it stays hard."""
    tt, ts = mods()
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    params = [(0.3, 129, 0.95), (0.7, 200, 0.6), (1.0, 1000, 0.95), (0.3, 200, 1.0)]
    kinds = ["grammar1", "grammar5", "grammar21", "grammar127"]
    for seed in range(3):
        x, k, p, _ = make_case(b, seed, params, kinds)
        hard, _ = row_classes(x, k, p)
        assert not bool(hard.any()), hard
        with env(SX_OPT_TOPK_TOPP_SYNCFREE="1"):
            new = ts.apply_top_k_top_p(x.clone(), k, p)
        ref = apply_top_k_top_p_pytorch(x.clone(), k, p)
        for r in range(b):
            if torch.equal(new[r], ref[r]):
                continue
            pr = float(p[r])
            if pr >= 1.0 and zero_prob_only_diff(x[r], new[r], ref[r]):
                continue
            assert boundary_distance(x[r], int(k[r]), pr) < BOUNDARY_TOL, (
                f"row {r} (k={int(k[r])}, p={pr}) differs from the reference"
            )
    x, k, p, _ = make_case(2, 0, [(0.3, 200, 0.95)], ["grammar500"])
    hard, _ = row_classes(x, k, p)
    assert bool(hard.all())


def test_mtp_warmup_pattern_zero_logits():
    """llm_base_proposer warms the path with all-zero logits, top_k=20 and
    p=None: every entry ties at the k-th value, so every entry is kept."""
    for b in (1, 4, 24):
        x = torch.zeros(b, V, device="cuda")
        k = torch.full((b,), 20, dtype=torch.int32, device="cuda")
        assert _check_topk_only(x, k) == 0


@pytest.mark.parametrize("b", (2, 8, 24))
def test_no_host_sync_topk_only(b):
    _, ts = mods()
    x, k, _, _ = make_case(b, 5, [(0.3, 20, 1.0), (0.3, V, 1.0)])
    ts.apply_top_k_top_p(x.clone(), k, None)  # compile / warm caches
    xc = x.clone()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ts.apply_top_k_top_p(xc, k, None)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("b", (2, 8, 24))
def test_strided_logits_and_expanded_params(b):
    """Row-strided logits (a column window of a wider buffer) and per-row
    parameters that are not dense 1-D arrays (``param.expand(n)`` as the
    spec-decode draft path builds them, or a stride-2 slice). The results must
    equal the dense run bitwise, and the buffer outside the window must be
    untouched. The sentinels after the first parameter element make a kernel
    that indexes K/P densely without honouring strides fail deterministically.
    """
    tt, ts = mods()
    x, _, _, _ = make_case(b, 21, DEPLOY[:-1], ["plain", "grammar21", "ktie10"])
    k_dense = torch.full((b,), 20, dtype=torch.int32, device="cuda")
    p_dense = torch.full((b,), 0.95, dtype=torch.float32, device="cuda")
    with env(SX_OPT_TOPK_TOPP_SYNCFREE="1"):
        dense = ts.apply_top_k_top_p(x.clone(), k_dense, p_dense)

        k_base = torch.full((2 * b,), V, dtype=torch.int32, device="cuda")
        p_base = torch.full((2 * b,), 1.0, dtype=torch.float32, device="cuda")
        k_base[0] = 20
        p_base[0] = 0.95
        k_exp, p_exp = k_base[:1].expand(b), p_base[:1].expand(b)
        k_base2 = torch.full((2 * b,), V, dtype=torch.int32, device="cuda")
        p_base2 = torch.full((2 * b,), 1.0, dtype=torch.float32, device="cuda")
        k_base2[::2] = 20
        p_base2[::2] = 0.95
        k_str, p_str = k_base2[::2], p_base2[::2]
        assert k_exp.stride(0) == 0 and k_str.stride(0) == 2

        for kk, pp in ((k_exp, p_exp), (k_str, p_str)):
            big = torch.full((b, V + 96), 7.0, device="cuda")
            view = big[:, 32 : 32 + V]
            view.copy_(x)
            assert view.stride(0) == V + 96 and view.stride(1) == 1
            assert tt.sm70_syncfree_topk_topp_eligible(view, kk, pp)
            out = ts.apply_top_k_top_p(view, kk, pp)
            assert out.data_ptr() == view.data_ptr()  # in place
            assert torch.equal(out, dense)
            assert bool((big[:, :32] == 7.0).all())
            assert bool((big[:, 32 + V :] == 7.0).all())


@pytest.mark.parametrize("vocab", (32768, V))
@pytest.mark.parametrize("top_p", (1.0, 0.95, 0.6))
@pytest.mark.parametrize("case", ("k_tie", "p_tie", "uniform", "unique"))
def test_tied_cutoff_fixtures_new_path(case, top_p, vocab):
    """The exact tie fixtures of tests/v1/sample/test_topk_topp_tied_cutoffs.py
    (which stay on the legacy route at V=32768) forced through the new path.
    Every row must match the full-vocabulary reference bitwise, except the
    all-tied "uniform" row with top-p < 1: that is a hard row by design and
    gets the Qrita kernel's own result (deviation (b) in the module
    docstring; the legacy route re-masked it with the reference)."""
    tt, _ = mods()
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    x = torch.full((8, vocab), -20.0, device="cuda")
    if case == "k_tie":
        x[:, :24] = 1.0
        x[:, :18] = 2.0
    elif case == "p_tie":
        x[:, :20] = 1.0
        x[:, :2] = 2.0
    elif case == "uniform":
        x.fill_(1.0)
    else:
        x[:, :32] = torch.arange(32, 0, -1, device="cuda") / 8
    k = torch.full((8,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((8,), top_p, device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    with env(SX_OPT_TOPK_TOPP_SYNCFREE_VOCABS="*"):
        assert tt.sm70_syncfree_topk_topp_eligible(x, k, p)
        hard, _ = row_classes(x, k, p)
        actual = tt.apply_top_k_top_p_triton(x.clone(), k, p)
    if case == "uniform" and top_p < 1.0:
        assert bool(hard.all())
        qn, _ = qrita_native(x, k, p)
        assert torch.equal(actual, qn)
        return
    assert not bool(hard.any())
    assert torch.equal(actual, expected)


# --------------------------------------------------------------------------
# microbenchmark
# --------------------------------------------------------------------------


def _bench(fn, src, iters=100, warmup=10):
    """GPU time (CUDA events around fn only) and host time per call, medians."""
    x = src.clone()
    for _ in range(warmup):
        x.copy_(src)
        fn(x)
    torch.cuda.synchronize()
    gpu, host = [], []
    for _ in range(iters):
        x.copy_(src)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        e.record()
        torch.cuda.synchronize()
        gpu.append(s.elapsed_time(e))
        host.append((t1 - t0) * 1e3)
    return statistics.median(gpu), statistics.median(host)


def _host_block_ms(fn, src, busy_cycles=int(7e6), iters=20):
    """Host time spent inside fn while the GPU is still busy with earlier work
    (the production situation: sampling is enqueued behind the forward)."""
    x = src.clone()
    fn(x)
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        x.copy_(src)
        torch.cuda._sleep(busy_cycles)  # ~5 ms of GPU work ahead of the sampler
        t0 = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    return statistics.median(out)


def run_benchmark():
    tt, ts = mods()

    def new_fn(k, p):
        return lambda x: ts.apply_top_k_top_p(x, k, p)

    def old_fn(k, p):
        return lambda x: legacy_apply(x, k, p)

    print(
        "\nB   | legacy gpu ms | new gpu ms | legacy host ms | new host ms | "
        "host blocked behind 5ms GPU work: legacy / new | torch.topk(KT) gpu ms"
    )
    for b in (1, 2, 4, 8, 16, 24, 32):
        x, k, p, _ = make_case(b, 100 + b, DEPLOY[:-1])  # all rows T=.3 k=20 p=.95
        og, oh = _bench(old_fn(k, p), x)
        ng, nh = _bench(new_fn(k, p), x)
        ob = _host_block_ms(old_fn(k, p), x)
        nb = _host_block_ms(new_fn(k, p), x)
        kt = tt._sx_config().kt
        tg, _ = _bench(lambda xx: torch.topk(xx, kt, dim=1, sorted=False), x)
        print(
            f"{b:<3} | {og:13.3f} | {ng:10.3f} | {oh:14.3f} | {nh:11.3f} | "
            f"{ob:9.3f} / {nb:6.3f} | {tg:.3f}"
        )
    print("\nKT sweep at B=24 (new path gpu ms / host ms):")
    x, k, p, _ = make_case(24, 7, DEPLOY[:-1])
    for kt in (32, 64, 128, 256, 512):
        with env(SX_OPT_TOPK_TOPP_COMPACT_KT=kt):
            g, h = _bench(new_fn(k, p), x)
        print(f"  KT={kt:<4}: {g:.3f} / {h:.3f}")
    print("\nMixed params with hard rows (B=24): legacy vs new gpu ms")
    x, k, p, _ = make_case(24, 9, HARD, ["plain", "flat"])
    print(
        f"  legacy {_bench(old_fn(k, p), x)[0]:.3f}  new {_bench(new_fn(k, p), x)[0]:.3f}"
    )


if __name__ == "__main__":
    import sys

    rc = pytest.main([__file__, "-q", "-s"])
    if SM70:
        run_benchmark()
    sys.exit(rc)
