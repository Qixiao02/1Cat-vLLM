# SPDX-License-Identifier: Apache-2.0
"""CPU-only logic check for the SX_OPT sync-free compact top-k/top-p.

No GPU, torch or Triton is needed (numpy only). The script mirrors
``_sx_compact_topk_topp_kernel`` and the threshold/scatter masking step by
step in float32 numpy. It fuzzes that mirror against a numpy port of
``apply_top_k_top_p_pytorch``, the reference contract: stable ascending sort
(ties by token id), keep >= k-th value, then softmax/cumsum top-p.

This validates the decision rules: row classes, the top-k threshold, the
top-p tie split, and the extended top-p-only/overflow rows. It does not
validate the Triton translation. The GPU test test_syncfree_topk_topp.py
does that on V100.

Run:  python sx_tests/sampler/sim_compact_topk_topp_logic.py
      (or python -m pytest -q on this file)
Expected: "unexplained mismatches: 0". A small number of "boundary" rows,
where an fp64 cumsum lies within 1e-5 of 1-p, is allowed.
"""

from __future__ import annotations

import numpy as np

F32 = np.float32
NEG_INF = F32(-np.inf)
POS_INF = F32(np.inf)


def reference_rows(x: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
    """numpy port of apply_top_k_top_p_pytorch (float32, stable sort)."""
    b, v = x.shape
    out = x.copy()
    for r in range(b):
        order = np.argsort(x[r], kind="stable")
        s = x[r][order].copy()
        kr = int(k[r])
        if kr < v:
            vk = s[v - kr]
            s[s < vk] = NEG_INF
        pr = F32(p[r])
        m = s.max()
        with np.errstate(invalid="ignore", over="ignore"):
            e = np.exp((s - m).astype(F32)).astype(F32)
            probs = (e / e.sum(dtype=F32)).astype(F32)
            cum = np.cumsum(probs, dtype=F32)
            mask = cum <= (F32(1.0) - pr)
        mask[-1] = False
        s[mask] = NEG_INF
        out[r][order] = s
    return out


def compact_rows(
    x: np.ndarray, k: np.ndarray, p: np.ndarray, kt: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[str]]:
    """Mirror of the kernel. Returns (masked logits, per-row class).
    Hard rows are returned unmodified (the Qrita pass finishes them)."""
    b, v = x.shape
    out = x.copy()
    classes = []
    for r in range(b):
        row = x[r]
        # torch.topk(sorted=False): any KT largest; randomise the tie choice at
        # the KT-th value to show the result does not depend on it.
        noise = rng.random(v) * 1e-3
        order_desc = np.lexsort((noise, -row.astype(np.float64)))
        cand_i = order_desc[:kt]
        cand_v = row[cand_i].copy()
        cand_v[cand_v == 0.0] = F32(0.0)  # canonical zero
        srt = np.lexsort((cand_i, cand_v))  # (value, id) ascending
        s_v = cand_v[srt]
        s_idx = cand_i[srt]
        lanes = np.arange(kt)
        top_lane = kt - 1
        v_max = s_v.max()
        v_last = s_v.min()

        k_raw = int(k[r])
        topk_on = 0 < k_raw < v
        k_c = min(max(k_raw, 1), kt)
        v_k = s_v[kt - k_c]
        pr = F32(p[r])
        topp_on = pr < 1.0

        finite_max = np.isfinite(v_max)
        all_masked = v_max == NEG_INF
        # v_last == -inf: fewer than KT finite logits, every finite logit is a
        # candidate, so k > KT is exact too (v_k = lane 0 = -inf).
        k_fits = topk_on and (k_raw <= kt or v_last == NEG_INF)
        complete = (v_last < v_k) or (v_k == NEG_INF)
        compact = k_fits and complete and finite_max
        overflow = k_fits and (v_k == v_last) and (v_k > NEG_INF) and finite_max
        topp_only = (not topk_on) and topp_on and finite_max
        extended = overflow or topp_only
        noop = ((not topk_on) and (not topp_on)) or all_masked

        kth = v_k if topk_on else NEG_INF
        keep_k = s_v >= kth

        need_scan = extended and topp_on
        below_sum = F32(0.0)
        n_eq_total = 0
        with np.errstate(invalid="ignore", over="ignore"):
            if need_scan:
                below = row < v_last
                below_sum = np.exp((row[below] - v_max).astype(F32)).astype(F32).sum(
                    dtype=F32
                )
                n_eq_total = int((row == v_last).sum())
            n_eq_cand = int((s_v == v_last).sum())
            extra_eq = (n_eq_total - n_eq_cand) if need_scan else 0
            e_last = F32(np.exp(F32(v_last - v_max)))
            tail = (below_sum if topp_only else F32(0.0)) + (
                F32(extra_eq) * e_last if extra_eq > 0 else F32(0.0)
            )
            tail = F32(tail)
            e = np.where(keep_k, np.exp((s_v - v_max).astype(F32)), F32(0.0)).astype(
                F32
            )
            z = F32(e.sum(dtype=F32) + tail)
            q = (e / z).astype(F32)
            cum = (np.cumsum(q, dtype=F32) + F32(tail / z)).astype(F32)
            rm = (cum <= (F32(1.0) - pr)) & (lanes != top_lane)
        keep = keep_k & ~(rm & topp_on)
        thr = s_v[keep].min() if keep.any() else POS_INF
        ext_ok = extended and (((not topp_on) and overflow) or (thr > v_last))
        handled = compact or ext_ok
        hard = not (handled or noop)
        if handled:
            out_row = out[r]
            out_row[out_row < thr] = NEG_INF
            scatter = (~keep) & (s_v >= thr)
            out_row[s_idx[scatter]] = NEG_INF
            classes.append("compact" if compact else "extended")
        elif noop:
            classes.append("noop")
        else:
            classes.append("hard")
    return out, classes


def boundary_distance(x: np.ndarray, k: int, p: float) -> float:
    """fp64 min |cum - (1-p)| of the reference computation (inf if p >= 1)."""
    if p >= 1.0:
        return float("inf")
    v = x.shape[0]
    s = np.sort(x.astype(np.float64), kind="stable")
    if k < v:
        s[s < s[v - k]] = -np.inf
    if not np.isfinite(s).any():
        return float("inf")
    e = np.exp(s - s.max())
    probs = e / e.sum()
    cum = np.cumsum(probs)
    nz = probs > 0
    return float(np.abs(cum[nz] - (1.0 - p)).min())


def make_batch(rng: np.random.Generator, b: int, v: int, kt: int):
    x = (rng.standard_normal((b, v)) * 2.0).astype(F32)
    n_top = min(256, v // 4)
    for r in range(b):
        ids = rng.integers(0, v, n_top)
        x[r, ids] += (rng.random(n_top) ** 3 * 16.0 + 2.0).astype(F32)
    x = x.astype(np.float16).astype(F32)  # FP16 LM head -> natural ties
    temps = rng.choice([0.3, 0.3, 0.3, 0.7, 1.0, 2.0], size=b).astype(F32)
    x = (x / temps[:, None]).astype(F32)
    k = rng.choice(
        [20, 20, 20, 20, 1, 5, 19, 21, kt - 16, kt - 1, kt, kt + 1, 200, v],
        size=b,
    ).astype(np.int64)
    p = rng.choice([0.95, 0.95, 0.95, 0.9, 0.8, 0.5, 0.2, 1.0], size=b).astype(F32)
    kinds = rng.choice(
        [
            "plain",
            "plain",
            "plain",
            "grammar",
            "ktie",
            "ktie_span",
            "ptie",
            "uniform",
            "allninf",
            "zeros",
            "flat",
        ],
        size=b,
    )
    for r in range(b):
        kind = kinds[r]
        if kind == "grammar":
            n = int(rng.choice([1, 5, 19, 20, 21, 500]))
            keep = rng.choice(v, size=n, replace=False)
            row = np.full(v, NEG_INF, dtype=F32)
            row[keep] = x[r, keep]
            x[r] = row
        elif kind in ("ktie", "ktie_span"):
            order = np.argsort(-x[r].astype(np.float64), kind="stable")
            kk = int(min(k[r], v - 50)) if k[r] < v else 20
            t = int(rng.integers(2, 41))
            lo = kk - 1 if kind == "ktie" else max(0, kk - 1 - t // 2)
            x[r, order[lo : lo + t]] = x[r, order[kk - 1]]
        elif kind == "ptie":
            row = np.full(v, F32(-20.0), dtype=F32)
            ids = rng.choice(v, size=40, replace=False)
            n_hi = int(rng.integers(1, 4))
            n_tie = int(rng.integers(5, 36))
            row[ids[:n_hi]] = F32(2.0)
            row[ids[n_hi : n_hi + n_tie]] = F32(1.0)
            x[r] = row
        elif kind == "uniform":
            x[r] = F32(1.0)
        elif kind == "allninf":
            x[r] = NEG_INF
        elif kind == "zeros":
            order = np.argsort(-x[r].astype(np.float64), kind="stable")
            x[r] = x[r] - x[r, order[19]]  # 20th largest becomes 0
            x[r, order[15:25]] = F32(0.0)  # a +0.0/-0.0 tie group across k=20
            x[r, order[15:25:2]] = F32(-0.0)
        elif kind == "flat":
            x[r] = (rng.standard_normal(v) * 0.05).astype(np.float16).astype(F32)
        if rng.random() < 0.1:
            k[r] = v  # top-k disabled for this row
    return x, k, p, kinds


def run(seed: int = 0, iters: int = 60, b: int = 16, v: int = 8192, kt: int = 64):
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    exact = boundary = unexplained = 0
    for _ in range(iters):
        x, k, p, kinds = make_batch(rng, b, v, kt)
        ref = reference_rows(x, k, p)
        new, classes = compact_rows(x, k, p, kt, rng)
        for r in range(b):
            cls = classes[r]
            counts[cls] = counts.get(cls, 0) + 1
            if cls == "hard":
                continue
            if cls == "noop":
                # The reference with p=1/k=V still masks zero-probability
                # entries (cumsum <= 0). The legacy Qrita route skips them too.
                same = np.array_equal(new[r], x[r])
            else:
                same = np.array_equal(new[r], ref[r])
                if not same and p[r] >= 1.0:
                    # p=1 rows: the reference also removes kept entries whose
                    # FP32 probability underflows to 0 (never samplable). The
                    # legacy Qrita route and this route keep them.
                    with np.errstate(invalid="ignore", over="ignore"):
                        e = np.exp((x[r] - x[r].max()).astype(F32))
                    diff = new[r] != ref[r]
                    same = bool(np.all(e[diff] == 0))
            if same:
                exact += 1
            elif boundary_distance(x[r], int(k[r]), float(p[r])) < 1e-5:
                boundary += 1
            else:
                unexplained += 1
                print(
                    f"UNEXPLAINED row: kind={kinds[r]} class={cls} k={k[r]} "
                    f"p={p[r]} new_kept={int(np.isfinite(new[r]).sum())} "
                    f"ref_kept={int(np.isfinite(ref[r]).sum())}"
                )
    print(
        f"seed={seed} v={v} kt={kt}: classes={counts} exact={exact} "
        f"boundary={boundary} unexplained mismatches: {unexplained}"
    )
    return unexplained


def test_large_k_with_few_finite_logits_is_exact():
    """k > KT on a row with fewer than KT finite logits (a grammar mask) is
    handled exactly; with at least KT finite logits it stays hard."""
    rng = np.random.default_rng(7)
    v, kt = 4096, 64
    for n_finite, expect in ((1, "compact"), (40, "compact"), (kt - 1, "compact"),
                             (kt, "hard"), (500, "hard")):
        for p in (0.95, 0.6, 1.0):
            x = np.full((1, v), NEG_INF, dtype=F32)
            ids = rng.choice(v, size=n_finite, replace=False)
            x[0, ids] = (rng.standard_normal(n_finite) * 3.0).astype(np.float16)
            k = np.array([200], dtype=np.int64)
            pp = np.array([p], dtype=F32)
            new, classes = compact_rows(x, k, pp, kt, rng)
            assert classes == [expect], (n_finite, p, classes)
            if expect == "compact":
                ref = reference_rows(x, k, pp)
                assert np.array_equal(new, ref) or (
                    boundary_distance(x[0], 200, p) < 1e-5
                ), (n_finite, p)


def test_compact_logic_matches_reference():
    for seed, v, kt in ((0, 8192, 64), (1, 8192, 128), (2, 4096, 32)):
        assert run(seed=seed, iters=25, b=16, v=v, kt=kt) == 0


if __name__ == "__main__":
    bad = 0
    for seed, v, kt in ((0, 8192, 64), (1, 8192, 128), (2, 4096, 32), (3, 248320, 128)):
        bad += run(seed=seed, iters=40 if v < 100000 else 3, b=16, v=v, kt=kt)
    raise SystemExit(1 if bad else 0)
