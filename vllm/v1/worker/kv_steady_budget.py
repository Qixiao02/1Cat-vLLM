# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steady-state KV budget for the SM70 V2 lanes (SX_OPT_KV_STEADY_BUDGET).

Pure arithmetic: no torch, no CUDA, no vLLM imports, so every number below is
testable on a machine without a GPU (see sx_tests/kv-steady-budget/).

Why this exists. ``Worker.determine_available_memory`` sizes the KV cache from
two ``profile_run()`` calls. That run has no KV cache and skips attention, so
everything the engine allocates afterwards is invisible to the budget. On the
4x V100-SXM2-32GB native-MTP lane the highest nvidia-smi ``memory.used`` is
31467 MiB at ``--gpu-memory-utilization 0.87`` (1003 code; 31811 MiB with the
1001 code): ~3.1-3.5 GiB above the utilisation budget, so the utilisation
cannot simply be raised.

Where that overshoot comes from (read from the code, nothing here is measured
on a GPU; the first three rows are not post-sizing at all):

    CUDA context + NCCL communicator + custom all-reduce  0.6-0.9 GiB  GUESSED
        (taken before the init snapshot, so outside util*total by design;
         custom all-reduce alone is 26.9 MiB, computed)
    CUDA graph capture: 27 PIECEWISE size sets of the target (every capture
        size plus the 19 PW-1 sizes 96..1024) x ~50 pieces, 8 FULL verify
        graphs, the draft-prefill and draft-decode managers  0.5-1.5 GiB  GUESSED
        (the no-MTP lane's whole capture is logged as 1.04 GiB)
    allocator blocks cached by the capture streams (the serving stream does not
        reuse them) and tuner leftovers                      0.2-0.8 GiB  GUESSED
    attention/indexer/GDN workspaces of a full prefill chunk that only the
        first long real request allocates (the warm-up runs 6-48 token prompts)
                                                             0.3-0.6 GiB  partly computed
    kernel modules loaded at their first launch              0.1-0.8 GiB  GUESSED
    per-stream TurboMind/cuBLAS/QSA workspaces (3 capture streams)
                                                             0.1-0.2 GiB  computed per stream
    verify sampler at 80 rows, top-k/top-p buffers           0.15-0.22 GiB computed

Already inside the KV budget, so not part of the overshoot: the 8192-token
activation peak of the profile run (MoE eager buffers 478 MiB, PLE short-conv
322-480 MiB, HC), the drafter/target persistent buffers allocated at load
(~384 MiB), the NVFP4 decode buffers (97 MiB), and the pinned PLE table, which
takes no device data memory (only ~25 MiB of GPU page tables, counted as
non-torch memory at load).

What the switch does (everything else in the worker stays as it is):

1. The graph pool is reserved before the KV cache is sized, as upstream does
   for SM70 V2 (4bbaf64fc, ``get_sm70_cudagraph_memory_reserve``: the profiled
   activation peak): ``available = requested - non_kv - graph_reserve``. This is
   the *utilisation* bound.
2. A *physical* bound is added: the KV cache may not take memory that the
   rest of the steady state needs. ``F_prof`` is the device-free memory right
   after the measured profile (activations freed; weights, residue, CUDA
   context and NCCL all live). The steady state still has to find room for the
   activation peak again, for every post-sizing allocation ``P`` and for a
   fixed headroom ``H`` (CUDA-free memory, default 576 MiB)::

       kv_physical = F_prof - activation_peak - P - H

   The KV cache is ``min(kv_utilisation, kv_physical)``. With the physical bound
   the utilisation stops deciding whether the card fits: any value high enough
   yields the largest safe KV cache, and the log says which value that is.
3. ``P`` comes from a lane reference (the highest measured memory.used of the
   lane at a known utilisation, minus what that utilisation budgeted, minus
   this device's own CUDA-context footprint) unless
   ``SX_OPT_KV_STEADY_RESERVE_MIB`` gives it. The references exist for the two
   admitted Qwen3.8 TP4 lanes only; any other model gets the graph reserve plus
   1 GiB.
4. One full-chunk prefill joins the warm-up (``SX_OPT_KV_STEADY_WARMUP_TOKENS``)
   so the lazily allocated workspaces exist, and are measured, before serving;
   and the capture streams' idle allocator blocks are released after the graph
   capture (what the V1 runner already does).
5. After the warm-up ``audit()`` compares the measured growth with ``P``; the
   worker logs the free memory at every startup phase boundary.

What it can and cannot give. The physical bound reproduces today's footprint at
the reference point, so the KV cache grows by exactly the headroom today's
sizing leaves unused: ``total - reference_peak - H`` (MTP lane: ~470 MiB if the
card's CUDA total is 32510 MiB, ~730 MiB if it is 32768 MiB), plus whatever the
idle-cache release frees once ``P`` is calibrated from the audit. Budgeting
cannot create memory: upstream's 139015-token KV cache at util 0.95 needs a
smaller non-KV footprint, not a better formula.

Nothing in here can measure ``P`` before the KV cache exists: V2 captures its
graphs against the KV cache, so a dry capture needs a provisional cache that is
torn down and re-created (the V1 runner's ``profile_cudagraph_memory`` does
that). That is deliberately not done; ``audit()`` and the per-phase memory
marks exist to calibrate the estimate instead.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

MiB = 1 << 20
GiB = 1 << 30

ENV_SWITCH = "SX_OPT_KV_STEADY_BUDGET"
ENV_HEADROOM_MIB = "SX_OPT_KV_STEADY_HEADROOM_MIB"
ENV_RESERVE_MIB = "SX_OPT_KV_STEADY_RESERVE_MIB"
ENV_LOAD_MIB = "SX_OPT_KV_STEADY_LOAD_MIB"
ENV_STRICT = "SX_OPT_KV_STEADY_STRICT"
ENV_WARMUP_TOKENS = "SX_OPT_KV_STEADY_WARMUP_TOKENS"
ENV_EMPTY_CACHE = "SX_OPT_KV_STEADY_EMPTY_CACHE"
ENV_EXTRA_MIB = "SX_OPT_KV_STEADY_EXTRA_MIB"  # only for lanes without a reference
ENV_CONTEXT_CAP_MIB = "SX_OPT_KV_STEADY_CONTEXT_CAP_MIB"

# Free device memory (CUDA's view: total minus used) the steady state must
# still have at its peak. The task this was built for asked for ~500 MiB and a
# peak below 32200 MiB on a card that nvidia-smi lists with 32768 MiB: 576 MiB
# keeps the predicted peak at or below 32192 MiB on such a card (and below 31934
# MiB where CUDA's usable total is the 32510 MiB that torch reports).
DEFAULT_HEADROOM_MIB = 576
# Growth between the end of the warm-up and the worst load the lane was
# measured under (four 8K prefills: +312 MiB MTP, +158 MiB no-MTP) plus margin
# for the 16K/32K shapes. It is part of the lane reference below, so it only
# matters for the audit, which cannot see load-time growth.
DEFAULT_LOAD_MIB = 384
# Reserve for lanes without a measured reference (DFlash2, other methods):
# upstream's activation-peak graph reserve plus this.
DEFAULT_UNREFERENCED_EXTRA_MIB = 1024
# Upper bound on the part of ``init_snapshot.cuda_memory`` that is taken to be
# this engine's own CUDA context + NCCL + custom all-reduce (0.6-0.9 GiB by the
# code, not measured). The lane reference subtracts that footprint from its
# overshoot, and the physical bound cancels it again, so it only matters that the
# subtracted value is right where it is NOT part of the reference: memory that
# other processes hold shows up in ``cuda_memory`` too, and beyond this cap it is
# kept out of the subtraction, so the physical bound shrinks with it.
DEFAULT_CONTEXT_CAP_MIB = 1280
# Headroom for the warm-up prompt below the window: the decode step after the
# prefill schedules 1 + k more tokens (k <= 7) and the block table row must hold them.
WARMUP_WINDOW_MARGIN = 16


@dataclass(frozen=True)
class LaneReference:
    """Measured steady peak of one lane at one utilisation (nvidia-smi MiB)."""

    name: str
    peak_mib: int
    util: float
    source: str
    spec_tokens: int = 0  # num_speculative_tokens of the measured run
    max_num_seqs: int = 0  # --max-num-seqs of the measured run (decides the graph set)


# Both rows are the highest ``memory.used`` of any GPU over a whole run on 4x
# V100-SXM2-32GB, TP4, Swift-1.5 Qwen3.8-Flash-Next NVFP4, including the four
# concurrent 8K prefills, from the 2026-10-03 runs of the 1003 code
# (sx_bench/results/2026-10-03-fork-vs-official-v1.5.1/, REPORT_TABLES.md
# section 5): MTP = arm FM, no-MTP = arm F1. The 2026-10-01 runs (1001 code)
# gave 31811 MiB (MTP) and 31839 MiB (no MTP); the MTP ports of 1003 lowered the
# MTP peak. Re-measure the baseline rows with sx_tests/kv-steady-budget/run_on_v100.sh
# before trusting them for a new build.
LANE_REFERENCES: dict[str, LaneReference] = {
    "mtp": LaneReference(
        "mtp",
        31467,
        0.87,
        "arm FM (1003 code): fork + native MTP k=4, max-num-seqs 16, util 0.87, "
        "peak 31467 MiB over 4x8K, long prompts and the answers check",
        spec_tokens=4,
        max_num_seqs=16,
    ),
    "nomtp": LaneReference(
        "nomtp",
        31839,
        0.90,
        "F1_sweep_c8/c16/c24.json: fork production no-MTP, max-num-seqs 24, "
        "util 0.90, idle 31681 MiB, peak 31839 MiB (arm F1 of the 1003 code: "
        "31831 MiB)",
        spec_tokens=0,
        max_num_seqs=24,
    ),
}


def _parse_flag(value: str | None) -> bool:
    return value is not None and value.strip() == "1"


def steady_budget_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True only for ``SX_OPT_KV_STEADY_BUDGET=1``; unset or ``0`` keeps today's
    behaviour byte for byte."""
    env = os.environ if environ is None else environ
    return _parse_flag(env.get(ENV_SWITCH))


def strict_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return _parse_flag(env.get(ENV_STRICT))


def empty_cache_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Release the capture streams' idle allocator blocks after the graph
    capture (default on under the switch; ``SX_OPT_KV_STEADY_EMPTY_CACHE=0``
    keeps them, e.g. if a graph-referenced buffer is not kept alive in Python:
    upstream ac1e67685 fixed such a case that this fork's base lacks)."""
    env = os.environ if environ is None else environ
    return env.get(ENV_EMPTY_CACHE, "1").strip() != "0"


def read_mib(
    name: str,
    default: int | None,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """A non-negative number setting (MiB, or tokens for the warm-up knob);
    invalid or negative values raise ValueError: a silently ignored memory knob
    is worse than a refused start."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {raw!r}")
    return int(value)


@dataclass(frozen=True)
class BudgetInputs:
    """Everything ``plan_kv_budget`` needs, in bytes unless stated."""

    total_memory: int  # init_snapshot.total_memory (what CUDA can hand out)
    requested_memory: int  # util * total_memory
    util: float
    init_cuda_memory: int  # init_snapshot.cuda_memory: context + NCCL (+ foreign)
    free_after_profile: int  # device free right after the measured profile
    non_kv_cache_memory: int  # weights + non-torch + activation peak + residue
    activation_peak: int  # torch peak increase of the measured profile
    graph_reserve: int  # upstream-style graph pool reserve (util bound)
    lane: str  # "mtp" / "nomtp" (admitted Qwen3.8 TP4 lanes) / "other"
    spec_tokens: int = 0  # num_speculative_tokens of this run
    max_num_seqs: int = 0  # --max-num-seqs of this run


@dataclass(frozen=True)
class BudgetPlan:
    kv_unreserved: int  # requested - non_kv: what the sizing gave without the switch
    kv_utilisation: int  # requested - non_kv - graph_reserve
    kv_physical: int  # free_after_profile - activation_peak - P - H
    kv_bytes: int  # min of the two
    limiting: str  # "utilisation" or "physical"
    post_sizing: int  # P
    post_sizing_source: str  # "env" / "lane reference ..." / "unreferenced lane"
    headroom: int  # H
    graph_reserve: int
    predicted_free_at_peak: int  # free memory expected at the steady peak
    util_to_fill_physical: float  # smallest util that no longer limits the KV
    lines: tuple[str, ...]  # log lines, one per fact

    @property
    def kv_gib(self) -> float:
        return self.kv_bytes / GiB


def estimate_post_sizing(
    inputs: BudgetInputs,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Post-sizing growth ``P`` in bytes and where the number came from.

    ``P`` excludes the activation peak (the physical bound subtracts that
    separately) and excludes the CUDA context/NCCL footprint (it is part of
    ``free_after_profile``). For a lane with a reference::

        P = reference_peak - reference_util * total - min(init_cuda_memory, cap)

    which is the reference run's own overshoot over its utilisation budget with
    this device's own context taken out (``DEFAULT_CONTEXT_CAP_MIB`` keeps memory
    held by other processes out of it). Without a reference it is the graph
    reserve plus ``DEFAULT_UNREFERENCED_EXTRA_MIB``.
    """
    explicit = read_mib(ENV_RESERVE_MIB, None, environ)
    if explicit is not None:
        return explicit * MiB, f"{ENV_RESERVE_MIB}={explicit} MiB"
    ref = LANE_REFERENCES.get(inputs.lane)
    if ref is None:
        extra = read_mib(ENV_EXTRA_MIB, DEFAULT_UNREFERENCED_EXTRA_MIB, environ)
        assert extra is not None
        return (
            inputs.graph_reserve + extra * MiB,
            f"unreferenced lane {inputs.lane!r}: graph reserve + {extra} MiB",
        )
    overshoot = ref.peak_mib * MiB - int(ref.util * inputs.total_memory)
    cap_mib = read_mib(ENV_CONTEXT_CAP_MIB, DEFAULT_CONTEXT_CAP_MIB, environ)
    assert cap_mib is not None
    context = min(inputs.init_cuda_memory, cap_mib * MiB)
    value = max(overshoot - context, inputs.graph_reserve)
    clipped = ""
    if context < inputs.init_cuda_memory:
        clipped = (
            f", of {inputs.init_cuda_memory / MiB:.0f} MiB in use at init "
            f"(the rest is not this engine's)"
        )
    return value, (
        f"lane reference {ref.name}: peak {ref.peak_mib} MiB at util {ref.util} "
        f"minus its budget minus this device's {context / MiB:.0f} MiB "
        f"context/NCCL{clipped} ({ref.source})"
    )


def reference_shape_note(inputs: BudgetInputs) -> str | None:
    """A warning when this run's shape differs from the one the lane reference
    was measured at (the graph set, hence P, depends on both)."""
    ref = LANE_REFERENCES.get(inputs.lane)
    if ref is None or (not inputs.spec_tokens and not inputs.max_num_seqs):
        return None
    if inputs.spec_tokens == ref.spec_tokens and (
        not inputs.max_num_seqs or inputs.max_num_seqs == ref.max_num_seqs
    ):
        return None
    return (
        "KV steady budget: NOTE the %s reference was measured at k=%d, "
        "max-num-seqs %d; this run has k=%d, max-num-seqs %d, so the graph set "
        "and P differ. Check the audit after the warm-up and set %s from it."
        % (
            ref.name,
            ref.spec_tokens,
            ref.max_num_seqs,
            inputs.spec_tokens,
            inputs.max_num_seqs,
            ENV_RESERVE_MIB,
        )
    )


def plan_kv_budget(
    inputs: BudgetInputs,
    environ: Mapping[str, str] | None = None,
) -> BudgetPlan:
    """KV bytes for one rank: ``min(utilisation bound, physical bound)``."""
    headroom_mib = read_mib(ENV_HEADROOM_MIB, DEFAULT_HEADROOM_MIB, environ)
    assert headroom_mib is not None
    headroom = headroom_mib * MiB
    post_sizing, source = estimate_post_sizing(inputs, environ)

    kv_unreserved = inputs.requested_memory - inputs.non_kv_cache_memory
    kv_util = kv_unreserved - inputs.graph_reserve
    kv_phys = (
        inputs.free_after_profile - inputs.activation_peak - post_sizing - headroom
    )
    if kv_phys < kv_util:
        kv, limiting = kv_phys, "physical"
    else:
        kv, limiting = kv_util, "utilisation"

    # Free memory at the steady peak for the chosen KV size (physical identity:
    # what is free after the profile, minus KV, activations, post-sizing).
    predicted_free = (
        inputs.free_after_profile - kv - inputs.activation_peak - post_sizing
    )
    # Utilisation at which the utilisation bound equals the physical bound.
    util_fill = (
        kv_phys
        + inputs.non_kv_cache_memory
        + inputs.graph_reserve
    ) / max(inputs.total_memory, 1)

    note = reference_shape_note(inputs)
    lines = (
        "KV steady budget: util %.4f of %.2f GiB = %.2f GiB requested; non-KV "
        "%.2f GiB (activation peak %.2f GiB), graph reserve %.2f GiB -> "
        "utilisation bound %.2f GiB (the sizing without this switch gives %.2f "
        "GiB at this utilisation)"
        % (
            inputs.util,
            inputs.total_memory / GiB,
            inputs.requested_memory / GiB,
            inputs.non_kv_cache_memory / GiB,
            inputs.activation_peak / GiB,
            inputs.graph_reserve / GiB,
            kv_util / GiB,
            kv_unreserved / GiB,
        ),
        "KV steady budget: free after profile %.2f GiB (includes %.2f GiB "
        "context/NCCL), post-sizing allocation P %.2f GiB [%s], headroom %.2f "
        "GiB -> physical bound %.2f GiB"
        % (
            inputs.free_after_profile / GiB,
            inputs.init_cuda_memory / GiB,
            post_sizing / GiB,
            source,
            headroom / GiB,
            kv_phys / GiB,
        ),
        "KV steady budget: KV cache %.2f GiB (%s bound); predicted free memory "
        "at the steady peak %.0f MiB; the utilisation bound stops limiting at "
        "--gpu-memory-utilization %.4f"
        % (kv / GiB, limiting, predicted_free / MiB, util_fill),
        # One machine-readable line (bytes) for sx_tests/kv-steady-budget/analyze.py.
        "KV steady budget [kv=%d total=%d requested=%d free_after_profile=%d "
        "activation=%d graph_reserve=%d post_sizing=%d headroom=%d limiting=%s "
        "kv_utilisation=%d kv_physical=%d init_cuda=%d]"
        % (
            kv,
            inputs.total_memory,
            inputs.requested_memory,
            inputs.free_after_profile,
            inputs.activation_peak,
            inputs.graph_reserve,
            post_sizing,
            headroom,
            limiting,
            kv_util,
            kv_phys,
            inputs.init_cuda_memory,
        ),
    ) + ((note,) if note else ())
    return BudgetPlan(
        kv_unreserved=kv_unreserved,
        kv_utilisation=kv_util,
        kv_physical=kv_phys,
        kv_bytes=kv,
        limiting=limiting,
        post_sizing=post_sizing,
        post_sizing_source=source,
        headroom=headroom,
        graph_reserve=inputs.graph_reserve,
        predicted_free_at_peak=predicted_free,
        util_to_fill_physical=util_fill,
        lines=lines,
    )


@dataclass(frozen=True)
class AuditInputs:
    """Memory measured after the warm-up, in bytes."""

    free_after_profile: int  # F_prof, as in BudgetInputs
    kv_bytes_allocated: int  # sum of the KV tensor sizes actually allocated
    free_after_kv: int  # device free right after initialize_kv_cache
    free_at_end: int  # device free at the end of compile_or_warm_up_model
    cached_free: int  # torch reserved - allocated at that point
    activation_peak: int
    graph_capture_bytes: int  # what capture_model measured
    post_sizing_planned: int  # P the plan assumed
    headroom_planned: int
    load_margin: int


@dataclass(frozen=True)
class AuditResult:
    kv_init_extra: int  # non-KV memory taken by initialize_kv_cache
    warmup_growth: int  # taken by compile_or_warm_up_model
    post_sizing_measured: int  # measured P (activation re-cache excluded)
    activation_deficit: int  # activation peak not yet held by the allocator cache
    projected_min_free: int  # free memory expected at the load peak
    ok: bool
    surplus: int  # projected_min_free - headroom (negative = short)
    suggested_reserve_mib: int  # RESERVE value that makes the plan exact
    lines: tuple[str, ...]


def audit(a: AuditInputs) -> AuditResult:
    """Compare the measured post-sizing growth with what the plan assumed.

    ``cached_free`` (torch reserved minus allocated) is memory the allocator
    already holds for the next activation peak, so only the part of the peak
    that is not cached yet still has to be found at serving time.
    """
    kv_init_extra = max(
        0, (a.free_after_profile - a.free_after_kv) - a.kv_bytes_allocated
    )
    warmup_growth = max(0, a.free_after_kv - a.free_at_end)
    cached_activation = min(a.cached_free, a.activation_peak)
    measured = kv_init_extra + warmup_growth - cached_activation
    deficit = max(0, a.activation_peak - a.cached_free)
    projected_min_free = a.free_at_end - deficit - a.load_margin
    surplus = projected_min_free - a.headroom_planned
    ok = surplus >= 0
    # The RESERVE (P) that would have made predicted free == headroom exactly:
    # P' = P_measured_with_load = measured + load margin.
    suggested = max(0, (measured + a.load_margin) // MiB)
    lines = [
        "KV steady audit: after the KV cache the device had %.0f MiB free "
        "(KV %.0f MiB, initialize_kv_cache non-KV %.0f MiB); the warm-up then "
        "took %.0f MiB (CUDA graphs %.0f MiB, allocator cache that holds the "
        "activation peak %.0f MiB); %.0f MiB free at the end"
        % (
            a.free_after_kv / MiB,
            a.kv_bytes_allocated / MiB,
            kv_init_extra / MiB,
            warmup_growth / MiB,
            a.graph_capture_bytes / MiB,
            cached_activation / MiB,
            a.free_at_end / MiB,
        ),
        "KV steady audit: measured post-sizing growth %.0f MiB (plan assumed "
        "%.0f MiB); +%.0f MiB load margin and %.0f MiB of the activation peak "
        "not cached yet -> %.0f MiB free expected at the load peak (headroom "
        "%.0f MiB)"
        % (
            measured / MiB,
            a.post_sizing_planned / MiB,
            a.load_margin / MiB,
            deficit / MiB,
            projected_min_free / MiB,
            a.headroom_planned / MiB,
        ),
    ]
    if ok:
        lines.append(
            "KV steady audit: OK, %.0f MiB above the headroom. Setting %s=%d "
            "would size the KV cache exactly to the measurement."
            % (surplus / MiB, ENV_RESERVE_MIB, suggested)
        )
    else:
        lines.append(
            "KV steady audit: SHORT by %.0f MiB; the steady peak will leave "
            "%.0f MiB free, below the %.0f MiB headroom. Set %s=%d (or lower "
            "--gpu-memory-utilization)."
            % (
                -surplus / MiB,
                projected_min_free / MiB,
                a.headroom_planned / MiB,
                ENV_RESERVE_MIB,
                suggested,
            )
        )
    return AuditResult(
        kv_init_extra=kv_init_extra,
        warmup_growth=warmup_growth,
        post_sizing_measured=measured,
        activation_deficit=deficit,
        projected_min_free=projected_min_free,
        ok=ok,
        surplus=surplus,
        suggested_reserve_mib=suggested,
        lines=tuple(lines),
    )


def steady_warmup_tokens(
    max_num_batched_tokens: int,
    max_model_len: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Prompt length of the extra large-shape warm-up (0 = none).

    The warm-up of ``warmup_kernels`` runs 6-48 token prompts, so the
    workspaces of a full prefill chunk (indexer score buffers, grouped page4
    workspace, GDN chunk temporaries, PLE short-conv rows) are first allocated
    by the first long real request, after the KV cache is sized and outside any
    audit. With the switch on, one full-chunk prefill runs at startup so those
    buffers exist (and are measured) before serving. Unset: a full chunk (kept
    ``WARMUP_WINDOW_MARGIN`` tokens inside the window so the decode step after it
    fits), ``0`` disables it.
    """
    cap = min(int(max_num_batched_tokens), int(max_model_len) - WARMUP_WINDOW_MARGIN)
    value = read_mib(ENV_WARMUP_TOKENS, None, environ)
    if value is None:
        return max(cap, 0)
    return min(int(value), cap)


def steady_warmup_fits(num_blocks: int, blocks_per_request: int) -> bool:
    """One warm-up request needs ``blocks_per_request`` KV blocks out of
    ``num_blocks`` (block 0 is the null block); a cache too small for it skips
    the large warm-up instead of indexing past the KV tensors."""
    return int(num_blocks) - 1 >= int(blocks_per_request)
