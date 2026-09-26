# SPDX-License-Identifier: Apache-2.0
"""b2-allreduce [C4] CPU model check (no GPU) of the two-epoch push protocol.

The GPU stress tests only see the interleavings real hardware happens to
produce. This drives a randomised, rank-skewed interleaving model of
sm70_cross_device_reduce{,_sum2}_1stage_push with launch sequences whose grids
come from the admission model (_push_ar_common.native_push_ctas), i.e. the
covering grids plus idle-CTA and grid-stride launches, and checks that no
rank ever overwrites an unconsumed slot or consumes a foreign packet. A
negative control with undersized (remapping) grids must be caught, so the
model has teeth.

Granularity: one unit = the 128 packs of one CTA pass; in the kernel every
thread of a CTA shares that CTA's epoch word, so cross-call slot reuse is
decided per unit.

  /opt/venv/bin/python -m pytest -q sx_tests/b2-allreduce/test_push_protocol_model_cpu.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _push_ar_common as C  # noqa: E402

RANKS = 4
UNIT_BYTES = C.PACK_BYTES_PER_BLOCK  # 128 packs of 16 B
SIZES = tuple(C.ROW_SIZES) + (8192, 10752, C.M32_5120)


def _units(nbytes: int) -> int:
    return -(-nbytes // UNIT_BYTES)


def _simulate(calls, seed, steps=400_000):
    """calls: list of (units, grid). Returns the first hazard or None."""
    rng = random.Random(seed)
    max_units = max(u for u, _ in calls)
    # slots[rank][epoch][src][unit]
    slots = [
        [[[None] * max_units for _ in range(RANKS)] for _ in range(2)]
        for _ in range(RANKS)
    ]
    epochs = [[0] * C.PUSH_MAX_BLOCKS for _ in range(RANKS)]
    call = [0] * RANKS
    kernels: list = [None] * RANKS

    def launch(r):
        units, grid = calls[call[r]]
        kernels[r] = {
            b: {"epoch": None, "work": [u for u in range(units) if u % grid == b]}
            for b in range(grid)
        }

    for r in range(RANKS):
        launch(r)
    fast = seed % RANKS
    for _ in range(steps):
        live = [r for r in range(RANKS) if kernels[r] is not None]
        if not live:
            return None
        r = fast if fast in live and rng.random() < 0.6 else rng.choice(live)
        kernel = kernels[r]
        if not kernel:
            call[r] += 1
            kernels[r] = None
            if call[r] < len(calls):
                launch(r)
            continue
        b = rng.choice(list(kernel))
        cta = kernel[b]
        if cta["epoch"] is None:
            cta["epoch"] = epochs[r][b]
            cta["state"] = {u: "write" for u in cta["work"]}
            continue
        e = cta["epoch"]
        pending = [u for u, s in cta["state"].items() if s != "done"]
        if not pending:
            epochs[r][b] = 1 - e
            del kernel[b]
            continue
        u = rng.choice(pending)
        k = call[r]
        phase = cta["state"][u]
        if phase == "write":
            for dst in range(RANKS):
                if slots[dst][e][r][u] is not None:
                    return f"call {k} rank {r} overwrote unconsumed slot e={e} u={u}"
                slots[dst][e][r][u] = (k, r)
            cta["state"][u] = "poll"
        elif phase == "poll":
            got = [slots[r][e][src][u] for src in range(RANKS)]
            if all(v is not None for v in got):
                for src, v in enumerate(got):
                    if v != (k, src):
                        return f"call {k} rank {r} consumed foreign {v} u={u}"
                cta["state"][u] = "clear"
        else:
            for src in range(RANKS):
                slots[r][e][src][u] = None
            cta["state"][u] = "done"
    return "did not finish within the step budget"


def _production_sequence(rng):
    env = dict(C.ARMS["new"])
    calls = []
    target = rng.randint(3, 9)
    while len(calls) < target:
        op = rng.choice(("plain", "sum2"))
        nbytes = rng.choice(SIZES)
        if rng.random() < 0.2 and C.wide_bytes(nbytes):
            low = C.covering_ctas(nbytes)
            env_call = dict(env, **{C.WIDE_BLOCKS: str(rng.randint(low, 80))})
        else:
            env_call = env
        grid = C.native_push_ctas(op, nbytes, env_call)
        if grid:  # pull calls never touch the push storage
            calls.append((_units(nbytes), grid))
    return calls


def test_admitted_launch_mixes_are_protocol_safe():
    for trial in range(300):
        rng = random.Random(trial)
        calls = _production_sequence(rng)
        hazard = _simulate(calls, seed=trial)
        assert hazard is None, (calls, hazard)


def test_undersized_grids_are_caught():
    caught = 0
    for trial in range(300):
        rng = random.Random(50_000 + trial)
        calls = []
        for units, grid in _production_sequence(rng):
            if units > 1 and grid >= units and rng.random() < 0.5:
                grid = rng.randint(1, units - 1)  # remaps units to other CTAs
            calls.append((units, grid))
        if _simulate(calls, seed=trial) is not None:
            caught += 1
    assert caught > 0
