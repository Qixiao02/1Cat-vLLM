# SPDX-License-Identifier: Apache-2.0
"""Four-GPU worker for the b2-allreduce tests ([C4] SX_OPT_PUSH_AR_WIDE).

Needs FOUR idle, fully NVLink-connected SM70 GPUs (one V100 quad, e.g. 0-3):
stop the serving container first. Run inside the image with the rebuilt
custom-AR extension (vllm/_C.abi3.so, or the sidecar through
VLLM_SM70_CUSTOM_AR_LIBRARY) and the patched custom_all_reduce.py:

  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/venv/bin/python -m torch.distributed.run \
      --standalone --nproc-per-node=4 sx_tests/b2-allreduce/_push_ar_wide_worker.py \
      --mode {admission|equality|graph|bench} --out /tmp/b2_ar.json [--cycles N]

test_push_ar_wide_tp4.py launches exactly this under pytest.

Modes
- admission: for every arm (new/old/pull/wide_only), op (regular/sum2) and size
  (5..160 KiB in 5-KiB steps, plus 8 KiB, 10.5 KiB, 80 KiB + 16 B and 320 KiB)
  capture ONE collective, replay it, and count the push CTAs from the per-CTA
  epoch words that flipped in the push storage (0 = pull). Must equal the
  Python model of the native admission (_push_ar_common.native_push_ctas), the
  flipped words must be exactly CTAs 0..n-1, and the result must be bitwise
  equal to the rank-ordered FP32 reference.
- equality: one graph per arm (new, wide_only, pull) with the regular and the
  sum2 collective for all 32 row sizes; changing inputs over several patterns.
  Push (new, wide_only) vs pull (eager registered-buffer pull for regular,
  captured pull kernel for sum2) and vs the torch rank-ordered reference:
  every non-NaN FP16 bit pattern equal (incl. +-0, +-inf), NaN positions equal.
  All four ranks bitwise identical (incl. NaN payloads). Canaries intact,
  push storage back to all-sentinel after every cycle, HC region untouched.
- graph: mixed-size stress. Five graphs (forward regular, reverse regular+sum2,
  48 x (AR + sum2) at M24 then M32, a permuted mix with rank-dependent sleeps
  inside the graph, and the wide-only row sweep), replayed in changing order
  with a rank-skew sleep, poisoned outputs, canaries and storage hygiene.
- bench: CUDA-graph microbenchmarks (48 back-to-back collectives per graph,
  per-collective us, max over ranks of the per-rank median): every size and
  op for arms pull/old/new, the 48-layer (AR + sum2) step at M in
  {1,2,4,8,12,16,17,20,24,28,32}, and a CTA sweep at M20/M24/M28.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _push_ar_common as C  # noqa: E402

F16 = torch.float16
MIXED = (
    5120,
    8192,
    10240,
    10752,
    15360,
    20480,
    25600,
    40960,
    61440,
    81920,
    81936,
    87040,
    102400,
    117760,
    122880,
    143360,
    158720,
    163840,
    327680,
)
PROBE_EXTRA = (8192, 10752, 81936, 327680)
SPECIAL_BITS = (
    0,  # +0
    -32768,  # -0
    1,  # smallest +subnormal
    -32767,  # smallest -subnormal
    15360,  # 1.0
    -17408,  # -1.0
    31743,  # 65504
    -1025,  # -65504
    31744,  # +inf
    -1024,  # -inf
    32639,  # 0x7f7f: the push sentinel (a NaN payload)
    32256,  # 0x7e00: canonical NaN
    -512,  # 0xfe00: negative NaN
    13824,  # 0.5
    1023,  # largest subnormal
    -31745,  # 0x83ff
)


class _Cai:
    """Minimal __cuda_array_interface__ wrapper over a raw device pointer."""

    def __init__(self, ptr: int, numel: int, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": (numel,),
            "typestr": typestr,
            "data": (int(ptr), False),
            "strides": None,
            "version": 2,
        }


def _raw_view(ptr: int, numel: int, typestr: str, dtype: torch.dtype):
    tensor = torch.as_tensor(
        _Cai(ptr, numel, typestr),
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if (
        tensor.dtype != dtype
        or tensor.data_ptr() != int(ptr)
        or tensor.numel() != numel
    ):
        raise RuntimeError("raw device view of the push storage failed")
    return tensor


class Ctx:
    def __init__(self):
        from vllm import _custom_ops as ops
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["WORLD_SIZE"]) != 4:
            raise RuntimeError("requires exactly four ranks")
        torch.cuda.set_device(self.local_rank)
        if torch.cuda.get_device_capability() != (7, 0):
            raise RuntimeError("requires SM70 (V100) GPUs")
        for name in ("VLLM_CUSTOM_ALLREDUCE_ALGO", "VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT"):
            if os.environ.get(name):
                raise RuntimeError(
                    f"unset {name}: the pull reference must be the default "
                    "one-stage kernel"
                )
        os.environ["VLLM_SM70_TP4_PUSH_ALLREDUCE"] = "1"
        dist.init_process_group("nccl")
        self.gloo = dist.new_group(backend="gloo")
        self.ca = CustomAllreduce(self.gloo, self.local_rank, max_size=1024 * 1024)
        if self.ca.disabled or not self.ca.fully_connected:
            raise RuntimeError("custom all-reduce unavailable or not fully connected")
        if self.ca.sm70_tp4_push_buffer_ptrs is None:
            raise RuntimeError("SM70 TP4 push storage was not registered")
        base = int(self.ca.sm70_tp4_push_buffer_ptrs[self.ca.rank])
        total = int(ops.sm70_tp4_push_allreduce_buffer_size())
        if total <= C.GENERIC_BUFFER_BYTES:
            raise RuntimeError(f"unexpected push storage size {total}")
        self.epochs = _raw_view(base, C.SIGNAL_BYTES // 4, "<i4", torch.int32)
        self.data = _raw_view(
            base + C.SIGNAL_BYTES,
            (C.GENERIC_BUFFER_BYTES - C.SIGNAL_BYTES) // 2,
            "<i2",
            torch.int16,
        )
        self.hc = _raw_view(
            base + C.GENERIC_BUFFER_BYTES,
            total - C.GENERIC_BUFFER_BYTES,
            "|u1",
            torch.uint8,
        )
        torch.cuda.synchronize()
        self.hc_snapshot = self.hc.clone()
        self.library = os.environ.get("VLLM_SM70_CUSTOM_AR_LIBRARY")
        self.failures: list[str] = []
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(20260926 + 7919 * self.rank)

    # -- helpers -----------------------------------------------------------
    def fail(self, message: str) -> None:
        if len(self.failures) < 200:
            self.failures.append(f"rank{self.rank}: {message}")

    def barrier(self):
        dist.barrier(group=self.gloo)

    def capture(self, body):
        torch.cuda.synchronize()
        self.barrier()
        graph = torch.cuda.CUDAGraph()
        with self.ca.capture(), torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize()
        self.barrier()
        return graph

    def call(self, op, a, b, out):
        if op == "plain":
            self.ca.all_reduce(a, out=out, registered=True)
        else:
            self.ca.all_reduce_sum2(a, b, out=out)

    def gather(self, x):
        peers = [torch.empty_like(x) for _ in range(4)]
        dist.all_gather(peers, x.contiguous())
        return peers

    def hygiene(self, where: str) -> None:
        torch.cuda.synchronize()
        self.barrier()
        bad = torch.stack(
            [
                (self.data != C.SENTINEL_I16).sum(),
                (self.epochs[C.PUSH_MAX_BLOCKS :] != 0).sum(),
                (
                    (self.epochs[: C.PUSH_MAX_BLOCKS] != 0)
                    & (self.epochs[: C.PUSH_MAX_BLOCKS] != 1)
                ).sum(),
                (self.hc != self.hc_snapshot).sum(),
            ]
        ).tolist()
        if any(bad):
            self.fail(
                f"{where}: push storage not clean (non-sentinel data={bad[0]}, "
                f"pad epoch words={bad[1]}, bad epochs={bad[2]}, HC bytes={bad[3]})"
            )

    def finish(self, out_path: Path, result: dict) -> None:
        torch.cuda.synchronize()
        gathered: list = [None] * 4
        dist.all_gather_object(gathered, self.failures, group=self.gloo)
        failures = [f for per_rank in gathered for f in per_rank]
        result["failures"] = failures
        result["library"] = self.library
        result["torch"] = torch.__version__
        if self.rank == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2) + "\n")
        if failures:
            raise AssertionError(
                f"{len(failures)} failure(s); first: {failures[:5]}"
            )


@contextmanager
def arm(name: str, **extra):
    env = dict(C.ARMS[name])
    env.update(extra)
    saved = {k: os.environ.get(k) for k in C.CONTROLLED}
    try:
        for key in C.CONTROLLED:
            value = env.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield env
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def guarded(numel: int):
    buffer = torch.full((numel + 16,), -37.0, device="cuda", dtype=F16)
    return buffer, buffer[8:-8]


def canaries_ok(buffer) -> torch.Tensor:
    return (buffer[:8] == -37).all() & (buffer[-8:] == -37).all()


def ref_plain(peers):
    acc = peers[0].float()
    for peer in peers[1:]:
        acc = acc + peer.float()
    return acc.half()


def ref_sum2(peers_a, peers_b):
    acc = (peers_a[0] + peers_b[0]).float()
    for a, b in zip(peers_a[1:], peers_b[1:]):
        acc = acc + (a + b).float()
    return acc.half()


def mismatch(actual, reference) -> torch.Tensor:
    """[non-NaN bit mismatches, NaN-position mismatches, NaN payload diffs]."""
    a_bits = actual.view(torch.int16)
    r_bits = reference.view(torch.int16)
    a_nan = torch.isnan(actual)
    r_nan = torch.isnan(reference)
    differ = a_bits != r_bits
    return torch.stack(
        [
            (differ & ~r_nan & ~a_nan).sum(),
            (a_nan != r_nan).sum(),
            (differ & a_nan & r_nan).sum(),
        ]
    )


def fill(ctx: Ctx, pattern: str, cycle: int, tensors) -> None:
    gen = ctx.gen
    for t in tensors:
        n = t.numel()
        if pattern == "random":
            t.normal_(generator=gen).mul_(0.03)
        elif pattern == "large":
            t.normal_(generator=gen).mul_(4096.0)
        elif pattern == "huge":
            sign = torch.randint(0, 2, (n,), generator=gen, device="cuda") * 2 - 1
            mag = torch.rand((n,), generator=gen, device="cuda") * 45000 + 20000
            t.copy_((sign * mag).half())
        elif pattern == "integers":
            t.copy_(torch.randint(-64, 65, (n,), generator=gen, device="cuda"))
        elif pattern == "signed_zero":
            t.zero_()
            t[(cycle + ctx.rank) % 2 :: 2] = -0.0
        elif pattern == "special":
            bits = torch.tensor(SPECIAL_BITS, device="cuda", dtype=torch.int16)
            index = torch.arange(n, device="cuda") * (ctx.rank + 3) + cycle
            t.copy_(bits[index % bits.numel()].view(F16))
            # Mix in finite random values so most elements stay finite.
            keep = torch.rand((n,), generator=gen, device="cuda") < 0.5
            noise = torch.randn((n,), generator=gen, device="cuda").mul_(0.5).half()
            t.copy_(torch.where(keep, t, noise))
        else:
            raise ValueError(pattern)


# ---------------------------------------------------------------------------
def mode_admission(ctx: Ctx, args) -> dict:
    sizes = tuple(C.ROW_SIZES) + PROBE_EXTRA
    inputs = {}
    for nbytes in sizes:
        a = torch.empty(nbytes // 2, device="cuda", dtype=F16)
        b = torch.empty_like(a)
        a.normal_(generator=ctx.gen).mul_(0.03)
        b.normal_(generator=ctx.gen).mul_(0.03)
        inputs[nbytes] = (a, b)
    refs = {}
    for nbytes, (a, b) in inputs.items():
        pa, pb = ctx.gather(a), ctx.gather(b)
        refs[nbytes] = (ref_plain(pa), ref_sum2(pa, pb))
    table: dict = {}
    for arm_name in ("new", "old", "pull", "wide_only"):
        for op in ("plain", "sum2"):
            for nbytes in sizes:
                a, b = inputs[nbytes]
                buffer, out = guarded(a.numel())
                with arm(arm_name) as env:
                    expected = C.native_push_ctas(op, nbytes, env)
                    graph = ctx.capture(lambda: ctx.call(op, a, b, out))
                out.fill_(float("nan"))
                torch.cuda.synchronize()
                before = ctx.epochs.clone()
                graph.replay()
                torch.cuda.synchronize()
                flipped = ctx.epochs != before
                count = int(flipped.sum())
                prefix = bool(flipped[:count].all()) if count else True
                reference = refs[nbytes][0 if op == "plain" else 1]
                bad = mismatch(out, reference).tolist()
                key = f"{arm_name}/{op}/{nbytes}"
                table[key] = {"expected_ctas": expected, "observed_ctas": count}
                if count != expected or not prefix:
                    ctx.fail(
                        f"admission {key}: observed {count} CTAs "
                        f"(prefix={prefix}), model {expected}"
                    )
                if bad[0] or bad[1] or not bool(canaries_ok(buffer)):
                    ctx.fail(f"admission {key}: result mismatch {bad}")
                del graph
            ctx.hygiene(f"admission {arm_name}/{op}")
    changed = {
        op: [
            n
            for n in C.ROW_SIZES
            if table[f"new/{op}/{n}"]["observed_ctas"]
            != table[f"old/{op}/{n}"]["observed_ctas"]
        ]
        for op in ("plain", "sum2")
    }
    return {"mode": "admission", "table": table, "changed_vs_old": changed}


def mode_equality(ctx: Ctx, args) -> dict:
    sizes = C.ROW_SIZES
    xa = [torch.zeros(n // 2, device="cuda", dtype=F16) for n in sizes]
    xb = [torch.zeros(n // 2, device="cuda", dtype=F16) for n in sizes]
    arms = ("new", "wide_only", "pull")
    outs = {
        (arm_name, op): [guarded(n // 2) for n in sizes]
        for arm_name in arms
        for op in ("plain", "sum2")
    }
    eager = [guarded(n // 2) for n in sizes]
    graphs = {}
    paths = {}
    for arm_name in arms:
        with arm(arm_name) as env:
            paths[arm_name] = {
                op: [C.native_push_ctas(op, n, env) for n in sizes]
                for op in ("plain", "sum2")
            }

            def body(arm_name=arm_name):
                for i in range(len(sizes)):
                    ctx.call("plain", xa[i], None, outs[(arm_name, "plain")][i][1])
                    ctx.call("sum2", xa[i], xb[i], outs[(arm_name, "sum2")][i][1])

            graphs[arm_name] = ctx.capture(body)
    patterns = ("random", "large", "huge", "integers", "signed_zero", "special")
    totals: dict[str, list[int]] = {}
    for pattern in patterns:
        for cycle in range(args.cycles):
            fill(ctx, pattern, cycle, xa + xb)
            for group in list(outs.values()) + [eager]:
                for _, interior in group:
                    interior.fill_(float("nan"))
            ctx.barrier()
            order = list(arms)
            order = order[cycle % 3 :] + order[: cycle % 3]
            for arm_name in order:
                if ctx.rank == cycle % 4:
                    torch.cuda._sleep(20000)
                graphs[arm_name].replay()
            for i in range(len(sizes)):
                ctx.ca.all_reduce(xa[i], out=eager[i][1], registered=False)
            torch.cuda.synchronize()
            rows = []
            keys = []
            canary = []
            for i, nbytes in enumerate(sizes):
                pa, pb = ctx.gather(xa[i]), ctx.gather(xb[i])
                r_plain, r_sum2 = ref_plain(pa), ref_sum2(pa, pb)
                pull_plain = eager[i][1]
                pull_sum2 = outs[("pull", "sum2")][i][1]
                checks = [
                    ("pull_eager_plain~torch", pull_plain, r_plain),
                    ("pull_graph_sum2~torch", pull_sum2, r_sum2),
                    ("pull_graph_plain~pull_eager", outs[("pull", "plain")][i][1], pull_plain),
                ]
                for arm_name in ("new", "wide_only"):
                    p = outs[(arm_name, "plain")][i][1]
                    s = outs[(arm_name, "sum2")][i][1]
                    checks += [
                        (f"{arm_name}_plain~pull", p, pull_plain),
                        (f"{arm_name}_sum2~pull", s, pull_sum2),
                        (f"{arm_name}_plain~torch", p, r_plain),
                        (f"{arm_name}_sum2~torch", s, r_sum2),
                    ]
                for name, actual, reference in checks:
                    rows.append(mismatch(actual, reference))
                    keys.append(f"{name}@{nbytes}")
                # Rank identity, full bit patterns (NaN payloads included).
                for arm_name in ("new", "pull"):
                    for op in ("plain", "sum2"):
                        mine = (
                            eager[i][1]
                            if (arm_name, op) == ("pull", "plain")
                            else outs[(arm_name, op)][i][1]
                        )
                        # NCCL has no int16: gather the FP16 bits (a pure copy)
                        # and compare them as int16 locally.
                        peers = [p.view(torch.int16) for p in ctx.gather(mine)]
                        diff = sum((p != peers[0]).sum() for p in peers[1:])
                        rows.append(torch.stack([diff, diff * 0, diff * 0]))
                        keys.append(f"rank_identity_{arm_name}_{op}@{nbytes}")
                for group in list(outs.values()) + [eager]:
                    canary.append(canaries_ok(group[i][0]))
            values = torch.stack(rows).tolist()
            if not bool(torch.stack(canary).all()):
                ctx.fail(f"equality {pattern}#{cycle}: canary overwritten")
            for key, (bits, nan_pos, nan_payload) in zip(keys, values):
                name = key.split("@")[0]
                total = totals.setdefault(name, [0, 0, 0])
                total[0] += bits
                total[1] += nan_pos
                total[2] += nan_payload
                if bits or nan_pos or (name.startswith("rank_identity") and nan_payload):
                    ctx.fail(
                        f"equality {pattern}#{cycle} {key}: bit mismatches={bits} "
                        f"nan-position mismatches={nan_pos}"
                    )
            ctx.hygiene(f"equality {pattern}#{cycle}")
    return {
        "mode": "equality",
        "cycles_per_pattern": args.cycles,
        "patterns": patterns,
        "sizes": sizes,
        "push_ctas_by_arm": paths,
        "totals_[bits,nan_pos,nan_payload]": totals,
    }


def mode_graph(ctx: Ctx, args) -> dict:
    sizes = sorted(set(MIXED) | set(C.ROW_SIZES))
    xa = {n: torch.zeros(n // 2, device="cuda", dtype=F16) for n in sizes}
    xb = {n: torch.zeros(n // 2, device="cuda", dtype=F16) for n in sizes}
    entries: list[tuple[str, int, tuple]] = []  # (op, nbytes, guarded)
    graphs: list[tuple[str, torch.cuda.CUDAGraph]] = []

    def add(op, nbytes):
        g = guarded(nbytes // 2)
        entries.append((op, nbytes, g))
        ctx.call(op, xa[nbytes], xb[nbytes], g[1])

    with arm("new"):
        graphs.append(("forward_plain", ctx.capture(lambda: [add("plain", n) for n in MIXED])))

        def reverse_mixed():
            for n in reversed(MIXED):
                add("plain", n)
                add("sum2", n)

        graphs.append(("reverse_mixed", ctx.capture(reverse_mixed)))

        def layers():
            for rows in (24, 32):
                for _ in range(48):
                    add("plain", rows * C.ROW_BYTES)
                    add("sum2", rows * C.ROW_BYTES)

        graphs.append(("layers_m24_m32", ctx.capture(layers)))
        shuffled = [(op, n) for n in MIXED for op in ("plain", "sum2")]
        random.Random(1234).shuffle(shuffled)
        shuffled.append(("plain", 24 * C.ROW_BYTES))  # odd count

        def permuted_skew():
            for j, (op, n) in enumerate(shuffled):
                if (j + ctx.rank) % 5 == 0:
                    torch.cuda._sleep(4000 + 3000 * ctx.rank)
                add(op, n)

        graphs.append(("permuted_skew", ctx.capture(permuted_skew)))
    with arm("wide_only"):

        def wide_rows():
            for n in C.ROW_SIZES:
                add("plain", n)
                add("sum2", n)

        graphs.append(("wide_only_rows", ctx.capture(wide_rows)))
    patterns = ("random", "large", "signed_zero", "special")
    checked = 0
    for pattern in patterns:
        for cycle in range(args.cycles):
            fill(ctx, pattern, cycle, list(xa.values()) + list(xb.values()))
            for _, _, (_, interior) in entries:
                interior.fill_(float("nan"))
            ctx.barrier()
            order = list(range(len(graphs)))
            random.Random(cycle * 31 + len(pattern)).shuffle(order)
            if cycle % 3 == 0:
                order.append(order[-1])  # same graph back to back
            if ctx.rank == cycle % 4:
                torch.cuda._sleep(20000)
            for index in order:
                graphs[index][1].replay()
            torch.cuda.synchronize()
            refs = {}
            for n in sizes:
                pa, pb = ctx.gather(xa[n]), ctx.gather(xb[n])
                refs[n] = (ref_plain(pa), ref_sum2(pa, pb))
            rows = [
                mismatch(interior, refs[n][0 if op == "plain" else 1])
                for op, n, (_, interior) in entries
            ]
            canary = torch.stack([canaries_ok(buffer) for _, _, (buffer, _) in entries])
            values = torch.stack(rows).tolist()
            checked += len(rows)
            if not bool(canary.all()):
                ctx.fail(f"graph {pattern}#{cycle}: canary overwritten")
            for (op, n, _), (bits, nan_pos, _) in zip(entries, values):
                if bits or nan_pos:
                    ctx.fail(
                        f"graph {pattern}#{cycle} {op}@{n}: bit mismatches={bits} "
                        f"nan-position mismatches={nan_pos}"
                    )
            ctx.hygiene(f"graph {pattern}#{cycle}")
    return {
        "mode": "graph",
        "graphs": [name for name, _ in graphs],
        "collectives_per_cycle": len(entries),
        "cycles_per_pattern": args.cycles,
        "patterns": patterns,
        "outputs_checked": checked,
    }


def _time_graph(ctx: Ctx, graph, collectives: int, inner: int, reps: int):
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    ctx.barrier()
    samples = []
    for _ in range(reps):
        ctx.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (inner * collectives))
    median = statistics.median(samples)
    per_rank: list = [None] * 4
    dist.all_gather_object(per_rank, median, group=ctx.gloo)
    return round(max(per_rank), 3)


def mode_bench(ctx: Ctx, args) -> dict:
    inner, reps, calls = args.inner, args.reps, 48
    sizes = tuple(C.ROW_SIZES) + (C.M32_5120,)
    xa = {n: torch.randn(n // 2, device="cuda", generator=ctx.gen).mul_(0.03).half() for n in sizes}
    xb = {n: torch.randn(n // 2, device="cuda", generator=ctx.gen).mul_(0.03).half() for n in sizes}
    outs = {n: torch.empty(n // 2, device="cuda", dtype=F16) for n in sizes}

    def graph_of(env_arm, ops_and_sizes, **extra):
        with arm(env_arm, **extra):
            return ctx.capture(
                lambda: [
                    ctx.call(op, xa[n], xb[n], outs[n]) for op, n in ops_and_sizes
                ]
            )

    per_size: dict = {}
    for op in ("plain", "sum2"):
        for n in sizes:
            if op == "sum2" and n == C.M32_5120:
                continue
            row: dict = {}
            timed: dict[int, float] = {}
            for arm_name in ("pull", "old", "new"):
                ctas = C.native_push_ctas(op, n, C.ARMS[arm_name])
                if arm_name == "pull" and ctas:
                    row["pull_us"] = None  # hard-wired push size
                    continue
                if ctas not in timed:
                    graph = graph_of(arm_name, [(op, n)] * calls)
                    timed[ctas] = _time_graph(ctx, graph, calls, inner, reps)
                    del graph
                row[f"{arm_name}_ctas"] = ctas
                row[f"{arm_name}_us"] = timed[ctas]
            per_size[f"{op}/{n}"] = row
    layers: dict = {}
    for m in (1, 2, 4, 8, 12, 16, 17, 20, 24, 28, 32):
        n = m * C.ROW_BYTES
        row = {}
        for arm_name in ("pull", "old", "new"):
            # CTAs per (regular, sum2); 0 = pull. The "pull" arm still pushes
            # the hard-wired 5-KiB (M1) and 80-KiB (M16) regular collectives.
            row[f"{arm_name}_ctas"] = [
                C.native_push_ctas(op, n, C.ARMS[arm_name]) for op in ("plain", "sum2")
            ]
            graph = graph_of(arm_name, [("plain", n), ("sum2", n)] * calls)
            row[f"{arm_name}_ms_per_48_layers"] = round(
                _time_graph(ctx, graph, 2 * calls, inner, reps) * 2 * calls / 1000.0, 4
            )
            del graph
        layers[f"M{m}"] = row
    sweep: dict = {}
    for m in (20, 24, 28):
        n = m * C.ROW_BYTES
        minimum = C.covering_ctas(n)
        for op in ("plain", "sum2"):
            for ctas in sorted({minimum, max(minimum, 64), C.PUSH_MAX_BLOCKS}):
                extra = {C.WIDE_BLOCKS: str(ctas)}
                env = dict(C.ARMS["new"], **extra)
                if C.native_push_ctas(op, n, env) != ctas:
                    ctx.fail(f"sweep model mismatch {op}@{n} ctas={ctas}")
                    continue
                graph = graph_of("new", [(op, n)] * calls, **extra)
                sweep[f"{op}/M{m}/{ctas}ctas"] = _time_graph(ctx, graph, calls, inner, reps)
                del graph
    ctx.hygiene("bench")
    if args.strict:
        for key, row in per_size.items():
            if row.get("old_ctas") == 0 and row.get("new_ctas"):
                if row["new_us"] >= row["old_us"]:
                    ctx.fail(f"bench {key}: push {row['new_us']} us not faster than pull {row['old_us']} us")
    if ctx.rank == 0:
        print("per-collective us (max over ranks), 48 back-to-back in one graph")
        print(f"{'op/bytes':>14} {'pull':>8} {'old':>8} {'new':>8}  ctas old->new")
        for key, row in per_size.items():
            print(
                f"{key:>14} {str(row.get('pull_us')):>8} {row['old_us']:>8} "
                f"{row['new_us']:>8}  {row['old_ctas']}->{row['new_ctas']}"
            )
        print("48 x (AR + sum2) per decode step, ms")
        for key, row in layers.items():
            print(f"{key:>4} {row}")
        print("CTA sweep (us per collective)", sweep, flush=True)
    return {
        "mode": "bench",
        "inner": inner,
        "reps": reps,
        "calls_per_graph": calls,
        "per_collective_us": per_size,
        "layer_step": layers,
        "cta_sweep_us": sweep,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("admission", "equality", "graph", "bench")
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    ctx = Ctx()
    try:
        result = {
            "admission": mode_admission,
            "equality": mode_equality,
            "graph": mode_graph,
            "bench": mode_bench,
        }[args.mode](ctx, args)
        ctx.finish(args.out, result)
        if ctx.rank == 0:
            print(f"PASS b2-allreduce {args.mode}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        ctx.ca.close()
        dist.destroy_process_group(ctx.gloo)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
