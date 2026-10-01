# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_MTP_HC_BATCH on FOUR GPUs: TP4 batch HC == the replicated MTP chain.

GPU: FOUR peer-connected V100 (TP4), like the deployment. Stop the serving
container first. Needs this tree installed into the imported vllm with the
rebuilt _C / _C_custom_ar (the batch HC op and the enlarged push buffer).

  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/venv/bin/python -m torch.distributed.run \
      --standalone --nproc-per-node=4 sx_tests/mtp-batch/test_mtp_hc_batch_tp4.py \
      [--model /models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4] [--pairs 96] \
      [--out /tmp/mtp_hc_batch_tp4.json]

Without --model it uses 8 random weight pairs (identical on all ranks). Under
plain pytest (no torchrun) the test is skipped.

The reference is what the MTP lane runs today at M5/M10 (the replicated
cuBLAS chain of _qwen38_sm70_fp16_fused_hc's fallback: F.linear down,
qwen4_exp_hc_silu, F.linear up, qwen4_exp_hc_gate_mix) under the MTP
precision policy (allow_fp16_reduced_precision_reduction=True,
allow_fp16_accumulation=False). Asserted on every rank, inside CUDA graphs
with changing inputs at six activation scales, for M5 and M10:
  * the native op in all three schedules (four launches, cooperative,
    cooperative + full unroll): LoRA [M, 320], block [M, 2560] and the
    injection [M, 4] are bitwise equal to the reference;
  * the model dispatcher (_qwen38_sm70_fp16_fused_hc with packed copies,
    inside an installed-lane FULL verify capture) takes the batch route and
    is bitwise equal too;
  * after alternating M5 / M10 graph replays with one extra eager pair in
    between (odd epoch counts on every channel), every path stays exact.
Prints: per-HC-pair CUDA-graph time (max over ranks, median) of the
reference and each schedule (upstream: 33.6 -> 21.1 us at M5, 34.6 -> 23.6
us at M10 for cooperative + full unroll).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

K_HC = 10240
SCALES = (0.0, 0.001, 0.03, 0.1, 1.0, 3.0)
SCHEDULES = {
    "four_launch": (False, False),
    "cooperative": (True, False),
    "cooperative_full_unroll": (True, True),
}


def _load_real_pairs(model: Path, count: int):
    from safetensors import safe_open

    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def get(name: str) -> torch.Tensor:
        with safe_open(model / mapping[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).half()

    pairs = []
    for layer in range(48):
        for role in ("attn", "mlp"):
            if len(pairs) >= count:
                return pairs
            prefix = f"model.language_model.layers.{layer}.{role}_hyper_connection."
            down = torch.zeros((336, K_HC), dtype=torch.float16)
            down[:320].copy_(get(prefix + "input_mix_weight_down.weight"))
            down[320:324].copy_(get(prefix + "block_inject_weight.weight"))
            up = get(prefix + "input_mix_weight_up.weight")
            pairs.append((down.cuda().contiguous(), up.cuda().contiguous()))
    return pairs


def _random_pairs(count: int):
    pairs = []
    for i in range(count):
        gen = torch.Generator(device="cuda").manual_seed(9000 + i)
        down = (torch.randn((336, K_HC), generator=gen, device="cuda") * 0.02).half()
        down[324:].zero_()
        up = (torch.randn((K_HC, 320), generator=gen, device="cuda") * 0.05).half()
        pairs.append((down.contiguous(), up.contiguous()))
    return pairs


def _reference(x, down, up):
    import torch.nn.functional as F

    dai = F.linear(x, down)
    lora = torch.ops.vllm.qwen4_exp_hc_silu(dai[..., :320], 4)
    gate = F.linear(lora, up)
    block = torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, 4)
    return lora, block, dai[..., 320:324]


def _native(comm, x, packed_down, packed_up, cooperative, full_unroll):
    m = x.shape[0]
    partials = torch.empty((20, m, 96), device=x.device, dtype=torch.float32)
    lora, local, block, injection = (x.new_empty((m, n)) for n in (320, 640, 2560, 4))
    comm.sm70_qwen38_hc_batch(
        x,
        packed_down,
        packed_up,
        partials,
        lora,
        local,
        block,
        injection,
        round_down_partials=True,
        cooperative=cooperative,
        full_unroll=full_unroll,
    )
    return lora, block, injection


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(
        torch.count_nonzero(
            a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)
        )
    )


def run(args) -> None:
    import torch.distributed as dist

    import vllm.models.qwen4_exp.nvidia.ops.hc  # noqa: F401
    from vllm.compilation import sm70_decode_graph as dg
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        raise RuntimeError("requires exactly four SM70 GPUs")
    matmul = torch.backends.cuda.matmul
    matmul.allow_fp16_reduced_precision_reduction = True
    matmul.allow_fp16_accumulation = False
    # Defaults only: the multi-row table (SX_OPT_ROWS*) decides whether the
    # dispatcher yields M5/M10 to the multi-row HC kernel.
    for key in (
        "SX_OPT_MTP_HC_BATCH",
        "SX_OPT_MTP_HC_COOPERATIVE",
        "SX_OPT_MTP_HC_FULL_UNROLL",
        "SX_OPT_MTP_BATCH_OVER_ROWS",
        "VLLM_SM70_MTP_HC_BATCH",
        "VLLM_SM70_MTP_HC_COOPERATIVE",
        "VLLM_SM70_MTP_HC_FULL_UNROLL",
        "SX_OPT_ROWS",
        "SX_OPT_ROWS_TABLE",
        "SX_OPT_ROWS_MAX_M",
        "SX_OPT_MTP_ROWS",
    ):
        os.environ.pop(key, None)
    gemv._sx_mtp_batch_config.cache_clear()
    gemv._sx_rows_config.cache_clear()
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=group, device=local_rank, max_size=8 * 1024 * 1024)
    failures: list[str] = []
    timings: dict[int, dict[str, float]] = {}
    batch_hits = [0]
    original_forward = hc._sx_hc_batch_forward

    def counting_forward(*a, **kw):
        out = original_forward(*a, **kw)
        batch_hits[0] += out is not None
        return out

    try:
        pairs = (
            _load_real_pairs(args.model, args.pairs)
            if args.model
            else _random_pairs(args.pairs)
        )
        for down, up in pairs:  # HC weights are replicated on every rank
            dist.broadcast(down, src=0)
            dist.broadcast(up, src=0)
        probe = torch.zeros((5, K_HC), device="cuda", dtype=torch.float16)
        if not comm.can_sm70_qwen38_hc_batch(probe):
            raise RuntimeError("TP4 batch HC op or registered push buffers missing")
        packed = [
            (
                hc._pack_hc_batch_weight(down, "down", comm.rank),
                hc._pack_hc_batch_weight(up, "up", comm.rank),
            )
            for down, up in pairs
        ]
        tp_group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=comm))
        lane_installed = dg.sm70_mtp_lane_installed()
        graphs_by_m = {}

        def check(m, xs, graphs, label) -> int:
            mismatches = 0
            for i in range(len(pairs)):
                ref = graphs["reference"][1][i]
                for kind, (_, outputs) in graphs.items():
                    if kind == "reference":
                        continue
                    out = outputs[i]
                    if kind == "dispatch":  # (block, injection)
                        mismatches += _bits_differ(out[0], ref[1])
                        mismatches += _bits_differ(out[1], ref[2])
                    else:
                        for a, b in zip(out, ref):
                            mismatches += _bits_differ(a, b)
            if mismatches:
                failures.append(
                    f"rank{rank} M={m} {label}: {mismatches} FP16 mismatches"
                )
            return mismatches

        def replay_all(graphs):
            for kind, (graph, outputs) in graphs.items():
                if kind != "reference":
                    for out in outputs:
                        for t in out:
                            t.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()

        for m in args.rows:
            gen = torch.Generator(device="cuda").manual_seed(20260927 + m)
            xs = torch.randn((len(pairs), m, K_HC), generator=gen, device="cuda").half()
            dist.broadcast(xs, src=0)

            def body(kind: str):
                outputs = []
                for i, (down, up) in enumerate(pairs):
                    pd, pu = packed[i]
                    if kind == "reference":
                        outputs.append(_reference(xs[i], down, up))
                    elif kind == "dispatch":
                        outputs.append(
                            hc._qwen38_sm70_fp16_fused_hc(xs[i], down, up, pd, pu)
                        )
                    else:
                        outputs.append(_native(comm, xs[i], pd, pu, *SCHEDULES[kind]))
                return outputs

            def capture(kind: str):
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with (
                    patch(
                        "vllm.distributed.parallel_state.get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(hc, "_sx_hc_batch_forward", counting_forward),
                    dg.sm70_decode_graph_compilation(kind == "dispatch"),
                    comm.capture(),
                    torch.cuda.graph(graph),
                ):
                    dg.set_sm70_mtp_lane_installed(kind == "dispatch")
                    try:
                        outputs = body(kind)
                    finally:
                        dg.set_sm70_mtp_lane_installed(lane_installed)
                torch.cuda.synchronize()
                dist.barrier()
                return graph, outputs

            # Warm cuBLAS / Triton / the native op once outside capture.
            _reference(xs[0], *pairs[0])
            for schedule in SCHEDULES.values():
                _native(comm, xs[0], *packed[0], *schedule)
            torch.cuda.synchronize()
            dist.barrier()
            kinds = ("reference", *SCHEDULES, "dispatch")
            before = batch_hits[0]
            graphs = {kind: capture(kind) for kind in kinds}
            if batch_hits[0] - before != len(pairs):
                failures.append(
                    f"rank{rank} M={m}: dispatcher took the batch route "
                    f"{batch_hits[0] - before}/{len(pairs)} times"
                )
            graphs_by_m[m] = (xs, graphs)
            for scale in SCALES:
                xs.normal_(0.0, 1.0, generator=gen).mul_(scale)
                dist.broadcast(xs, src=0)
                replay_all(graphs)
                check(m, xs, graphs, f"scale {scale}")

            samples = {kind: [] for kind in graphs}
            for kind in graphs:
                for _ in range(10):
                    graphs[kind][0].replay()
            torch.cuda.synchronize()
            dist.barrier()
            for rnd in range(8):
                order = list(graphs) if rnd % 2 == 0 else list(graphs)[::-1]
                for kind in order:
                    for _ in range(4):
                        dist.barrier()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[kind][0].replay()
                        end.record()
                        end.synchronize()
                        all_ms = [None] * 4
                        dist.all_gather_object(
                            all_ms, start.elapsed_time(end), group=group
                        )
                        samples[kind].append(max(all_ms) * 1000 / len(pairs))
            timings[m] = {kind: statistics.median(v) for kind, v in samples.items()}

        # Alternate widths on the SAME graphs; one extra eager pair per step
        # makes every channel's epoch count odd before the next replay.
        transitions = 0
        for step, m in enumerate([5, 10, 5, 5, 10, 10, 5] * 2):
            if m not in graphs_by_m:
                continue
            xs, graphs = graphs_by_m[m]
            xs.normal_(0.0, 0.1 * (1 + step % 3))
            dist.broadcast(xs, src=0)
            replay_all(graphs)
            check(m, xs, graphs, f"transition {step}")
            cooperative, full_unroll = SCHEDULES[list(SCHEDULES)[step % 3]]
            _native(comm, xs[0], *packed[0], cooperative, full_unroll)
            torch.cuda.synchronize()
            dist.barrier()
            transitions += 1

        all_failures = [None] * 4
        dist.all_gather_object(all_failures, failures, group=group)
        result = {
            "pairs": len(pairs),
            "real_weights": bool(args.model),
            "rows": list(args.rows),
            "scales": list(SCALES),
            "transitions": transitions,
            "precision": {
                "allow_fp16_reduced_precision_reduction": True,
                "allow_fp16_accumulation": False,
            },
            "failures": all_failures,
            "us_per_hc_pair_max_rank": timings,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        if rank == 0:
            print(json.dumps(result, indent=2), flush=True)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2) + "\n")
        if any(all_failures):
            raise AssertionError(
                f"TP4 batch HC != replicated MTP chain: {all_failures}"
            )
        if rank == 0:
            print("PASS: TP4 batch HC == replicated MTP chain on all ranks.")
    finally:
        comm.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument(
        "--rows", type=lambda s: [int(v) for v in s.split(",")], default=[5, 10]
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 1 <= args.pairs <= 96 or any(m not in (5, 10) for m in args.rows):
        parser.error("use 1..96 pairs and rows from {5, 10}")
    return args


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "4",
    reason="4-GPU test: launch with torch.distributed.run --nproc-per-node=4",
)
def test_mtp_hc_batch_equals_replicated_chain_tp4():
    run(_parse([]))


if __name__ == "__main__":
    run(_parse())
