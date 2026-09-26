# SPDX-License-Identifier: Apache-2.0
"""dense-multirow MR3 on FOUR GPUs: rows HC == production TP4-sharded M=1 HC.

GPU: FOUR peer-connected V100 (TP4), like the deployment.  Stop the serving
container first.  Needs the overlay installed into the imported vllm and the
image's custom-AR extension (same as benchmarks/kernels/benchmark_sm70_hc_tp4.py).

  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/venv/bin/python -m torch.distributed.run \
      --standalone --nproc-per-node=4 sx_tests/dense-multirow/test_rows_hc_tp4.py \
      [--model /models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4] [--pairs 96]

Without --model it uses 8 random weight pairs (identical on all ranks).  Under
plain pytest (no torchrun) the test is skipped.

Asserts on every rank, inside CUDA graphs with 8 changing inputs:
  for M in {2, 4, 8}, each row of the replicated multi-row HC route
  (_sx_hc_rows_forward, default plan) == the production M=1 route
  (torch.ops.vllm.qwen38_sm70_fp16_fused_hc on that (1, 10240) row with the
  TP4 custom all-reduce: sharded down + push all-gather + fused up/mix/gather
  or whichever sharded variant the loaded extension provides), block and
  injection, bitwise.  This is the C1 == C>1 claim end to end for HC.
Prints: per-HC-call CUDA-graph time (max over ranks, median of 64 replays)
of production M=1 x M loop, rows route and the old cuBLAS chain.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K_HC = 10240


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


def _old_chain(x, down, up):
    import torch.nn.functional as F

    dai = F.linear(x, down)
    lora = torch.ops.vllm.qwen4_exp_hc_silu(dai[..., :320], 4)
    gate = F.linear(lora, up)
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, 4), dai[..., 320:324]


def run(args) -> None:
    import torch.distributed as dist

    import vllm.models.qwen4_exp.nvidia.ops.hc  # noqa: F401
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.get_device_capability() != (
        7,
        0,
    ):
        raise RuntimeError("requires exactly four SM70 GPUs")
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=group, device=local_rank, max_size=8 * 1024 * 1024)
    failures: list[str] = []
    timings: dict[int, dict[str, float]] = {}
    try:
        pairs = (
            _load_real_pairs(args.model, args.pairs)
            if args.model
            else _random_pairs(args.pairs)
        )
        for down, up in pairs:  # HC weights are replicated on every rank
            dist.broadcast(down, src=0)
            dist.broadcast(up, src=0)
        probe = torch.zeros((1, K_HC), device="cuda", dtype=torch.float16)
        if not comm.can_sm70_qwen38_hc_shard(probe):
            raise RuntimeError("production TP4 HC shard route unavailable")
        tp_group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=comm))
        for m in (2, 4, 8):
            plan = hc._sx_hc_rows_plan_for_m(m) or hc._SxHcRowsPlan(
                min(m, 4), 4, min(m, 8), True
            )
            gen = torch.Generator(device="cuda").manual_seed(20260926 + m)
            xs = torch.randn(
                (len(pairs), m, K_HC), generator=gen, device="cuda"
            ).half()

            def capture(kind: str):
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                outputs = []
                with (
                    patch(
                        "vllm.distributed.parallel_state.get_tp_group",
                        return_value=tp_group,
                    ),
                    comm.capture(),
                    torch.cuda.graph(graph),
                ):
                    for i, (down, up) in enumerate(pairs):
                        if kind == "m1":
                            outputs.append(
                                [
                                    torch.ops.vllm.qwen38_sm70_fp16_fused_hc(
                                        xs[i, r : r + 1], down, up
                                    )
                                    for r in range(m)
                                ]
                            )
                        elif kind == "rows":
                            outputs.append(hc._sx_hc_rows_forward(xs[i], down, up, plan))
                        else:
                            outputs.append(_old_chain(xs[i], down, up))
                torch.cuda.synchronize()
                dist.barrier()
                return graph, outputs

            # Warm the Triton/cuBLAS paths once outside capture.
            with patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ):
                hc._sx_hc_rows_forward(xs[0], *pairs[0], plan)
                _old_chain(xs[0], *pairs[0])
            graphs = {kind: capture(kind) for kind in ("m1", "rows", "cublas")}
            mismatches = 0
            for step in range(8):
                xs.normal_(generator=gen)
                if step == 1:
                    xs.mul_(0.01)
                if step == 2:
                    xs.mul_(300.0)
                # The sharded M=1 route assumes replicated activations; make
                # that explicit instead of relying on identical RNG streams.
                dist.broadcast(xs, src=0)
                for kind, (graph, outputs) in graphs.items():
                    for out in outputs:
                        for t in out if kind != "m1" else [z for pair in out for z in pair]:
                            t.fill_(float("nan"))
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                for i in range(len(pairs)):
                    block, injection = graphs["rows"][1][i]
                    for r in range(m):
                        ref_block, ref_inj = graphs["m1"][1][i][r]
                        mismatches += int(
                            torch.count_nonzero(
                                block[r].view(torch.int16) != ref_block[0].view(torch.int16)
                            )
                        )
                        mismatches += int(
                            torch.count_nonzero(
                                injection[r].view(torch.int16)
                                != ref_inj[0].contiguous().view(torch.int16)
                            )
                        )
            if mismatches:
                failures.append(f"rank{rank} M={m}: {mismatches} FP16 mismatches")

            per_kind = {}
            for kind in ("m1", "rows", "cublas"):
                graph = graphs[kind][0]
                for _ in range(20):
                    graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            samples = {kind: [] for kind in graphs}
            for rnd in range(8):
                order = list(graphs) if rnd % 2 == 0 else list(graphs)[::-1]
                for kind in order:
                    graph = graphs[kind][0]
                    for _ in range(8):
                        dist.barrier()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        all_ms = [None] * 4
                        dist.all_gather_object(
                            all_ms, start.elapsed_time(end), group=group
                        )
                        samples[kind].append(max(all_ms) * 1000 / len(pairs))
            for kind, values in samples.items():
                per_kind[kind] = statistics.median(values)
            timings[m] = per_kind
            del graphs
            torch.cuda.synchronize()
            dist.barrier()
        all_failures = [None] * 4
        dist.all_gather_object(all_failures, failures, group=group)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "pairs": len(pairs),
                        "real_weights": bool(args.model),
                        "failures": all_failures,
                        "us_per_hc_call_max_rank": timings,
                    },
                    indent=2,
                ),
                flush=True,
            )
        if any(all_failures):
            raise AssertionError(f"rows HC != production M=1 HC: {all_failures}")
        if rank == 0:
            print("PASS: rows HC rows == production TP4 M=1 HC on all ranks.")
    finally:
        comm.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=8)
    return parser.parse_args(argv)


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "4",
    reason="4-GPU test: launch with torch.distributed.run --nproc-per-node=4",
)
def test_rows_hc_equals_production_tp4_m1():
    run(_parse([]))


if __name__ == "__main__":
    run(_parse())
