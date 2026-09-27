# SPDX-License-Identifier: Apache-2.0
"""Four-GPU worker: 25-KiB MTP5 push all-reduce gating in the MTP lane.

Needs FOUR idle, fully NVLink-connected SM70 GPUs (stop the serving
container first). Reuses the b2-allreduce harness (sx_tests/b2-allreduce:
push-storage epoch probe, rank-ordered FP32 references, arms):

  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/venv/bin/python -m torch.distributed.run \
      --standalone --nproc-per-node=4 sx_tests/b3-moe-verify/_mtp5_push_worker.py \
      --out /tmp/b3_mtp5.json [--cycles N]

Asserts on every rank:
  * constructing CustomAllreduce while the current vLLM config is the k = 4
    MTP lane (VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5 unset) sets it to "1";
  * admission of the [5, 2560] FP16 payload (25 KiB), regular and sum2:
    production arm (wide on) 13 CTAs either way; with SX_OPT_PUSH_AR_WIDE=0
    the lane arm (MTP5=1) pushes the sum2 collective on 13 CTAs while the
    dev2 arm (MTP5=0) pulls; every result bitwise equal to the rank-ordered
    FP32 reference;
  * a captured graph of 8 x (regular + sum2) over 20/25/30/40/25 KiB payloads
    in the lane arm with the wide admission off replays bitwise equal to the
    references over changing inputs, push storage clean afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "b2-allreduce"))

import _mtp_common as M  # noqa: E402  (keeps import vllm on the installed package)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import _push_ar_common as C  # noqa: E402
import _push_ar_wide_worker as W  # noqa: E402

MTP5_BYTES = 5 * 2560 * 2
STRESS_SIZES = (20480, MTP5_BYTES, 30720, 40960, MTP5_BYTES)


def _lane_ctx() -> W.Ctx:
    import vllm.config.vllm as vcfg

    os.environ.pop(C.MTP5_ENV, None)
    os.environ.pop("SX_OPT_MTP_MOE_ROUTES", None)
    os.environ.pop("SX_OPT_MTP_LANE", None)
    original = vcfg.get_current_vllm_config_or_none
    vcfg.get_current_vllm_config_or_none = lambda: M.lane_config(4)
    try:
        ctx = W.Ctx()
    finally:
        vcfg.get_current_vllm_config_or_none = original
    if os.environ.get(C.MTP5_ENV) != "1":
        ctx.fail("the k=4 MTP lane did not default VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5")
    return ctx


def admission(ctx: W.Ctx) -> dict:
    a = torch.empty(MTP5_BYTES // 2, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    a.normal_(generator=ctx.gen).mul_(0.03)
    b.normal_(generator=ctx.gen).mul_(0.03)
    pa, pb = ctx.gather(a), ctx.gather(b)
    refs = {"plain": W.ref_plain(pa), "sum2": W.ref_sum2(pa, pb)}
    arms = {
        "production": ("new", {C.MTP5_ENV: "1"}),
        "lane_wide_off": ("old", {C.MTP5_ENV: "1"}),
        "dev2_wide_off": ("old", {C.MTP5_ENV: "0"}),
    }
    expected = {
        ("production", "plain"): 13, ("production", "sum2"): 13,
        ("lane_wide_off", "plain"): 13, ("lane_wide_off", "sum2"): 13,
        ("dev2_wide_off", "plain"): 13, ("dev2_wide_off", "sum2"): 0,
    }
    table = {}
    for label, (arm_name, extra) in arms.items():
        for op in ("plain", "sum2"):
            buffer, out = W.guarded(a.numel())
            with W.arm(arm_name, **extra) as env:
                model = C.native_push_ctas(op, MTP5_BYTES, env)
                graph = ctx.capture(lambda op=op, out=out: ctx.call(op, a, b, out))
            out.fill_(float("nan"))
            torch.cuda.synchronize()
            before = ctx.epochs.clone()
            graph.replay()
            torch.cuda.synchronize()
            flipped = ctx.epochs != before
            count = int(flipped.sum())
            bad = W.mismatch(out, refs[op]).tolist()
            table[f"{label}/{op}"] = count
            if count != expected[(label, op)] or count != model:
                ctx.fail(f"{label}/{op}: {count} CTAs, expected "
                         f"{expected[(label, op)]}, model {model}")
            if bad[0] or bad[1] or not bool(W.canaries_ok(buffer)):
                ctx.fail(f"{label}/{op}: result mismatch {bad}")
            del graph
        ctx.hygiene(f"admission {label}")
    return table


def stress(ctx: W.Ctx, cycles: int) -> int:
    xa = [torch.zeros(n // 2, device="cuda", dtype=torch.float16)
          for n in STRESS_SIZES]
    xb = [torch.zeros_like(t) for t in xa]
    outs = [[W.guarded(t.numel()) for t in xa] for _ in range(2)]

    def body():
        for _ in range(8):
            for i in range(len(STRESS_SIZES)):
                ctx.call("plain", xa[i], None, outs[0][i][1])
                ctx.call("sum2", xa[i], xb[i], outs[1][i][1])

    with W.arm("old", **{C.MTP5_ENV: "1"}):
        graph = ctx.capture(body)
    checked = 0
    for cycle in range(cycles):
        W.fill(ctx, ("random", "large", "integers", "signed_zero")[cycle % 4],
               cycle, xa + xb)
        for pair in outs:
            for _, out in pair:
                out.fill_(float("nan"))
        torch.cuda.synchronize()
        ctx.barrier()
        graph.replay()
        torch.cuda.synchronize()
        for i in range(len(STRESS_SIZES)):
            pa, pb = ctx.gather(xa[i]), ctx.gather(xb[i])
            for k, ref in enumerate((W.ref_plain(pa), W.ref_sum2(pa, pb))):
                buffer, out = outs[k][i]
                bad = W.mismatch(out, ref).tolist()
                if bad[0] or bad[1] or not bool(W.canaries_ok(buffer)):
                    ctx.fail(f"stress cycle {cycle} size {STRESS_SIZES[i]} "
                             f"op {k}: {bad}")
                checked += 1
        ctx.hygiene(f"stress cycle {cycle}")
    del graph
    return checked


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=16)
    args = parser.parse_args(argv)
    ctx = _lane_ctx()
    try:
        result = {"mode": "mtp5", "admission": admission(ctx),
                  "outputs_checked": stress(ctx, args.cycles)}
        ctx.finish(args.out, result)
        if ctx.rank == 0:
            print("PASS b3-moe-verify mtp5 push", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        ctx.ca.close()
        dist.destroy_process_group(ctx.gloo)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
