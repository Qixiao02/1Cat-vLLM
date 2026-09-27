#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V100 tile sweep for the MTP draft MoE table (SX_OPT_MTP_DRAFT_TILES).

GPU: ONE V100 (SM70), nothing else running on it. Run inside image
1.8.0-dev2 with the b3-draft files bind-mounted over the vllm package:

  /opt/venv/bin/python sx_tests/b3-draft/sweep_mtp_draft_tiles.py \\
      --weights /models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 \\
      --out /tmp/mtp_draft_sweep.json --emit-table /tmp/mtp_draft_tiles.json

Use the result with SX_OPT_MTP_DRAFT_TILES_FILE=/tmp/mtp_draft_tiles.json, or
copy it over vllm/model_executor/layers/fused_moe/configs/
sx_sm70_mtp_draft_tiles_E512_N160_K2560_topk10.json (no code change).

What is timed: one draft MoE call exactly as the drafter runs it -
fused_experts on the TP4-local shape E=512, w13 [512, 320, 2560],
w2 [512, 2560, 160], top-10, FP16 (config selection, naive / sorted expert
assignment, W13 GEMM, SiLU*mul, W2 GEMM, moe_sum) - inside a CUDA graph that
chains ``--routings`` distinct routings (and ``--weight-copies`` weight copies)
so consecutive calls touch different experts; weights are cold (>1 GB vs the
6 MB L2). Per width every candidate is screened, then the ``--top`` best plus
the baselines are re-timed with A/B-alternated replays.

Candidates: BLOCK_SIZE_M x BLOCK_SIZE_N x BLOCK_SIZE_K x num_warps x
num_stages x GROUP_SIZE_M from the flags (default BM {2,4,8,16}, BN {32,64,128},
BK {64}, warps {4,8}, stages {2,3}, G {1}). 1Cat's known-good M5 tile
BM2/BN128/BK64/w4/s3 (docs/design/sm70_qwen38_flash_next_nvfp4.md) is in the
default grid. BK=64 keeps the legacy per-row K order (1Cat saw bitwise-equal
rows); add --bk 32,64 to also try BK32 (draft-only numerics change).

Baselines per width: prev (image 1.8.0-dev2: 1Cat tile at M1/M5, 0.0.3 tile
BM16/BN32/BK64 elsewhere), 1cat (1Cat tile forced), table (the SX table as
currently configured).

Table emission: the best tile per measured width, preferring - within
``--tie`` (2%) of the fastest - the previous width's choice (fewer Triton
variants to warm), then 1Cat's tile, then BK64. Range boundaries sit at the
midpoints between measured widths; the last range ends at --max-table-m.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _b3_draft_common as C  # noqa: E402
import torch  # noqa: E402

DEFAULT_M = (
    "1,2,3,4,5,6,8,10,12,15,16,18,20,24,30,32,36,40,48,60,64,72,80,96,100,120,128"
)
ONECAT = {
    "BLOCK_SIZE_M": 2,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.replace(" ", "").split(",") if v]


def _key(cfg: dict[str, int]) -> tuple[int, ...]:
    return (
        cfg["BLOCK_SIZE_M"],
        cfg["BLOCK_SIZE_N"],
        cfg["BLOCK_SIZE_K"],
        cfg["GROUP_SIZE_M"],
        cfg["num_warps"],
        cfg["num_stages"],
    )


def candidates(args) -> list[dict[str, int]]:
    grid = []
    for bm, bn, bk, g, w, s in itertools.product(
        _ints(args.bm),
        _ints(args.bn),
        _ints(args.bk),
        _ints(args.group_m),
        _ints(args.warps),
        _ints(args.stages),
    ):
        grid.append(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": g,
                "SPLIT_K": 1,
                "num_warps": w,
                "num_stages": s,
            }
        )
    if all(_key(c) != _key(ONECAT) for c in grid):
        grid.append(dict(ONECAT))
    return grid


def _time_one(bench: C.MoeBench, ctx_factory, replays: int) -> float:
    graph = bench.graph(ctx_factory)
    try:
        return statistics.median(C.time_graph(graph, bench.calls, replays))
    finally:
        del graph
        torch.cuda.synchronize()


def _check_output(m: int, experts, cfg: dict[str, int], seed: int) -> dict[str, object]:
    w1, w2 = experts[0]
    weights, ids = C.make_routing(m, seed)
    hidden = C.make_hidden(m, seed + 1)
    with C.previous_default_ctx():
        prev = C.run_moe(hidden, w1, w2, weights, ids)
    with C.tile_ctx(cfg):
        got = C.run_moe(hidden, w1, w2, weights, ids)
    torch.cuda.synchronize()
    return {
        "bitwise_vs_prev": C.bit_equal16(got, prev),
        "max_abs_vs_prev": C.max_abs(got, prev),
        "finite": bool(torch.isfinite(got).all()),
    }


def sweep_width(m: int, experts, grid, args) -> dict[str, object]:
    bench = C.MoeBench(
        m,
        experts,
        routings=args.routings,
        seed=args.seed + m,
        overlap=args.overlap,
        group=args.group,
    )
    screened = []
    errors = {}
    for cfg in grid:
        try:
            us = _time_one(bench, lambda cfg=cfg: C.tile_ctx(cfg), args.replays)
        except Exception as err:  # noqa: BLE001 - Triton OutOfResources etc.
            errors[C.tile_str(cfg)] = f"{type(err).__name__}: {str(err)[:160]}"
            torch.cuda.synchronize()
            continue
        screened.append((us, cfg))
    screened.sort(key=lambda item: item[0])
    finalists = screened[: args.top]
    graphs = {
        "prev": bench.graph(C.previous_default_ctx),
        "1cat": bench.graph(lambda: C.tile_ctx(ONECAT)),
        "table": bench.graph(C.table_ctx),
    }
    for _, cfg in finalists:
        name = C.tile_str(cfg)
        if name not in graphs:
            graphs[name] = bench.graph(lambda cfg=cfg: C.tile_ctx(cfg))
    final = C.bench_alternating(graphs, bench.calls, rounds=args.rounds)
    del graphs
    torch.cuda.synchronize()
    with C.table_ctx():
        table_cfg = C.selected_config(m)
    with C.previous_default_ctx():
        prev_cfg = C.selected_config(m)
    ranked = [
        {"config": cfg, "tile": C.tile_str(cfg), "us": final[C.tile_str(cfg)]}
        for _, cfg in finalists
    ]
    ranked.sort(key=lambda item: item["us"])
    return {
        "m": m,
        "distinct_experts_uniform": C.expected_distinct_experts(m),
        "prev_us": final["prev"],
        "prev_tile": C.tile_str(prev_cfg),
        "1cat_us": final["1cat"],
        "table_us": final["table"],
        "table_tile": C.tile_str(table_cfg),
        "finalists": ranked,
        "screen": [{"tile": C.tile_str(cfg), "us": us} for us, cfg in screened],
        "errors": errors,
    }


def choose(results: list[dict[str, object]], tie: float) -> dict[int, dict]:
    chosen: dict[int, dict] = {}
    previous = None
    for res in sorted(results, key=lambda r: r["m"]):
        finalists = res["finalists"] or [
            {"config": dict(ONECAT), "tile": C.tile_str(ONECAT), "us": res["1cat_us"]}
        ]
        best_us = finalists[0]["us"]
        near = [f for f in finalists if f["us"] <= best_us * (1.0 + tie)]

        def rank(f):
            cfg = f["config"]
            return (
                0 if previous is not None and _key(cfg) == _key(previous) else 1,
                0 if _key(cfg) == _key(ONECAT) else 1,
                0 if cfg["BLOCK_SIZE_K"] == 64 else 1,
                f["us"],
            )

        pick = min(near, key=rank)
        chosen[res["m"]] = pick
        previous = pick["config"]
    return chosen


def emit_table(chosen: dict[int, dict], max_table_m: int, meta: dict) -> dict:
    ms = sorted(chosen)
    entries = []
    for i, m in enumerate(ms):
        lo = 1 if i == 0 else (ms[i - 1] + m) // 2 + 1
        hi = (m + ms[i + 1]) // 2 if i + 1 < len(ms) else max(m, max_table_m)
        cfg = {k: v for k, v in chosen[m]["config"].items() if k != "SPLIT_K"}
        if entries and entries[-1]["config"] == cfg:
            entries[-1]["m"][1] = hi
            entries[-1]["measured"][str(m)] = round(chosen[m]["us"], 2)
        else:
            entries.append(
                {"m": [lo, hi], "config": cfg, "measured": {str(m): round(chosen[m]["us"], 2)}}
            )
    return {
        "_comment": [
            "SX_OPT_MTP_DRAFT_TILES table emitted by sx_tests/b3-draft/sweep_mtp_draft_tiles.py.",
            "m: inclusive width range; measured: us per draft MoE call at the swept widths.",
        ],
        "shape": list(C.SHAPE),
        **meta,
        "table": entries,
    }


def _interp(results: dict[int, float], m: int) -> float | None:
    if m in results:
        return results[m]
    lower = [k for k in results if k < m]
    upper = [k for k in results if k > m]
    if not lower or not upper:
        return None
    a, b = max(lower), min(upper)
    return results[a] + (results[b] - results[a]) * (m - a) / (b - a)


def print_round_estimate(results: list[dict], chosen: dict[int, dict]) -> None:
    prev = {r["m"]: r["prev_us"] for r in results}
    best = {m: c["us"] for m, c in chosen.items()}
    rows = []
    for b in (1, 2, 4, 8, 12, 16, 24):
        for split in (True, False):
            dec = b if split else -(-b // 5) * 5
            p0, p1 = _interp(prev, 5 * b), _interp(prev, dec)
            n0, n1 = _interp(best, 5 * b), _interp(best, dec)
            if None in (p0, p1, n0, n1):
                continue
            before = p0 + 3 * p1
            after = n0 + 3 * n1
            rows.append(
                [
                    f"C{b}",
                    "split" if split else "padded",
                    f"{5 * b}+3x{dec}",
                    f"{before / 1000:.2f}",
                    f"{after / 1000:.2f}",
                    f"{(before - after) / 1000:.2f}",
                ]
            )
    C.print_table(
        "k=4 draft MoE per round (step0 at 5B + 3 decode steps; ms, from measured widths)",
        ["batch", "draft graphs", "widths", "prev", "swept", "saved"],
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--m", default=DEFAULT_M, help="widths to sweep")
    parser.add_argument("--bm", default="2,4,8,16")
    parser.add_argument("--bn", default="32,64,128")
    parser.add_argument("--bk", default="64")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--stages", default="2,3")
    parser.add_argument("--group-m", default="1")
    parser.add_argument("--weights", default="random", help="'random' or checkpoint dir")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--weight-copies", type=int, default=1)
    parser.add_argument("--routings", type=int, default=8, help="routings per graph")
    parser.add_argument("--overlap", type=float, default=0.0, help="row expert overlap")
    parser.add_argument("--group", type=int, default=5, help="rows per request (overlap)")
    parser.add_argument("--replays", type=int, default=12, help="screen replays")
    parser.add_argument("--rounds", type=int, default=6, help="final A/B rounds")
    parser.add_argument("--top", type=int, default=5, help="finalists per width")
    parser.add_argument("--tie", type=float, default=0.02)
    parser.add_argument("--max-table-m", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--check", action="store_true", help="output check of choices")
    parser.add_argument("--out", default="mtp_draft_sweep.json")
    parser.add_argument("--emit-table", default="")
    args = parser.parse_args()

    if not C.sm70_available():
        print("requires an SM70 GPU", file=sys.stderr)
        return 2
    ms = sorted(set(_ints(args.m)))
    grid = candidates(args)
    print(f"widths {ms}\n{len(grid)} candidates: {[C.tile_str(c) for c in grid]}")

    t0 = time.time()
    if args.weights == "random":
        experts = [C.make_experts(1234 + i) for i in range(max(1, args.weight_copies))]
    else:
        real = C.load_real_experts(args.weights, args.tp_rank)
        experts = [real] + [
            C.make_experts(1234 + i) for i in range(max(0, args.weight_copies - 1))
        ]
    print(f"weights ready ({args.weights}) in {time.time() - t0:.1f}s")

    results = []
    with torch.inference_mode():
        for m in ms:
            t1 = time.time()
            res = sweep_width(m, experts, grid, args)
            results.append(res)
            top = res["finalists"][0] if res["finalists"] else None
            print(
                f"M{m:4d}: prev {res['prev_us']:8.1f} us ({res['prev_tile']}), "
                f"1Cat {res['1cat_us']:8.1f}, table {res['table_us']:8.1f} "
                f"({res['table_tile']}), best "
                f"{top['us'] if top else float('nan'):8.1f} ({top['tile'] if top else '-'})"
                f"  [{time.time() - t1:.0f}s, {len(res['errors'])} failed]",
                flush=True,
            )
        chosen = choose(results, args.tie)
        if args.check:
            for m, pick in chosen.items():
                pick["check"] = _check_output(m, experts, pick["config"], args.seed + 7 * m)

    rows = []
    for res in results:
        pick = chosen[res["m"]]
        rows.append(
            [
                res["m"],
                f"{res['distinct_experts_uniform']:.0f}",
                f"{res['prev_us']:.1f}",
                f"{res['1cat_us']:.1f}",
                f"{res['table_us']:.1f}",
                f"{pick['us']:.1f}",
                pick["tile"],
                f"{res['prev_us'] / pick['us']:.2f}x",
                str(pick.get("check", {}).get("bitwise_vs_prev", "-")),
            ]
        )
    C.print_table(
        "MTP draft MoE sweep (us per fused_experts call, CUDA graph, cold weights)",
        ["M", "D(10M)", "prev", "1Cat", "table", "chosen", "chosen tile", "prev/chosen", "bitwise"],
        rows,
    )
    print_round_estimate(results, chosen)

    try:
        import triton

        triton_version = triton.__version__
    except Exception:  # noqa: BLE001
        triton_version = "unknown"
    meta = {
        "device": torch.cuda.get_device_name(),
        "source": (
            f"sweep {datetime.datetime.now().isoformat(timespec='seconds')} "
            f"weights={args.weights} routings={args.routings} overlap={args.overlap}"
        ),
        "triton_version": triton_version,
    }
    payload = {
        "meta": {**meta, "torch_version": torch.__version__, "args": vars(args)},
        "results": results,
        "chosen": {str(m): pick for m, pick in chosen.items()},
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"\nwrote {args.out}")
    if args.emit_table:
        table = emit_table(chosen, args.max_table_m, meta)
        with open(args.emit_table, "w") as f:
            json.dump(table, f, indent=2)
        print(f"wrote {args.emit_table}: " + "; ".join(
            f"M{e['m'][0]}-{e['m'][1]} {C.tile_str(e['config'])}" for e in table["table"]
        ))
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
