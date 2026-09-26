# SPDX-License-Identifier: Apache-2.0
"""PW-1 end to end on the full TP4 model (4 x V100, offline in-process engine).

Drives the real engine step by step (VLLM_ENABLE_V1_MULTIPROCESSING=0, so the
scheduler sees every add_request between two known steps and batch
composition is identical across runs), with the production engine arguments
of deploy/inference/entrypoint.sh and the Flash-Next compose environment.

Modes (one process per mode; the script sets the switch before importing vllm):
  pw        default (SX_OPT_PIECEWISE_MIXED=1): mixed/prefill steps of 25..1024
            tokens replay PIECEWISE graphs at the padded size
  eagerpad  SX_OPT_PIECEWISE_EAGER_PADDED=1: same padded token counts, eager
            (no replay) -> isolates graph capture/replay from padding
  off       SX_OPT_PIECEWISE_MIXED=0: previous behaviour (eager, unpadded)

Run (GPUs 0-3 free, inside the image with the patched files mounted):
  cd sx_tests/b2-piecewise-graphs
  /opt/venv/bin/python e2e_pw_mixed.py run --mode pw       --out /tmp/pw.json
  /opt/venv/bin/python e2e_pw_mixed.py run --mode eagerpad --out /tmp/eagerpad.json
  /opt/venv/bin/python e2e_pw_mixed.py run --mode off      --out /tmp/off.json
  /opt/venv/bin/python e2e_pw_mixed.py run --mode pw --skip-bench --out /tmp/pw2.json
  /opt/venv/bin/python e2e_pw_mixed.py compare /tmp/pw.json /tmp/pw2.json --bitwise
  /opt/venv/bin/python e2e_pw_mixed.py compare /tmp/pw.json /tmp/eagerpad.json --bitwise
  /opt/venv/bin/python e2e_pw_mixed.py compare /tmp/pw.json /tmp/off.json
The pw vs pw2 run establishes run-to-run determinism first: TurboMind
autotunes the M<=24 routed-MoE shapes at warm-up by timing (not PW-1 related;
mixed steps above 24 tokens use the fixed default spec). If pw vs pw2 is not
bitwise, repeat the equality runs with --skip-bench
--env VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=0 for every mode.
Each run starts the engine once (~10-15 min incl. capture). Check the pw/eagerpad
startup log for "SX PW-1: PIECEWISE CUDA graphs for mixed/prefill steps of
25..1024 tokens ..." and "Graph capturing finished in N secs, took X GiB"
(1.8.0-dev1 baseline: 58 s, 0.42 GiB); off must not print the SX PW-1 line.

Phase 1 (equality): 16 greedy decode streams (ignore_eos, fixed lengths, so
the step schedule is identical in every mode even if tokens diverge), then
prompts of 8, 30, 100, 250, 450, 700, 784, 900, 1000 and 1500 tokens injected
one at a time every 4 steps -> mixed steps of about 24..1600 tokens (PIECEWISE
24, 48, 128, 320, 512, 768, 832, 960, 1024 and eager above 1024; the 1500
prompt is split by mamba align into 784 + 716). Token ids and top-5 logprobs
of every request are stored.
  pw vs eagerpad --bitwise: MUST be identical (tokens and logprob floats).
  pw vs off: padded GEMM shapes may round differently (like any batch-shape
  change); expect identical or late, near-tie divergences. The report gives
  identical requests, first divergence and the top-1/top-2 margin there.

Phase 2 (microbench, design item "16 decodes + 450/784-token prompt"): 16
decodes of 2K-token prompts in steady state (FULL 16 graph), then prompts of
100, 450, 784, 2 x 450 and 1000 tokens injected (max_tokens 1, 5 repeats
each); the longest engine.step() among the next 4 is the mixed step. Also
prefill-only 450/784 steps with no decodes running. Reported: median steady
decode step, median mixed/prefill step per scenario; compare prints pw/off.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

MODES = {
    "pw": {"SX_OPT_PIECEWISE_MIXED": "1", "SX_OPT_PIECEWISE_EAGER_PADDED": "0"},
    "eagerpad": {"SX_OPT_PIECEWISE_MIXED": "1", "SX_OPT_PIECEWISE_EAGER_PADDED": "1"},
    "off": {"SX_OPT_PIECEWISE_MIXED": "0", "SX_OPT_PIECEWISE_EAGER_PADDED": "0"},
}
# compose.swift15-flashnext-tp4-gpu0123.yaml (setdefault: an explicit env wins)
DEPLOY_ENV = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_QWEN4EXP_PLE_HOST_GIB": "12",
    "VLLM_SM70_QWEN38_HYBRID_PLE": "0",
    "VLLM_PLE_CPU_OFFLOAD": "0",
    "VLLM_PLE_DISK_OFFLOAD": "0",
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "1",
    "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "240",
    "OMP_NUM_THREADS": "8",
}
SENTENCES = [
    "The maintenance window for the storage cluster starts at nine in the evening.",
    "Each replica keeps a local copy of the index and rebuilds it after a restart.",
    "When the queue grows beyond its soft limit, new work is routed to the spare pool.",
    "The report lists every request that finished later than its deadline last week.",
    "A short summary at the top explains which services were affected and for how long.",
    "Operators prefer small, reversible changes that can be rolled back in minutes.",
    "The billing export runs nightly and writes one compressed file per customer.",
    "If a node stops sending heartbeats, the scheduler marks its tasks as lost.",
    "Latency percentiles are computed over five minute windows and kept for a month.",
    "The migration plan moves one region at a time and pauses between regions.",
    "Configuration files are validated before they are copied to the servers.",
    "A failed health check triggers an automatic restart after a short delay.",
    "Test traffic is replayed against the staging cluster before every release.",
    "The dashboard shows throughput, error rate and queue depth for each tenant.",
    "Old snapshots are deleted once the newer copy has been verified.",
    "The on-call engineer writes a timeline while the incident is still open.",
]
PHASE1_DECODE_LENS = [40, 64, 90, 120, 150, 180, 200, 230, 260, 300,
                      60, 80, 110, 140, 170, 210]
PHASE1_INJECT_LENS = [8, 30, 100, 250, 450, 700, 784, 900, 1000, 1500]
PHASE2_SCENARIOS = [
    ("mix100", [100]),
    ("mix450", [450]),
    ("mix784", [784]),
    ("mix450x2", [450, 450]),
    ("mix1000", [1000]),
]


def _build_engine(mode: str, model: str, extra: list[str], env: list[str]):
    for key, value in DEPLOY_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.update(MODES[mode])
    for item in env:
        key, _, value = item.partition("=")
        os.environ[key] = value
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine

    argv = [
        "--model", model,
        "--tensor-parallel-size", "4",
        "--dtype", "half",
        "--attention-backend", "FLASH_ATTN_V100",
        "--max-model-len", "131072",
        "--max-num-seqs", "24",
        "--max-num-batched-tokens", "8192",
        "--gpu-memory-utilization", "0.90",
        "--kv-cache-dtype", "auto",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--language-model-only",
        *extra,
    ]
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    engine_args = EngineArgs.from_cli_args(parser.parse_args(argv))
    return LLMEngine.from_engine_args(engine_args)


class Driver:
    def __init__(self, engine, seed: int = 20260926):
        from vllm import SamplingParams

        self.engine = engine
        self.SamplingParams = SamplingParams
        tok = engine.get_tokenizer()
        rng = random.Random(seed)
        text = " ".join(rng.choice(SENTENCES) for _ in range(900))
        self.corpus = tok.encode(text, add_special_tokens=False)
        self.tok = tok
        self.rng = rng
        self.counter = 0
        self.results: dict[str, dict] = {}
        self.step_times: list[float] = []

    def prompt(self, length: int) -> list[int]:
        self.counter += 1
        tag = self.tok.encode(f"Record {self.counter:05d}. ", add_special_tokens=False)
        start = self.rng.randrange(0, len(self.corpus) - length - 1)
        ids = tag + self.corpus[start : start + length]
        return ids[:length]

    def add(self, rid: str, length: int, max_tokens: int, logprobs: int | None = 5):
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            logprobs=logprobs,
            detokenize=False,
        )
        self.engine.add_request(rid, {"prompt_token_ids": self.prompt(length)}, params)

    def step(self) -> float:
        t0 = time.perf_counter()
        outputs = self.engine.step()
        dt = time.perf_counter() - t0
        self.step_times.append(dt)
        for out in outputs:
            comp = out.outputs[0]
            rec = self.results.setdefault(out.request_id, {})
            rec["token_ids"] = list(comp.token_ids)
            rec["n_out"] = len(comp.token_ids)
            if comp.logprobs is not None:
                rec["logprobs"] = [
                    sorted(
                        ([int(t), float(lp.logprob)] for t, lp in pos.items()),
                        key=lambda item: (-item[1], item[0]),
                    )
                    for pos in comp.logprobs
                ]
            rec["finished"] = bool(out.finished)
        return dt

    def n_out(self, rid: str) -> int:
        return self.results.get(rid, {}).get("n_out", 0)

    def drain(self, limit: int = 100000):
        steps = 0
        while self.engine.has_unfinished_requests() and steps < limit:
            self.step()
            steps += 1


def run_phase1(drv: Driver) -> dict:
    ids = [f"p1-dec-{i:02d}" for i in range(len(PHASE1_DECODE_LENS))]
    for rid, length in zip(ids, PHASE1_DECODE_LENS):
        drv.add(rid, length, max_tokens=200)
    while min(drv.n_out(rid) for rid in ids) < 1:
        drv.step()
    for length in PHASE1_INJECT_LENS:
        for _ in range(4):
            drv.step()
        drv.add(f"p1-inj-{length:04d}", length, max_tokens=24)
    drv.drain()
    return {rid: drv.results[rid] for rid in sorted(drv.results) if rid.startswith("p1-")}


def run_phase2(drv: Driver, reps: int) -> dict:
    ids = [f"p2-dec-{i:02d}" for i in range(16)]
    for rid in ids:
        drv.add(rid, 2048, max_tokens=4000, logprobs=None)
    while min(drv.n_out(rid) for rid in ids) < 2:
        drv.step()
    for _ in range(8):
        drv.step()
    steady = [drv.step() for _ in range(24)]
    bench: dict[str, list[float]] = {}
    for label, lengths in PHASE2_SCENARIOS:
        samples = []
        for rep in range(reps):
            for j, length in enumerate(lengths):
                drv.add(f"p2-{label}-{rep}-{j}", length, max_tokens=1, logprobs=None)
            window = [drv.step() for _ in range(4)]
            samples.append(max(window))
            for _ in range(4):
                drv.step()
        bench[label] = samples
    drv.engine.abort_request(ids)
    drv.drain()
    for label, length in (("solo450", 450), ("solo784", 784)):
        samples = []
        for rep in range(reps):
            drv.add(f"p2-{label}-{rep}", length, max_tokens=1, logprobs=None)
            window = []
            while drv.engine.has_unfinished_requests():
                window.append(drv.step())
            samples.append(max(window))
        bench[label] = samples
    return {
        "steady_decode_ms": 1e3 * statistics.median(steady),
        "steps_ms": {k: [1e3 * v for v in vals] for k, vals in bench.items()},
        "median_ms": {k: 1e3 * statistics.median(vals) for k, vals in bench.items()},
    }


def cmd_run(args) -> int:
    t0 = time.time()
    engine = _build_engine(
        args.mode, args.model, args.engine_arg or [], args.env or []
    )
    startup_s = time.time() - t0
    drv = Driver(engine)
    phase1 = run_phase1(drv)
    phase2 = run_phase2(drv, args.reps) if not args.skip_bench else None
    report = {
        "mode": args.mode,
        "env": {
            k: os.environ.get(k)
            for k in list(MODES["pw"])
            + list(DEPLOY_ENV)
            + [item.partition("=")[0] for item in args.env or []]
        },
        "startup_s": startup_s,
        "phase1": phase1,
        "phase2": phase2,
    }
    with open(args.out, "w") as f:
        json.dump(report, f)
    print(f"[PW-1 e2e] mode={args.mode} startup {startup_s:.0f} s -> {args.out}")
    if phase2:
        print(f"  steady M16 decode step {phase2['steady_decode_ms']:.2f} ms")
        for k, v in phase2["median_ms"].items():
            print(f"  {k:10s} step {v:8.2f} ms")
    return 0


def _margin(logprobs_pos) -> float | None:
    if logprobs_pos is None or len(logprobs_pos) < 2:
        return None
    return logprobs_pos[0][1] - logprobs_pos[1][1]


def cmd_compare(args) -> int:
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    ok = True
    same = 0
    rows = []
    for rid, ra in a["phase1"].items():
        rb = b["phase1"].get(rid)
        if rb is None:
            rows.append((rid, "missing in B", None))
            ok = False
            continue
        ta, tb = ra["token_ids"], rb["token_ids"]
        la, lb = ra.get("logprobs"), rb.get("logprobs")
        if ta == tb and la == lb:
            same += 1
            continue
        idx = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if idx is None and ta == tb:
            lp_idx = next(
                (i for i, (x, y) in enumerate(zip(la or [], lb or [])) if x != y),
                "n/a",
            )
            rows.append((rid, f"tokens equal, logprobs differ from pos {lp_idx}", None))
        else:
            idx = idx if idx is not None else min(len(ta), len(tb))
            margin = _margin(la[idx]) if la and idx < len(la) else None
            rows.append((rid, f"first token divergence at {idx}/{len(ta)}", margin))
        if args.bitwise:
            ok = False
    total = len(a["phase1"])
    print(f"[PW-1 compare] {args.a} ({a['mode']}) vs {args.b} ({b['mode']}): "
          f"{same}/{total} requests identical (tokens + top-5 logprobs)")
    for rid, what, margin in rows:
        m = f", top1-top2 margin {margin:.4f}" if margin is not None else ""
        print(f"  {rid}: {what}{m}")
    pa, pb = a.get("phase2"), b.get("phase2")
    if pa and pb:
        print(f"  steady decode: {pa['steady_decode_ms']:.2f} vs "
              f"{pb['steady_decode_ms']:.2f} ms")
        for k in pa["median_ms"]:
            va, vb = pa["median_ms"][k], pb["median_ms"].get(k)
            if vb:
                print(f"  {k:10s} {va:8.2f} vs {vb:8.2f} ms  ({vb - va:+.2f} ms, "
                      f"x{vb / va:.2f})")
    print(f"  startup: {a['startup_s']:.0f} vs {b['startup_s']:.0f} s")
    if args.bitwise and not ok:
        print("FAIL: --bitwise requested and outputs differ")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=sorted(MODES), required=True)
    run.add_argument("--model", default="/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4")
    run.add_argument("--out", required=True)
    run.add_argument("--reps", type=int, default=5)
    run.add_argument("--skip-bench", action="store_true")
    run.add_argument("--engine-arg", action="append",
                     help="extra engine CLI argument (repeatable)")
    run.add_argument("--env", action="append",
                     help="KEY=VALUE set before vllm is imported (repeatable)")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--bitwise", action="store_true")
    args = parser.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
