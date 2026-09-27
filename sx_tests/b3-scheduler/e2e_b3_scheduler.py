# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of the batch-3a scheduler group on a live TP4 server.

GPUs: 4 x V100-32GB (TP4), image shixiang/1cat-vllm-v100:1.8.0-dev2-sm70main
with this overlay bind-mounted (vllm/v1/core/sched/scheduler.py). Stdlib only;
reuses the prompt builder of ../align-chunking/e2e_align_multiblock.py. Run on
an otherwise idle server.

1) miss == hit (default mode). For each prompt length (1000, 1200, 2048, 3136,
   8192, 32768 tokens of natural text, fixed seed): request 1 (prefix-cache
   miss) and request 2 (identical prompt), greedy, 64 tokens, streamed.
   ASSERT greedy output 2 == output 1 (the resend restores the deepest
   checkpoint of run 1 and replays exactly run 1's remaining chunks), and
   ASSERT the vllm:prefix_cache_hits delta of request 2 == the chunk-plan
   expectation for --lane / --tail-min (printed per length). Records TTFT and,
   in the MTP lane, the acceptance length (1 + accepted / drafts).

   MTP lane (compose override with --speculative-config
   '{"method":"mtp","num_speculative_tokens":4}', gmu 0.87):
     e2e_b3_scheduler.py --lane mtp --block 816 --out /tmp/b3_mtp_new.json
   Log line: "SX align multi-block prefill chunking enabled (mtp lane, k=4,
   eagle tail): state block 816, ...". Expected hits (tail_min 3136):
   1000/1200/2048/3136 -> 0, 8192 -> 7344, 32768 -> 31824.
   Control: restart with SX_OPT_ALIGN_MULTIBLOCK_SPEC=0 (per-block, log
   "... disabled (speculative decoding keeps per-block checkpoints
   (SX_OPT_ALIGN_MULTIBLOCK_SPEC=0))"), run with --per-block and
   --out /tmp/b3_mtp_old.json; then --compare new old: TTFT 2K/8K/32K should
   drop ~30-40% (profile: 0.71/3.82/10.37 s at k4 per-block), outputs may
   differ only by late near-tie divergences (chunk grid), acceptance length
   within noise.

   No-MTP lane (production compose): --lane nomtp --block 784. Expected hits
   with PF4 (tail_min 3136): 1000..3136 -> 0, 8192 -> 7840, 32768 -> 31360.
   Control: SX_OPT_ALIGN_TAIL_MIN_TOKENS=0 and --tail-min 0 (1.8.0-dev2
   plan: 2048 -> 1568, 8192 -> 7840, 32768 -> 32144).

2) --mixed: decode stalls under the decode-aware cap (PF3, opt-in).
   N background greedy streams (--decoders, default 16 no-MTP / 4 MTP) of
   ~200-token prompts; after --warmup seconds a --inject-len prompt (8192)
   arrives, and 0.2 s later a 450-token prompt. Reports per-stream max and
   p99 inter-chunk gap inside the prefill window, tokens produced in the
   window, and both TTFTs. A/B by restarting the server with
   SX_OPT_PREFILL_CAP_WITH_DECODES=0 / 1 (optionally SX_OPT_PREFILL_CAP_BLOCKS
   2/3/4, SX_OPT_PREFILL_CAP_TOKENS 0/4096) and --compare the JSON files.
   Expected [derived, design_2 PF3]: cap 3 cuts the max gap for an 8K prompt
   at C16 from ~1.1 s to ~0.4 s at +8-17% long-prompt TTFT; the 450-token
   prompt's TTFT drops from ~1.3-1.5 s to ~0.4-0.55 s.

3) --shared: shared system prompt with short distinct suffixes (review fix
   for PF4). Four requests = one system prompt (--shared-len, default 1600
   no-MTP / 2000 MTP) + a distinct --suffix-len (default 300) suffix, sent
   one after another on a reset prefix cache. The shared checkpoint boundary
   is then also each request's own tail checkpoint position. ASSERT the
   prefix_cache_hits deltas: PF4 (tail_min >= prompt length) -> [0, 0, L, L]
   (the second sharer takes the shared-prefix stop, the third one on hits);
   tail_min 0 (1.8.0-dev2) -> [0, L, L, L]. L = 1568 (no-MTP, block 784) or
   816 (MTP, block 816, both attention page layouts). A result of
   [0, 0, 0, 0] is the regression (no sharer ever restores the state).

Exit code 0 on success, 1 when an assertion fails.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "align-chunking"),
)

from e2e_align_multiblock import (  # noqa: E402
    _build_prompt,
    _get,
    _margin,
    _model,
    _post,
    _prefix_hits,
    _stream_completion,
)

BUDGET = 8192


def chunk_plan(length: int, block: int, eagle: bool, tail_min: int,
               per_block: bool = False, budget: int = BUDGET) -> list[int]:
    """Chunk ends of one prompt on an idle engine (mirrors the scheduler)."""
    last = length - 1
    tail = last - last % block
    if eagle:
        tail = max(tail - block, 0)
    ends, start = [], 0
    while start < length:
        end = start + min(length - start, budget)
        if end < length and end // block * block > start:
            end = end // block * block
        if not per_block and start % block == 0:
            skip_tail = (tail_min > 0 and end >= length
                         and length - start <= tail_min)
            stops = [] if skip_tail else [tail]
        else:
            stops = [(start // block + 1) * block, tail]
        end = min([s for s in stops if start < s < end], default=end)
        ends.append(end)
        start = end
    return ends


def expected_hit(length: int, block: int, eagle: bool, tail_min: int,
                 per_block: bool = False) -> int:
    reachable = (length - 1) // block * block - (block if eagle else 0)
    ends = chunk_plan(length, block, eagle, tail_min, per_block)
    return max([0] + [e for e in ends if e % block == 0 and e <= reachable])


def shared_expectation(lane: str, block: int, tail_min: int, shared_len: int,
                       suffix_len: int) -> list[int] | None:
    """Expected prefix-hit tokens of four sequential sharers (see 3)), or
    None when the geometry is not the one this check models (the shared
    checkpoint must be each request's own tail checkpoint position)."""
    n = shared_len + suffix_len
    if n > BUDGET:
        return None
    if lane == "nomtp":
        boundary = shared_len // block * block
        if boundary == 0 or boundary != (n - 1) // block * block:
            return None
    else:
        # Eagle/MTP: the tail is one block below the last full block, and a
        # system prompt of >= 2 blocks in a request of < 3 blocks maps to the
        # first block boundary in both attention page layouts (16 / block).
        if shared_len < 2 * block or (n - 1) // block != 2:
            return None
        boundary = block
    if tail_min and n <= tail_min:
        return [0, 0, boundary, boundary]
    return [0, boundary, boundary, boundary]


def run_shared(args) -> int:
    base = args.base_url.rstrip("/")
    model = args.model or _model(base)
    shared_len = args.shared_len or (2000 if args.lane == "mtp" else 1600)
    want = shared_expectation(args.lane, args.block, args.tail_min, shared_len,
                              args.suffix_len)
    try:
        _post(base, "/reset_prefix_cache", {}, timeout=60)
    except Exception:
        pass
    system = _build_prompt(base, model, shared_len, seed=shared_len,
                           nonce=f"{args.nonce}-shared-{time.time_ns()}")
    hits = []
    for i in range(4):
        suffix = _build_prompt(base, model, args.suffix_len, seed=7000 + i,
                               nonce=f"{args.nonce}-sfx-{i}-{time.time_ns()}")
        h0 = _prefix_hits(base)
        _stream_completion(base, model, system + suffix, args.max_tokens)
        hits.append(int(_prefix_hits(base) - h0))
    print(f"shared system prompt {shared_len} + suffix {args.suffix_len} "
          f"({args.lane}, block {args.block}, tail_min {args.tail_min}): hits "
          f"{hits}, expected {want}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"shared": {"hits": hits, "expected": want}}, f, indent=1)
    if want is not None and hits != want:
        print("  FAIL: unexpected shared-prefix hits")
        return 1
    return 0


def _metric_sum(base: str, names: tuple[str, ...]) -> float:
    total = 0.0
    for line in _get(base, "/metrics").splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


def _spec_counters(base: str) -> tuple[float, float]:
    accepted = _metric_sum(base, ("vllm:spec_decode_num_accepted_tokens_total",
                                  "vllm:spec_decode_num_accepted_tokens"))
    drafts = _metric_sum(base, ("vllm:spec_decode_num_drafts_total",
                                "vllm:spec_decode_num_drafts"))
    return accepted, drafts


def run_miss_hit(args) -> int:
    base = args.base_url.rstrip("/")
    model = args.model or _model(base)
    eagle = args.lane == "mtp"
    print(f"model: {model}; lane {args.lane}, state block {args.block}, "
          f"tail_min {args.tail_min}, per_block {args.per_block}")
    try:
        _post(base, "/reset_prefix_cache", {}, timeout=60)
    except Exception:
        pass
    ok = True
    out = {"model": model, "lane": args.lane, "block": args.block,
           "tail_min": args.tail_min, "per_block": args.per_block,
           "lengths": {}}
    for length in args.lengths:
        ids = _build_prompt(base, model, length, seed=length, nonce=args.nonce)
        a0, d0 = _spec_counters(base)
        h0 = _prefix_hits(base)
        miss = _stream_completion(base, model, ids, args.max_tokens)
        h1 = _prefix_hits(base)
        hit = _stream_completion(base, model, ids, args.max_tokens)
        h2 = _prefix_hits(base)
        a1, d1 = _spec_counters(base)
        want = expected_hit(length, args.block, eagle, args.tail_min,
                            args.per_block)
        plan = chunk_plan(length, args.block, eagle, args.tail_min, args.per_block)
        same = miss["tokens"] == hit["tokens"] and miss["text"] == hit["text"]
        acc_len = 1 + (a1 - a0) / (d1 - d0) if d1 > d0 else None
        row = {
            "ttft_miss": miss["ttft"], "ttft_hit": hit["ttft"],
            "hits_first": h1 - h0, "hits_second": h2 - h1,
            "expected_hit": want, "plan": plan, "miss_equals_hit": same,
            "acceptance_length": acc_len,
            "tokens": miss["tokens"], "text": miss["text"],
            "top_logprobs": miss["top_logprobs"],
        }
        out["lengths"][str(length)] = row
        acc = f" | acceptance length {acc_len:.2f}" if acc_len else ""
        print(f"L={length}: plan {plan} | TTFT miss {miss['ttft']:.3f}s hit "
              f"{hit['ttft']:.3f}s | hits first {h1 - h0:.0f} second "
              f"{h2 - h1:.0f} (expected {want}) | miss==hit {same}{acc}")
        if not same:
            ok = False
            print("  FAIL: greedy output differs between prefix-cache miss and hit")
        if h2 - h1 != want:
            ok = False
            print("  FAIL: unexpected prefix-cache hit length on the resend")
        if h1 - h0 > 0:
            print("  WARN: first request already hit the cache; use a new --nonce")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.out)
    return 0 if ok else 1


def _stream_times(base: str, model: str, prompt_ids: list[int], max_tokens: int,
                  stop: threading.Event | None, rec: dict) -> None:
    payload = {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": True,
    }
    req = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    rec["t0"] = time.perf_counter()
    rec["times"] = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if any(c.get("text") for c in chunk.get("choices", [])):
                rec["times"].append(time.perf_counter())
            if stop is not None and stop.is_set():
                break  # closing the stream aborts the request server-side


def run_mixed(args) -> int:
    base = args.base_url.rstrip("/")
    model = args.model or _model(base)
    stop = threading.Event()
    decoders = []
    for i in range(args.decoders):
        ids = _build_prompt(base, model, 200, seed=5000 + i,
                            nonce=f"{args.nonce}-dec-{i}-{time.time_ns()}")
        rec: dict = {}
        t = threading.Thread(target=_stream_times,
                             args=(base, model, ids, 4096, stop, rec), daemon=True)
        decoders.append((t, rec))
    long_ids = _build_prompt(base, model, args.inject_len, seed=args.inject_len,
                             nonce=f"{args.nonce}-long-{time.time_ns()}")
    short_ids = _build_prompt(base, model, 450, seed=450,
                              nonce=f"{args.nonce}-short-{time.time_ns()}")
    for t, _ in decoders:
        t.start()
    time.sleep(args.warmup)
    long_rec: dict = {}
    short_rec: dict = {}
    tl = threading.Thread(target=_stream_times,
                          args=(base, model, long_ids, 8, None, long_rec))
    ts = threading.Thread(target=_stream_times,
                          args=(base, model, short_ids, 8, None, short_rec))
    t_inject = time.perf_counter()
    tl.start()
    time.sleep(0.2)
    ts.start()
    tl.join()
    ts.join()
    t_done = time.perf_counter()
    time.sleep(0.5)
    stop.set()
    for t, _ in decoders:
        t.join(timeout=60)
    gaps_all, max_gaps, tokens_in_window = [], [], []
    for _, rec in decoders:
        times = [x for x in rec.get("times", []) if t_inject - 0.5 <= x <= t_done + 0.5]
        gaps = [b - a for a, b in zip(times, times[1:])]
        gaps_all += gaps
        max_gaps.append(max(gaps) if gaps else float("nan"))
        tokens_in_window.append(len(times))
    gaps_all.sort()
    p99 = gaps_all[int(0.99 * (len(gaps_all) - 1))] if gaps_all else float("nan")
    row = {
        "decoders": args.decoders, "inject_len": args.inject_len,
        "ttft_long": long_rec["times"][0] - long_rec["t0"],
        "ttft_short": short_rec["times"][0] - short_rec["t0"],
        "gap_max": max(max_gaps), "gap_p99": p99,
        "gap_mean": statistics.fmean(gaps_all) if gaps_all else float("nan"),
        "chunks_in_window_mean": statistics.fmean(tokens_in_window),
        "window_s": t_done - t_inject,
    }
    print(json.dumps(row, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mixed": row}, f, indent=1)
        print("wrote", args.out)
    return 0


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"compare {path_a} (A) vs {path_b} (B)")
    if "mixed" in a and "mixed" in b:
        for key in ("ttft_long", "ttft_short", "gap_max", "gap_p99", "gap_mean",
                    "chunks_in_window_mean"):
            print(f"  {key}: A {a['mixed'][key]:.3f} B {b['mixed'][key]:.3f}")
        return 0
    for length in sorted(set(a["lengths"]) & set(b["lengths"]), key=int):
        ra, rb = a["lengths"][length], b["lengths"][length]
        ta, tb = ra["tokens"], rb["tokens"]
        div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if div is None and len(ta) == len(tb):
            verdict = "identical"
        else:
            div = div if div is not None else min(len(ta), len(tb))
            ma = _margin(ra["top_logprobs"][div] if div < len(ra["top_logprobs"])
                         else None)
            mb = _margin(rb["top_logprobs"][div] if div < len(rb["top_logprobs"])
                         else None)
            verdict = (f"first divergence at output token {div} "
                       f"(top1-top2 logprob margin A {ma}, B {mb})")
        speed = (rb["ttft_miss"] / ra["ttft_miss"]) if ra["ttft_miss"] else float("nan")
        print(f"L={length}: {verdict}; TTFT miss A {ra['ttft_miss']:.3f}s B "
              f"{rb['ttft_miss']:.3f}s (B/A {speed:.2f}x); acceptance A "
              f"{ra.get('acceptance_length')} B {rb.get('acceptance_length')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--lane", choices=("nomtp", "mtp"), default="nomtp")
    p.add_argument("--block", type=int, default=None,
                   help="Mamba state block (784 no-MTP, 816 MTP bring-up)")
    p.add_argument("--tail-min", type=int, default=3136)
    p.add_argument("--per-block", action="store_true",
                   help="expectations for the per-block policy (control run)")
    p.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                   default=[1000, 1200, 2048, 3136, 8192, 32768])
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--nonce", default="sx-b3-sched-e2e-v1")
    p.add_argument("--out", default=None)
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--decoders", type=int, default=None)
    p.add_argument("--inject-len", type=int, default=8192)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    p.add_argument("--shared", action="store_true",
                   help="shared system prompt + short suffix check (3)")
    p.add_argument("--shared-len", type=int, default=None)
    p.add_argument("--suffix-len", type=int, default=300)
    args = p.parse_args()
    if args.block is None:
        args.block = 816 if args.lane == "mtp" else 784
    if args.decoders is None:
        args.decoders = 4 if args.lane == "mtp" else 16
    if args.compare:
        return compare(*args.compare)
    try:
        if args.shared:
            return run_shared(args)
        return run_mixed(args) if args.mixed else run_miss_hit(args)
    except urllib.error.URLError as e:
        print("server not reachable:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
