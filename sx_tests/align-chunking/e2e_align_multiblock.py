# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of SX_OPT_ALIGN_MULTIBLOCK on the live TP4 server (4 GPUs).

Stdlib only; talks to the OpenAI-compatible server of the deployed engine
(image shixiang/1cat-vllm-v100:1.7.2-sm70main + this overlay, TP4, prefix
caching on). Run it on an otherwise idle server.

Plan (the validation agent runs A, B and C):

A. New policy (default env). Start the server with the overlay, confirm the log
   line "SX align multi-block prefill chunking enabled: state block 784, ...",
   then:
       /opt/venv/bin/python e2e_align_multiblock.py --base-url http://127.0.0.1:8000 \
           --out /tmp/align_new.json --prefill-bench
   For each prompt length (2048, 8192, 32768 tokens of natural text, fixed seed):
     1. request 1 (prefix-cache miss): greedy, max_tokens 64, streamed; TTFT.
     2. request 2 (identical prompt, prefix-cache hit): TTFT and the
        vllm:prefix_cache_hits delta. Expected hit = (L-1)//784*784 with the
        tail checkpoint (2048 -> 1568, 8192 -> 7840, 32768 -> 32144).
     3. ASSERT: greedy output of request 2 == request 1 (token by token). The
        hit request replays exactly the miss request's last chunk from the
        same cached KV and recurrent state, so this must be bitwise equal.
   --prefill-bench: C in {1, 2, 4} concurrent distinct 8K prompts, max_tokens 1;
   prints aggregate prefill tok/s (baseline profile: 3.2K tok/s at C1,
   5.0K at C2, 5.7K at C4 with the per-block rule).

B. Old policy: restart the same server with SX_OPT_ALIGN_MULTIBLOCK=0 (log line
   "... disabled (SX_OPT_ALIGN_MULTIBLOCK=0) ...") and run
       ... --out /tmp/align_old.json --prefill-bench
   Request 1 == request 2 must also hold here.

C. Compare:  e2e_align_multiblock.py --compare /tmp/align_new.json /tmp/align_old.json
   Chunk boundaries differ between the policies (GDN 64-token chunk grid and
   GEMM M change), so outputs are NOT expected to be bitwise equal; the report
   lists, per length, identical-or-first-divergence index plus the logprob
   margin at that position. Expect identical text or a late divergence at a
   near-tie; an early divergence with a large margin is a failure to escalate.
   TTFT(new) should be ~2x lower at 8K/32K; 2K ~1.2-1.5x.

Exit code 0 on success, 1 when an assertion fails.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

BLOCK = 784


def _post(base: str, path: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(base: str, path: str) -> str:
    with urllib.request.urlopen(base + path, timeout=60) as resp:
        return resp.read().decode()


def _model(base: str) -> str:
    return json.loads(_get(base, "/v1/models"))["data"][0]["id"]


def _prefix_hits(base: str) -> float:
    total = 0.0
    for line in _get(base, "/metrics").splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


WORDS = dict(
    adj="amber brisk calm dusty eager frosty gentle hollow ivory jolly keen "
        "lunar misty noble olive plain quiet rusty silver tidy urban vivid".split(),
    noun="barrel crate lantern ledger parcel satchel spindle trunk valve "
         "wagon anchor bucket candle drum engine furnace".split(),
    city="Aldmere Brookfield Carrow Dunmore Eastwick Fairhaven Glenrock "
         "Harlow Ivybridge Juniper Kestrel Larkspur".split(),
    day="Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
)


def _natural_text(seed: int, n_sentences: int) -> str:
    rng = random.Random(seed)
    out = []
    for i in range(n_sentences):
        out.append(
            f"Ledger entry {i}: the {rng.choice(WORDS['adj'])} "
            f"{rng.choice(WORDS['noun'])} from {rng.choice(WORDS['city'])} "
            f"recorded {rng.randrange(2, 9999)} items on "
            f"{rng.choice(WORDS['day'])}, and the clerk noted that the "
            f"{rng.choice(WORDS['adj'])} {rng.choice(WORDS['noun'])} was "
            f"moved to {rng.choice(WORDS['city'])}."
        )
    return " ".join(out)


def _build_prompt(base: str, model: str, length: int, seed: int, nonce: str):
    question = (
        "\n\nQuestion: Which city appears most often in the ledger above, and "
        "how many entries mention it? Think briefly, then answer.\nAnswer:"
    )
    q = _post(base, "/tokenize", {"model": model, "prompt": question,
                                  "add_special_tokens": False})["tokens"]
    head = _post(base, "/tokenize", {"model": model, "prompt": f"[{nonce}] ",
                                     "add_special_tokens": False})["tokens"]
    body: list[int] = []
    sentences = 200
    while len(head) + len(body) + len(q) < length:
        text = _natural_text(seed, sentences)
        body = _post(base, "/tokenize", {"model": model, "prompt": text,
                                         "add_special_tokens": False})["tokens"]
        sentences *= 2
    body = body[: length - len(head) - len(q)]
    ids = head + body + q
    assert len(ids) == length
    return ids


def _stream_completion(base: str, model: str, prompt_ids: list[int],
                       max_tokens: int) -> dict:
    payload = {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0,
        "logprobs": 2, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    tokens: list[str] = []
    top: list[dict] = []
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if ttft is None and (choice.get("text") or choice.get("logprobs")):
                    ttft = time.perf_counter() - t0
                text.append(choice.get("text") or "")
                lp = choice.get("logprobs") or {}
                tokens += lp.get("tokens") or []
                top += lp.get("top_logprobs") or []
    return {"ttft": ttft, "e2e": time.perf_counter() - t0, "text": "".join(text),
            "tokens": tokens, "top_logprobs": top, "usage": usage}


def _prefill_bench(base: str, model: str, nonce: str, length: int = 8192) -> list:
    rows = []
    for conc in (1, 2, 4):
        prompts = [
            _build_prompt(base, model, length, 777, f"{nonce}-bench-{conc}-{i}-"
                          f"{time.time_ns()}")
            for i in range(conc)
        ]
        results: list = [None] * conc

        def worker(i):
            results[i] = _stream_completion(base, model, prompts[i], 1)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        tput = conc * length / wall
        rows.append({"concurrency": conc, "wall_s": wall, "prefill_tok_s": tput,
                     "ttft_max": max(r["ttft"] for r in results)})
        print(f"  prefill C{conc} x {length}: wall {wall:.2f}s "
              f"-> {tput:,.0f} tok/s (ttft max {rows[-1]['ttft_max']:.2f}s)")
    return rows


def run(args) -> int:
    base = args.base_url.rstrip("/")
    model = args.model or _model(base)
    print("model:", model)
    try:
        _post(base, "/reset_prefix_cache", {}, timeout=60)
    except Exception:
        pass  # dev endpoint may be disabled; a fresh server/nonce also works
    ok = True
    out = {"model": model, "nonce": args.nonce, "lengths": {}}
    for length in args.lengths:
        ids = _build_prompt(base, model, length, seed=length, nonce=args.nonce)
        h0 = _prefix_hits(base)
        miss = _stream_completion(base, model, ids, args.max_tokens)
        h1 = _prefix_hits(base)
        hit = _stream_completion(base, model, ids, args.max_tokens)
        h2 = _prefix_hits(base)
        expect_hit = (length - 1) // BLOCK * BLOCK
        same = miss["tokens"] == hit["tokens"] and miss["text"] == hit["text"]
        row = {
            "ttft_miss": miss["ttft"], "ttft_hit": hit["ttft"],
            "hits_first": h1 - h0, "hits_second": h2 - h1,
            "expected_hit_tail_on": expect_hit,
            "miss_equals_hit": same,
            "tokens": miss["tokens"], "text": miss["text"],
            "top_logprobs": miss["top_logprobs"],
        }
        out["lengths"][str(length)] = row
        print(f"L={length}: TTFT miss {miss['ttft']:.3f}s hit {hit['ttft']:.3f}s | "
              f"prefix hits first {h1 - h0:.0f} second {h2 - h1:.0f} "
              f"(tail-on expectation {expect_hit}) | miss==hit {same}")
        if not same:
            ok = False
            print("  FAIL: greedy output differs between prefix-cache miss and hit")
        if h1 - h0 > 0:
            print("  WARN: first request already hit the cache; use a new --nonce "
                  "or a fresh server for a cold TTFT")
    if args.prefill_bench:
        out["prefill_bench"] = _prefill_bench(base, model, args.nonce)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.out)
    return 0 if ok else 1


def _margin(top: dict | None) -> float | None:
    if not top or len(top) < 2:
        return None
    vals = sorted(top.values(), reverse=True)
    return vals[0] - vals[1]


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"compare {path_a} (A) vs {path_b} (B)")
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
        print(f"L={length}: {verdict}; TTFT miss A {ra['ttft_miss']:.3f}s "
              f"B {rb['ttft_miss']:.3f}s (B/A {speed:.2f}x); hit tokens A "
              f"{ra['hits_second']:.0f} B {rb['hits_second']:.0f}")
    for label, r in (("A", a), ("B", b)):
        for row in r.get("prefill_bench", []):
            print(f"prefill bench {label}: C{row['concurrency']} "
                  f"{row['prefill_tok_s']:,.0f} tok/s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                   default=[2048, 8192, 32768])
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--nonce", default="sx-align-e2e-v1")
    p.add_argument("--out", default=None)
    p.add_argument("--prefill-bench", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("NEW_JSON", "OLD_JSON"))
    args = p.parse_args()
    if args.compare:
        return compare(*args.compare)
    try:
        return run(args)
    except urllib.error.URLError as e:
        print("server not reachable:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
