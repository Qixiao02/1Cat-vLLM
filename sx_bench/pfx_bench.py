#!/usr/bin/env python3
"""pfx_bench.py --port P --model M --out FILE [--lengths 8000,16000,32000,64000] [--conc 4] [--gen 400] [--passes 2]
                [--gpus 0,1,2,3]

Cold prefill + steady-state decode at a fixed concurrency (default 4), streaming chat completions.

Every request gets its own random prompt (seeded random words behind a random session id), so no two prompts share
a prefix and nothing can hit the prefix cache: every prompt is a full cold prefill. Prompts are sized with the
engine's own /tokenize (chat form), so prompt_tokens matches the target within about 0.2%. The same --seed gives
the same prompts on any engine with the same tokenizer.

Per cell (length x pass) CONC requests start together (t = seconds after the common start):
  ttft[i]          first generated token of request i
  prefill_tok_s    sum(prompt_tokens) / max(ttft): prompt tokens per second until every request has its first token
  decode window    [max(first token time), min(last token time)]: every request is decoding
  decode_tok_s     tokens per second of one request inside that window (mean over requests); decode_agg = their sum
Token counts over time come from continuous usage stats, not from counting SSE chunks.

When the KV cache cannot hold all CONC prompts at once, some requests wait until others finish and the window above
does not exist. The "wave" fields describe the part that ran together: the requests whose first token arrived before
any request finished (wave == CONC when everything fits), with prefill and decode computed over that wave only.

Detailed record (schema 2), per cell:
  requests[]   per request: prompt/completion tokens, send offset, ttft, finish, its own decode rate (first to last
               token), its rate inside the decode window, tokens it already had when the window opened, inter-token
               gaps (p50/p90/p99/max, ms) and the full token timeline [ms since cell start, tokens so far]
  server       change of every vllm:*_total / *_sum / *_count metric over the cell (prompt and generation tokens,
               prefix-cache queries and hits, preemptions, and the engine's own TTFT / queue / prefill / decode /
               e2e histograms as count + sum of seconds)
  timeline[]   once a second: KV usage, running and waiting requests, and per GPU utilisation %, memory MiB, power W
"""
import argparse
import http.client
import json
import os
import random
import subprocess
import threading
import time

WORDS = (
    "river stone bridge market water engineer tide ledger bakery corner bread morning apple cake school library "
    "thursday night worker book autumn hill road farm tractor harbor master boat net children paper canal parent "
    "price coal clock tower winter stair lantern garden window letter station train signal platform ticket coat "
    "umbrella rain street lamp bicycle baker miller carpenter hammer timber roof chimney smoke kettle table chair "
    "candle mirror curtain carpet door key lock gate fence orchard cherry plum barley wheat mill wheel stream pond "
    "heron swallow sparrow fox badger meadow valley ridge summit glacier pebble sand dune shore lighthouse anchor "
    "rope sail mast compass chart captain sailor cargo crate barrel warehouse office desk pencil paper ink stamp "
    "envelope parcel courier village council mayor festival drum flute violin choir dance ribbon mask lantern "
    "copper silver iron glass brick mortar plaster paint brush canvas frame gallery museum statue fountain square "
    "quiet slow narrow bright heavy gentle distant early sudden careful ordinary hidden broken patient steady "
    "walks carries measures repairs counts writes opens closes follows gathers watches returns crosses climbs"
).split()


def get(port, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    data = conn.getresponse().read()
    conn.close()
    return data.decode()


def post(port, path, body, timeout=600):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError("HTTP %d %s" % (resp.status, data[:300]))
    return json.loads(data)


def raw_metrics(port):
    """Every vllm:* sample as {name: value}; names keep their _total/_sum/_count suffix, labels are dropped
    except for histogram buckets, which are skipped."""
    out = {}
    info = {}
    for line in get(port, "/metrics").splitlines():
        if not line.startswith("vllm:") or "_bucket{" in line:
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0][5:]
        if name == "cache_config_info" and "{" in head:
            for item in head.split("{", 1)[1].rstrip("}").split('",'):
                key, _, val = item.partition('="')
                info[key.strip()] = val.strip('"')
            continue
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    out["_cache_config"] = info
    return out


def metrics(port):
    return raw_metrics(port)


def count_tokens(port, model, text):
    return post(port, "/tokenize", {
        "model": model, "messages": [{"role": "user", "content": text}], "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False}})["count"]


def build_prompt(port, model, target, seed):
    """A prompt of `target` prompt tokens (chat template included) that shares no prefix with any other seed."""
    rng = random.Random(seed)
    head = "Session %032x. Read the notes below, then follow the instruction at the end.\n\n" % rng.getrandbits(128)
    tail = "\n\nInstruction: write a long and detailed essay about the history of bridges. Do not stop early."
    words = [rng.choice(WORDS) for _ in range(int(target * 1.3) + 64)]
    k = int(target / 1.15)
    count = 0
    for _ in range(8):
        k = max(1, min(k, len(words)))
        text = head + " ".join(words[:k]) + tail
        count = count_tokens(port, model, text)
        if abs(count - target) <= max(4, target // 1000):
            break
        k += int(round((target - count) * k / max(count, 1)))
    return text, count


def stream_one(port, model, text, gen, barrier, out, idx):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": text}], "max_tokens": gen, "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}})
    rec = {"events": [], "prompt_tokens": None, "error": None}
    out[idx] = rec
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
        barrier.wait()
        rec["t_send"] = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["error"] = "HTTP %d %s" % (resp.status, resp.read()[:300])
            return
        last = 0
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens"):
                rec["prompt_tokens"] = usage["prompt_tokens"]
            done = usage.get("completion_tokens")
            if done is None:                     # no continuous usage: count text chunks instead
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                done = last + 1 if (delta.get("content") or delta.get("reasoning_content")) else last
            if done > last:                      # one event per step that produced tokens
                rec["events"].append((now, done))
                last = done
        conn.close()
    except Exception as exc:  # noqa: BLE001 - recorded and reported per request
        rec["error"] = repr(exc)


def gpu_sample(gpus):
    """[[utilisation %, memory MiB, power W], ...] for the given GPU indices, or None."""
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits",
             "-i", gpus], capture_output=True, text=True, timeout=5).stdout
        return [[float(x) for x in line.split(",")] for line in text.strip().splitlines()]
    except Exception:  # noqa: BLE001 - sampling is best effort
        return None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def window_rates(out, members, lo, hi):
    rates = {}
    for i in members:
        inside = [e for e in out[i]["events"] if lo <= e[0] <= hi]
        if len(inside) >= 2 and inside[-1][0] > inside[0][0]:
            rates[i] = (inside[-1][1] - inside[0][1]) / (inside[-1][0] - inside[0][0])
    return rates


def run_cell(port, model, prompts, gen, gpus=None):
    conc = len(prompts)
    wall_start = time.strftime("%F %T")
    loadavg = open("/proc/loadavg").read().split()[:3] if os.path.exists("/proc/loadavg") else None
    before = raw_metrics(port)
    out = [None] * conc
    barrier = threading.Barrier(conc)
    threads = [threading.Thread(target=stream_one, args=(port, model, p, gen, barrier, out, i))
               for i, p in enumerate(prompts)]
    timeline = []
    state = {"stop": False, "t0": time.perf_counter()}

    def poll():
        while not state["stop"]:
            tick = time.perf_counter()
            sample = {"t": round(tick - state["t0"], 2)}
            try:
                m = raw_metrics(port)
                sample.update(kv=round(m.get("kv_cache_usage_perc", 0.0), 4),
                              running=int(m.get("num_requests_running", 0)),
                              waiting=int(m.get("num_requests_waiting", 0)))
            except Exception:  # noqa: BLE001 - metrics are best effort
                pass
            if gpus:
                sample["gpu"] = gpu_sample(gpus)
            timeline.append(sample)
            time.sleep(max(0.0, 1.0 - (time.perf_counter() - tick)))

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    state["stop"] = True
    poller.join()
    after = raw_metrics(port)
    server = {}
    for name, value in after.items():
        if name.endswith(("_total", "_sum", "_count")) and isinstance(value, float):
            delta = value - before.get(name, 0.0)
            if delta:
                server[name] = round(delta, 4)
    cell = {
        "wall_start": wall_start, "loadavg": loadavg,
        "kv_peak": round(max([s.get("kv", 0.0) for s in timeline] or [0.0]), 4),
        "waiting_peak": max([s.get("waiting", 0) for s in timeline] or [0]),
        "running_peak": max([s.get("running", 0) for s in timeline] or [0]),
        "cache_hit_tokens": server.get("prefix_cache_hits_total", 0),
        "preemptions": server.get("num_preemptions_total", 0),
        "server": server, "timeline": timeline,
    }
    errors = [r["error"] for r in out if r["error"]]
    if errors or any(not r["events"] for r in out):
        cell["errors"] = errors or ["no tokens"]
        return cell
    start = min(r["t_send"] for r in out)
    firsts = [r["events"][0][0] for r in out]
    lasts = [r["events"][-1][0] for r in out]
    prompt_tokens = [r["prompt_tokens"] for r in out]
    ttft = sorted(f - start for f in firsts)
    # everyone decoding: the definition used for the headline numbers
    win_a, win_b = max(firsts), min(lasts)
    rates = window_rates(out, range(conc), win_a, win_b)
    full = len(rates) == conc
    # the part that ran together when the KV cache cannot hold every prompt
    wave = [i for i in range(conc) if firsts[i] < win_b]
    wave_a = max(firsts[i] for i in wave)
    wave_rates = window_rates(out, wave, wave_a, win_b)
    wave_ok = len(wave_rates) == len(wave)
    requests = []
    for i, r in enumerate(out):
        times = [e[0] for e in r["events"]]
        gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
        tokens = r["events"][-1][1]
        span = times[-1] - times[0]
        requests.append({
            "i": i, "prompt_tokens": r["prompt_tokens"], "completion_tokens": tokens,
            "send_offset_s": round(r["t_send"] - start, 4),
            "ttft_s": round(times[0] - start, 3), "finish_s": round(times[-1] - start, 3),
            "own_decode_tok_s": round((tokens - r["events"][0][1]) / span, 2) if span > 0 else None,
            "window_decode_tok_s": round(wave_rates[i], 2) if i in wave_rates else None,
            "in_wave": i in wave,
            "tokens_before_window": max([e[1] for e in r["events"] if e[0] <= wave_a] or [0]),
            "itl_ms": {"p50": round(percentile(gaps, 0.5), 1), "p90": round(percentile(gaps, 0.9), 1),
                       "p99": round(percentile(gaps, 0.99), 1), "max": round(max(gaps), 1)} if gaps else None,
            "events": [[int(round((t - start) * 1000)), n] for t, n in r["events"]],
        })
    cell.update({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": [r["events"][-1][1] for r in out],
        "ttft": [round(x, 3) for x in ttft],
        "ttft_first": round(ttft[0], 3), "ttft_last": round(ttft[-1], 3),
        "prefill_tok_s": round(sum(prompt_tokens) / max(ttft), 1),
        "decode_window_s": round(win_b - win_a, 3),
        "decode_tok_s": round(sum(rates.values()) / conc, 2) if full else None,
        "decode_tok_s_min": round(min(rates.values()), 2) if full else None,
        "decode_tok_s_max": round(max(rates.values()), 2) if full else None,
        "decode_agg_tok_s": round(sum(rates.values()), 2) if full else None,
        "e2e_s": round(max(lasts) - start, 3),
        "wave": len(wave),
        "wave_ttft_last": round(wave_a - start, 3),
        "wave_prefill_tok_s": round(sum(prompt_tokens[i] for i in wave) / (wave_a - start), 1),
        "wave_window_s": round(win_b - wave_a, 3),
        "wave_decode_tok_s": round(sum(wave_rates.values()) / len(wave), 2) if wave_ok else None,
        "wave_decode_agg_tok_s": round(sum(wave_rates.values()), 2) if wave_ok else None,
        "requests": requests,
    })
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="8000,16000,32000,64000")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--gen", type=int, default=400)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--gpus", default="", help="GPU indices of this engine for the per-second samples, e.g. 0,1,2,3")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    first = raw_metrics(a.port)
    result = {
        "schema": 2, "args": vars(a), "started": time.strftime("%F %T"),
        "engine": {"version": json.loads(get(a.port, "/version")).get("version"),
                   "models": [{"id": m.get("id"), "max_model_len": m.get("max_model_len")}
                              for m in json.loads(get(a.port, "/v1/models")).get("data", [])],
                   "cache_config": first.get("_cache_config")},
        "cells": [],
    }

    def save():
        json.dump(result, open(a.out, "w"), indent=1)

    def prompts_for(n, length, tag):
        return [build_prompt(a.port, a.model, length, "%d-c%d-%s-%d-%d" % (a.seed, a.conc, tag, length, i))[0]
                for i in range(n)]

    # Warm-up, not recorded in cells: the multi-request path on short prompts and one long single request.
    t0 = time.time()
    warm = run_cell(a.port, a.model, prompts_for(a.conc, 2000, "warm"), 32)
    warm1 = run_cell(a.port, a.model, prompts_for(1, 16000, "warm1"), 16)
    result["warmup"] = [{k: c.get(k) for k in ("prefill_tok_s", "decode_tok_s", "ttft", "errors")}
                        for c in (warm, warm1)]
    print("warm-up %.0fs: C%d 2K %s | C1 16K %s" % (
        time.time() - t0, a.conc, warm.get("prefill_tok_s", warm.get("errors")),
        warm1.get("prefill_tok_s", warm1.get("errors"))), flush=True)

    for p in range(a.passes):
        for length in lengths:
            prompts = prompts_for(a.conc, length, "p%d" % p)      # built (and tokenized) before the clock starts
            while raw_metrics(a.port).get("num_requests_running", 0) > 0:
                time.sleep(0.5)
            cell = run_cell(a.port, a.model, prompts, a.gen, a.gpus or None)
            cell.update({"conc": a.conc, "length": length, "pass": p})
            result["cells"].append(cell)
            save()
            if "errors" in cell:
                print("pass %d C%-2d len %6d ERROR %s" % (p, a.conc, length, cell["errors"]), flush=True)
                continue
            note = "" if cell["wave"] == a.conc else " | WAVE %d/%d: prefill %.1f decode %s (last of wave %.2fs)" % (
                cell["wave"], a.conc, cell["wave_prefill_tok_s"], cell["wave_decode_tok_s"], cell["wave_ttft_last"])
            print("pass %d C%-2d len %6d prompt %s | ttft first/last %6.2f/%6.2f s | prefill %7.1f tok/s | decode %6.2f"
                  " tok/s per stream, %7.2f agg (window %.1fs) | e2e %6.1fs | kv peak %.1f%% | wait %d | cache hits %d"
                  " | preempt %d%s" % (
                      p, a.conc, length, cell["prompt_tokens"][0], cell["ttft_first"], cell["ttft_last"],
                      cell["prefill_tok_s"], cell["decode_tok_s"] or 0, cell["decode_agg_tok_s"] or 0,
                      cell["decode_window_s"], cell["e2e_s"], 100 * cell["kv_peak"], cell["waiting_peak"],
                      cell["cache_hit_tokens"], cell["preemptions"], note), flush=True)
    result["finished"] = time.strftime("%F %T")
    save()
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
