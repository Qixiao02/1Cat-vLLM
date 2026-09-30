"""mtp_bench.py <port> <model> <out.json> : speed on natural requests, meant for the MTP arms (also runs without MTP).

Two workloads, each at concurrency 1 (8 requests one after another) and 4 (two groups of 4 sent together):
  code : 8 programming tasks, thinking off, greedy, up to 512 tokens
  chat : 8 open questions (Chinese and English), thinking on, the server's default sampling, up to 1024 tokens
Per request: prompt and completion tokens, time to first token, decode tok/s = (completion tokens - 1) /
(last token - first token). Per cell: mean per-request decode tok/s, aggregate tok/s (all completion tokens / time
from the first send to the last token), and tokens per verify round from the engine's spec-decode counters
(1 + accepted / drafts; empty without speculative decoding)."""
import json
import sys
import threading
import time
import urllib.request

port, model, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
CODE = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into seconds, with tests.",
    "用 Python 实现一个 LRU 缓存类，支持 get/put，O(1) 复杂度，并写出单元测试。",
    "Implement quicksort and mergesort in C++ with templates, and a main() that benchmarks both.",
    "Write a SQL schema for an e-commerce order system (users, products, orders, order_items) and five example queries.",
    "用 JavaScript 写防抖和节流两个函数，解释它们的区别，并给出使用示例。",
    "Write a Go HTTP server with two JSON endpoints and graceful shutdown.",
    "Write a Python script that reads a CSV of sales, groups by month and product, and plots a bar chart with matplotlib.",
    "Implement a thread-safe bounded blocking queue in Java with put/take and a small producer-consumer demo.",
]
CHAT = [
    "为什么天空是蓝色的？请从物理角度解释，并说明日落为什么是红色的。",
    "Compare TCP and QUIC: handshake, head-of-line blocking and congestion control.",
    "一家跨境电商想降低退货率，请给出分步骤的分析和改进方案。",
    "Explain step by step, for a beginner, how a transformer decoder generates text.",
    "比较定期盘点、永续盘存和 ABC 分析三种库存管理方法，各自适合什么场景？",
    "A train leaves at 9:40 and arrives at 13:05 after covering 287 km. What is its average speed? Show the reasoning.",
    "写一封给供应商的邮件，说明因为质量问题需要退回一批货，语气专业但友好。",
    "What are the trade-offs between microservices and a modular monolith for a 10-person team?",
]


def spec_counters():
    text = urllib.request.urlopen("http://127.0.0.1:%d/metrics" % port, timeout=10).read().decode()
    got = {"drafts": 0.0, "accepted": 0.0, "found": False}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            got["drafts"] += float(line.rsplit(" ", 1)[1]); got["found"] = True
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            got["accepted"] += float(line.rsplit(" ", 1)[1])
    return got


def one(prompt, workload, record):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True}}
    if workload == "code":
        body.update(max_tokens=512, temperature=0, chat_template_kwargs={"enable_thinking": False})
    else:
        body.update(max_tokens=1024, chat_template_kwargs={"enable_thinking": True})
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        now = time.time()
                        first = first or now
                        last = now
    except Exception as e:  # noqa: BLE001
        record.update(error=repr(e)[:300])
        return
    n = usage["completion_tokens"] if usage else 0
    record.update(prompt_tokens=usage["prompt_tokens"] if usage else None, completion_tokens=n, t_send=t0,
                  t_first=first, t_last=last, ttft=(first - t0) if first else None,
                  decode_tok_s=((n - 1) / (last - first)) if first and last and last > first and n > 1 else None)


def cell(workload, conc):
    prompts = CODE if workload == "code" else CHAT
    before = spec_counters()
    records = [dict(prompt=p[:60]) for p in prompts]
    t0 = time.time()
    for start in range(0, len(prompts), conc):
        threads = [threading.Thread(target=one, args=(prompts[i], workload, records[i]))
                   for i in range(start, min(start + conc, len(prompts)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.time() - t0
    after = spec_counters()
    ok = [r for r in records if "error" not in r and r.get("decode_tok_s")]
    drafts = after["drafts"] - before["drafts"]
    res = dict(workload=workload, conc=conc, requests=records, errors=len(records) - len(ok),
               completion_tokens=sum(r["completion_tokens"] for r in ok),
               decode_tok_s_mean=round(sum(r["decode_tok_s"] for r in ok) / len(ok), 2) if ok else None,
               agg_tok_s=round(sum(r["completion_tokens"] for r in ok) / wall, 1), wall_s=round(wall, 1),
               tokens_per_round=round(1 + (after["accepted"] - before["accepted"]) / drafts, 3) if drafts else None,
               ttft_mean=round(sum(r["ttft"] for r in ok) / len(ok), 3) if ok else None)
    print("[mtp_bench] %-4s C%d: decode %s tok/s per request, aggregate %s tok/s, tokens per round %s, "
          "%d tokens, ttft %s s, errors %d" % (workload, conc, res["decode_tok_s_mean"], res["agg_tok_s"],
                                              res["tokens_per_round"], res["completion_tokens"], res["ttft_mean"],
                                              res["errors"]), flush=True)
    return res


one(CODE[0], "code", {})  # warm-up, not recorded
cells = [cell(w, c) for w in ("code", "chat") for c in (1, 4)]
json.dump({"port": port, "cells": cells}, open(out, "w"), ensure_ascii=False, indent=1)
