"""calib_traffic.py <port> <model> <out.json> : calibration traffic for the QSA K/V observer.

Sends a mix of real text through the chat endpoint of the eager calibration instance: Python and C++/CUDA source,
English and Chinese documentation, a Chinese Q&A set, an engine log, JSON records, random words (the benchmark's
prompt material), short chats in both languages with thinking on, a tool-call request and a multi-turn chat;
prompt lengths from a few dozen tokens to about 36K. Every K and V row written to the cache is observed, prefill
and decode, so prompt and generated text both count. Requests up to 12K tokens run three at a time, the longer
ones alone (the eager calibration instance has a 40,960-token context and a KV pool of about that size)."""
import glob
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/mnt/2t/build/pfx_ab")
from pfx_bench import WORDS  # noqa: E402

port, model, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
SRC = "/mnt/2t/build/official-d304698/src"
DOCS = "/mnt/raid1/Wren-dev/docs"


def post(path, body, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def files(patterns, skip=0):
    paths = sorted(p for pat in patterns for p in glob.glob(pat, recursive=True) if os.path.isfile(p))
    text = []
    for p in paths[skip:]:
        try:
            text.append("### %s\n%s\n" % (os.path.relpath(p, "/"), open(p, encoding="utf-8").read()))
        except (UnicodeDecodeError, OSError):
            continue
        if sum(map(len, text)) > 700_000:
            break
    return "".join(text)


def fit(text, target):
    """Cut text to about `target` tokens, from the chars-per-token ratio of its first 20,000 characters."""
    sample = text[:20000]
    ratio = len(sample) / post("/tokenize", {"model": model, "prompt": sample})["count"]
    return text[: int(target * ratio * 0.97)]


rng = random.Random(20260930)
py = files([SRC + "/vllm/v1/**/*.py"])
py2 = files([SRC + "/vllm/model_executor/**/*.py", SRC + "/vllm/entrypoints/**/*.py"])
cpp = files([SRC + "/csrc/**/*.cpp", SRC + "/csrc/**/*.cu", SRC + "/csrc/**/*.h", SRC + "/flash-attention-v100/**/*.cu",
             SRC + "/flash-attention-v100/**/*.cpp"])
en = files([SRC + "/docs/**/*.md"])
zh = files([DOCS + "/platform/*.md", DOCS + "/README.md"])
zh_qa = files([DOCS + "/eval-center/*.md"])
log = open("/mnt/2t/build/mtp_kv/engine_p707-u87s24.log", encoding="utf-8", errors="replace").read()
words = " ".join(rng.choice(WORDS) for _ in range(50_000))
records = json.dumps([{"id": i, "sku": "SX-%05d" % rng.randrange(10 ** 5), "amount": round(rng.uniform(1, 9999), 2),
                       "qty": rng.randrange(1, 500), "city": rng.choice(["深圳", "广州", "Shanghai", "Austin"]),
                       "ok": rng.random() < 0.8} for i in range(260)], ensure_ascii=False, indent=1)
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询某个城市某一天的天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
                   "required": ["city", "date"]}}},
    {"type": "function", "function": {
        "name": "get_order_status", "description": "Look up the status of an order by its id",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]


def doc(text, target, ask):
    return [{"role": "user", "content": fit(text, target) + "\n\n" + ask}] if text else None


# name, messages, max_tokens, thinking, temperature (None = server default), extra body
SMALL = [
    ("py-8k", doc(py, 8000, "解释这段代码的主要结构，并指出三处可以改进的地方。"), 400, True, None, {}),
    ("cpp-12k", doc(cpp, 12000, "Review this code for bugs and undefined behaviour. Be specific."), 300, True, None, {}),
    ("zh-8k", doc(zh, 8000, "用中文总结上面文档的要点，列出 5 条。"), 500, True, None, {}),
    ("json-6k", doc(records, 6000, "计算所有记录 amount 字段的总和，并给出 amount 最大的三条记录的 id。"),
     300, True, None, {}),
    ("chat-poem", [{"role": "user", "content": "写一首关于秋天码头的七言绝句，再用两句话解释它的意境。"}],
     400, True, None, {}),
    ("chat-translate", [{"role": "user", "content": "把下面这段话翻译成英文和日文：我们的仓库今天下午三点以后不再收货，"
                         "请提前联系调度确认卸货时间。"}], 300, True, None, {}),
    ("chat-math", [{"role": "user", "content": "一个水池有两个进水管和一个出水管，单开甲管 6 小时注满，单开乙管 8 小时注满，"
                    "单开出水管 12 小时放空。三管同时开，几小时注满？写出推导。"}], 600, True, None, {}),
    ("chat-tcp", [{"role": "user", "content": "解释 TCP 三次握手和四次挥手，并说明为什么挥手需要四次。"}],
     500, False, None, {}),
    ("chat-code", [{"role": "user", "content": "Write a Python function that merges overlapping intervals, with type "
                    "hints, a docstring and three pytest cases."}], 600, True, 0, {}),
    ("chat-os", [{"role": "user", "content": "Explain the difference between processes and threads, and when a "
                  "coroutine is the better choice."}], 400, False, None, {}),
    ("tools", [{"role": "user", "content": "帮我查一下深圳明天的天气，再查订单 A1024 的状态。"}], 300, False, 0,
     {"tools": TOOLS, "tool_choice": "auto"}),
    ("multi-turn", [
        {"role": "system", "content": "你是世翔公司的客服助手，回答要简洁、礼貌，不确定的事情要明确说明。"},
        {"role": "user", "content": "我的订单显示已发货三天了还没有物流更新，怎么办？"},
        {"role": "assistant", "content": "您好，给您带来不便很抱歉。请提供订单号，我帮您核实承运商的揽收记录。"},
        {"role": "user", "content": "订单号是 SX20260928-0417，另外我想把收货地址改到公司。"}], 400, True, None, {}),
]
LARGE = [
    ("py-30k", doc(py2, 30000, "Write a concise summary of the modules above and list their public functions."),
     300, False, 0, {}),
    ("en-16k", doc(en, 16000, "Summarise the documentation above in ten bullet points."), 400, False, None, {}),
    ("zhqa-30k", doc(zh_qa, 30000, "从上面的问答集中挑出三个问题，给出更详细的回答。"), 400, False, 0, {}),
    ("log-20k", doc(log, 20000, "找出这份日志里的错误和警告，并按原因归类。"), 300, False, None, {}),
    ("words-30k", doc(words, 30000, "Summarise the text above in three sentences."), 200, False, None, {}),
    ("py-36k", doc(py + cpp, 36000, "List every class name defined in the code above."), 150, False, 0, {}),
]


def run(item):
    name, messages, max_tokens, thinking, temperature, extra = item
    if messages is None:
        return {"name": name, "skipped": "no source text"}
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    if temperature is not None:
        body["temperature"] = temperature
    t0 = time.time()
    try:
        r = post("/v1/chat/completions", body)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "error": repr(e)[:300], "seconds": round(time.time() - t0, 1)}
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + (msg.get("content") or "")
    res = {"name": name, "prompt_tokens": r["usage"]["prompt_tokens"],
           "completion_tokens": r["usage"]["completion_tokens"], "finish": r["choices"][0]["finish_reason"],
           "seconds": round(time.time() - t0, 1), "tool_calls": len(msg.get("tool_calls") or []),
           "text_head": text[:160]}
    print("[calib] %-14s prompt %6d gen %4d %-10s %6.1fs" % (name, res["prompt_tokens"], res["completion_tokens"],
                                                           res["finish"], res["seconds"]), flush=True)
    return res


t0 = time.time()
with ThreadPoolExecutor(3) as pool:
    results = list(pool.map(run, SMALL))
results += [run(item) for item in LARGE]
bad = [r for r in results if "error" in r or "skipped" in r]
json.dump({"seconds": round(time.time() - t0, 1), "requests": results}, open(out, "w"), ensure_ascii=False, indent=1)
print("[calib] %d requests, %d prompt tokens, %d generated tokens, %.0fs, %d failed or skipped%s" % (
    len(results), sum(r.get("prompt_tokens", 0) for r in results),
    sum(r.get("completion_tokens", 0) for r in results), time.time() - t0, len(bad),
    "".join("\n[calib] BAD %s" % json.dumps(r, ensure_ascii=False) for r in bad)))
sys.exit(1 if len(bad) > 3 else 0)
