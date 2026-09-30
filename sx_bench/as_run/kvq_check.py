"""kvq_check.py <port> <model> <out.json> [lengths] : answer-quality check of one engine, greedy, thinking off.

Needle retrieval: a prompt of random words with eight "verification code" records spread evenly over its length;
the model has to return all eight codes. One prompt per length (default 8000,32000,64000 words, about as many
tokens). Then four short questions with a checkable answer. The same seed gives the same prompts on any engine, so
the FP16-KV engine and the E4M3-KV engine answer identical requests."""
import json
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, "/mnt/2t/build/pfx_ab")
from pfx_bench import WORDS  # noqa: E402

port, model, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
lengths = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "8000,32000,64000").split(",")]
DEPOTS = ["Halden", "Moravia", "Tressel", "Quillon", "Berwick", "Isolde", "Ferrand", "Oskarn"]


def chat(content, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    return r["choices"][0]["message"].get("content") or "", r["usage"], time.time() - t0


def needle_prompt(n_words, seed):
    rng = random.Random(seed)
    codes = {d: "%06d" % rng.randrange(10 ** 6) for d in DEPOTS}
    chunk = n_words // (len(DEPOTS) + 1)
    parts = []
    for depot in DEPOTS:
        parts.append(" ".join(rng.choice(WORDS) for _ in range(chunk)))
        parts.append("\n[Record] The verification code of depot %s is %s.\n" % (depot, codes[depot]))
    parts.append(" ".join(rng.choice(WORDS) for _ in range(chunk)))
    ask = ("\n\nThe text above contains eight records of the form \"The verification code of depot X is N\". "
           "Return all eight as one JSON object mapping the depot name to its code as a string. JSON only.")
    return "".join(parts) + ask, codes


result = {"port": port, "needle": [], "short": []}
for n in lengths:
    prompt, codes = needle_prompt(n, 7000 + n)
    text, usage, dt = chat(prompt, 300)
    hit = sum(1 for d, c in codes.items() if re.search(r'"%s"\s*:\s*"?%s"?' % (d, c), text))
    result["needle"].append({"words": n, "prompt_tokens": usage["prompt_tokens"], "correct": hit, "of": len(codes),
                             "seconds": round(dt, 1), "answer": text[:400]})
    print("[kvq %d] needle %6d tokens: %d/%d codes correct (%.1fs)" % (port, usage["prompt_tokens"], hit, len(codes), dt),
          flush=True)

SHORT = [
    ("计算 37×43+125 等于多少？只输出数字。", r"\b1716\b"),
    ("把这些数字从小到大排序：42, 7, 19, 88, 3, 56。只输出排序结果，用英文逗号分隔。", r"3\s*,\s*7\s*,\s*19\s*,\s*42\s*,\s*56\s*,\s*88"),
    ("What is the capital of Australia? Answer with one word.", r"(?i)canberra"),
    ("把这句话翻译成英文：今天的会议推迟到明天下午三点。", r"(?i)(?=.*tomorrow)(?=.*(3|three))(?=.*(postpone|delay|reschedul|moved|put off))"),
]
for question, pattern in SHORT:
    text, usage, dt = chat(question, 120)
    ok = bool(re.search(pattern, text, re.S))
    result["short"].append({"question": question, "answer": text[:200], "ok": ok})
    print("[kvq %d] %s  %s -> %s" % (port, "ok  " if ok else "FAIL", question[:28], text[:70].replace("\n", " ")),
          flush=True)
json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
print("[kvq %d] needle %d/%d, short %d/%d" % (
    port, sum(x["correct"] for x in result["needle"]), sum(x["of"] for x in result["needle"]),
    sum(x["ok"] for x in result["short"]), len(SHORT)))
