"""image_bench.py <port> <model> <out.json> : image requests against an engine that has the vision tower loaded.
Every request carries its own noisy 448x448 PNG (so the multimodal processor cache never hits), a one-sentence prompt, greedy decoding, 64 new
tokens, streaming. C1: 6 requests one after the other; C4: 8 requests, 4 at a time. Per request: time to first token, end-to-end time,
generated tokens. Standard library only."""
import base64, json, os, random, struct, sys, threading, time, urllib.request, zlib


def png(w, h, seed):
    rnd = random.Random(seed)
    base = rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)
    rows = []
    for _ in range(h):
        row = bytearray([0])
        for _ in range(w):
            row += bytes(min(255, max(0, c + rnd.randrange(-40, 41))) for c in base)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]


def one(seed):
    url = "data:image/png;base64," + base64.b64encode(png(448, 448, seed)).decode()
    body = {"model": model, "max_tokens": 64, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": "Describe this image in one sentence."}]}]}
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    ttft, toks, text = None, 0, ""
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    toks = d["usage"].get("completion_tokens", toks)
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
                    if piece:
                        text += piece
                        if ttft is None:
                            ttft = time.time() - t0
    except Exception as e:  # noqa: BLE001
        return dict(seed=seed, error="%s: %s" % (type(e).__name__, str(e)[:160]))
    e2e = time.time() - t0
    return dict(seed=seed, ttft=ttft, e2e=e2e, tokens=toks, tok_s=(toks / (e2e - ttft)) if ttft and e2e > ttft and toks else None, text=text[:120])


def run(conc, n, seed0):
    res = [None] * n
    nxt = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= n:
                return
            res[i] = one(seed0 + i)
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res, time.time() - t0


one(9999)  # warm-up (not recorded)
report = {}
for conc, n, seed0 in ((1, 6, 100), (4, 8, 200)):
    res, wall = run(conc, n, seed0)
    ok = [r for r in res if "error" not in r and r.get("ttft")]
    report["C%d" % conc] = dict(requests=res, wall=wall, errors=n - len(ok))
    if ok:
        print("[image_bench] C%d: %d/%d ok, ttft mean %.2fs, e2e mean %.2fs, decode %.1f tok/s per request, wall %.1fs" % (
            conc, len(ok), n, sum(r["ttft"] for r in ok) / len(ok), sum(r["e2e"] for r in ok) / len(ok),
            sum(r["tok_s"] or 0 for r in ok) / len(ok), wall))
    else:
        print("[image_bench] C%d: all %d requests failed: %s" % (conc, n, res[0]))
json.dump(report, open(out, "w"), indent=1)
