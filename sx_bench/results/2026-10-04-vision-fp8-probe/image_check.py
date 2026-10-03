"""image_check.py <port> <model> : one chat request with a generated image (a solid red square), to see that the vision path works.
Pure standard library (no PIL): the PNG is written with zlib."""
import base64, json, struct, sys, time, urllib.request, zlib

def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

port, model = sys.argv[1], sys.argv[2]
for name, rgb in (("red", (230, 20, 20)), ("blue", (20, 40, 230))):
    url = "data:image/png;base64," + base64.b64encode(png(224, 224, rgb)).decode()
    body = {"model": model, "max_tokens": 48, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "What is the main color of this image? Answer with one word."}]}]}
    req = urllib.request.Request("http://localhost:%s/v1/chat/completions" % port, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        m = r["choices"][0]["message"]
        print("image %-4s -> %r (%.1fs) usage %s" % (name, (m.get("content") or "").strip()[:60], time.time() - t0, r.get("usage")))
    except Exception as e:  # noqa: BLE001
        print("image %-4s -> FAILED %s: %s" % (name, type(e).__name__, str(e)[:200]))
