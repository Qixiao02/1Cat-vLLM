# SPDX-License-Identifier: Apache-2.0
"""Memory stress for a running engine: long prompts at the shapes that decide
the steady peak. Standard library only.

    python3 stress.py --port 8141 --model Swift-1.5-Qwen3.8-Flash-Next \
        --out stress.json [--events events.log]

Phase A sends four 8K and two 16K prompts at the same moment; phase B sends one
32K prompt (32000 tokens, so prompt + 64 generated tokens fit a 32768 window).
Every prompt is unique (random words behind a random session id), so prefix
caching cannot short-cut the prefill that the test is about. Prompt lengths are
calibrated against the engine's own ``/tokenize``.

``--events`` appends ``<epoch> stress_begin`` / ``stress_end`` lines that
``analyze.py`` uses to cut the GPU memory samples. Exit status is 0 when every
request returned HTTP 200, 1 otherwise. A failed request is a finding (the
engine may have run out of memory), not a script error.
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

WORDS = (
    "alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river "
    "stone cloud forest window engine copper silver garden harbor island "
    "lantern meadow orchid pepper quartz rocket saddle timber velvet walnut "
    "yellow zephyr anchor bridge candle dragon ember falcon glacier hollow"
).split()


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url: str, payload: dict, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        # The engine is local: a proxy from the environment must not get in the way.
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:500]
    except Exception as exc:  # noqa: BLE001 - connection reset etc. is a result
        return 0, f"{type(exc).__name__}: {exc}"


def make_text(rng: random.Random, words: int) -> str:
    session = rng.getrandbits(48)
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return f"session {session:012x}: {body}"


def calibrated_prompt(
    base: str, model: str, target_tokens: int, seed: int, tolerance: float = 0.01
) -> tuple[str, int]:
    """A unique prompt of about ``target_tokens`` tokens (engine's tokenizer)."""
    rng = random.Random(seed)
    words = max(8, int(target_tokens * 0.8))
    count = 0
    text = ""
    for _ in range(4):
        text = make_text(rng, words)
        status, body = _post(
            f"{base}/tokenize", {"model": model, "prompt": text}, timeout=120
        )
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"/tokenize failed: {status} {body}")
        count = int(body.get("count", len(body.get("tokens", []))))
        if abs(count - target_tokens) <= tolerance * target_tokens:
            break
        words = max(8, int(words * target_tokens / max(count, 1)))
    return text, count


def one_request(
    base: str, model: str, prompt: str, max_tokens: int, out: dict
) -> None:
    started = time.time()
    status, body = _post(
        f"{base}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=1800,
    )
    out["status"] = status
    out["seconds"] = round(time.time() - started, 2)
    if status == 200 and isinstance(body, dict):
        usage = body.get("usage", {})
        out["prompt_tokens"] = usage.get("prompt_tokens")
        out["completion_tokens"] = usage.get("completion_tokens")
    else:
        out["error"] = body if isinstance(body, str) else json.dumps(body)[:500]


def run_phase(
    base: str, model: str, name: str, lengths: list[int], max_tokens: int, seed: int
) -> list[dict]:
    prepared = []
    for index, length in enumerate(lengths):
        prompt, tokens = calibrated_prompt(base, model, length, seed * 100 + index)
        prepared.append({"phase": name, "target": length, "tokens": tokens, "prompt": prompt})
    results: list[dict] = [
        {k: v for k, v in item.items() if k != "prompt"} for item in prepared
    ]
    threads = [
        threading.Thread(
            target=one_request,
            args=(base, model, item["prompt"], max_tokens, results[i]),
        )
        for i, item in enumerate(prepared)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def _event(path: str | None, name: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f} {name}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8141)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default="Swift-1.5-Qwen3.8-Flash-Next")
    parser.add_argument("--out", required=True)
    parser.add_argument("--events")
    parser.add_argument("--seed", type=int, default=2026100201)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--concurrent", default="8000,8000,8000,8000,16000,16000",
        help="comma separated prompt tokens sent at the same time (phase A)",
    )
    parser.add_argument(
        "--single", default="32000",
        help="prompt tokens of the single request after phase A (phase B; empty = none)",
    )
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    concurrent = [int(x) for x in args.concurrent.split(",") if x.strip()]
    single = [int(x) for x in args.single.split(",") if x.strip()]

    _event(args.events, "stress_begin")
    results: list[dict] = []
    if concurrent:
        results += run_phase(base, args.model, "A", concurrent, args.max_tokens, args.seed)
    for index, length in enumerate(single):
        results += run_phase(base, args.model, "B", [length], args.max_tokens, args.seed + 1 + index)
    _event(args.events, "stress_end")

    failed = [r for r in results if r.get("status") != 200]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {"requests": results, "failed": len(failed), "total": len(results)},
            handle,
            indent=1,
        )
    for r in results:
        print(
            "stress %s target=%s tokens=%s status=%s %.1fs %s"
            % (
                r["phase"], r["target"], r.get("tokens"), r.get("status"),
                r.get("seconds", 0.0), r.get("error", ""),
            )
        )
    print(f"stress done: {len(results) - len(failed)}/{len(results)} requests ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
