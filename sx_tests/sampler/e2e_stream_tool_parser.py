# SPDX-License-Identifier: Apache-2.0
"""cpu-8 end-to-end check against a live OpenAI-compatible server (stdlib only).

Streams four greedy, seeded chat requests and prints, for each: the joined
content, the tool calls, the finish reason, the chunk count and a SHA-256
of the content.
  1. json_schema, no tools
  2. plain text, no tools
  3. tools with tool_choice "auto"
  4. tools with tool_choice "none"

Run once with SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS=0 on the server and once
with the default (1), then diff the two outputs. Requests 1 and 2 must
match: the JSON grammar forbids <tool_call> markup, and plain answers
contain none. Request 3 still goes through the tool parser and must match.
Request 4 matches unless the model ignores tool_choice "none" and emits
tool-call markup: the old stream turned that into tool_calls, the new one
returns it as content (the non-streaming path already did so).

    python sx_tests/sampler/e2e_stream_tool_parser.py \
        --url http://127.0.0.1:8000 --model <served-model-name> [--api-key KEY]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["name", "score"],
}


def stream(url, model, api_key, body):
    body = dict(body, model=model, stream=True, temperature=0.0, seed=7, max_tokens=128)
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    content, tool_calls, finish, chunks = [], [], None, 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunks += 1
            payload = json.loads(line[6:])
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls.append(delta["tool_calls"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    text = "".join(content)
    return {
        "content": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "tool_call_chunks": len(tool_calls),
        "finish_reason": finish,
        "chunks": chunks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()
    msg = [{"role": "user", "content": "Give a name and a score for a fictional player."}]
    cases = {
        "json_schema_no_tools": {
            "messages": msg,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "player", "schema": SCHEMA},
            },
        },
        "plain_no_tools": {"messages": msg},
        "tools_auto": {
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
        },
        "tools_none": {
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        },
    }
    for name, body in cases.items():
        print(name, json.dumps(stream(args.url, args.model, args.api_key, body), ensure_ascii=False))


if __name__ == "__main__":
    main()
