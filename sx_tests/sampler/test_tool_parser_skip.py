# SPDX-License-Identifier: Apache-2.0
"""cpu-8: a streamed chat request without tools (or with tool_choice "none")
must not run the configured tool parser. Its content must pass through
exactly as it would without a tool parser.

CPU only, no GPU or model:
    /opt/venv/bin/python -m pytest -q sx_tests/sampler/test_tool_parser_skip.py

The serving change (vllm/entrypoints/openai/chat_completion/serving.py) sets
``parser.tool_parser = None`` on the per-stream parser instances when
SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS is on (the default),
``not request.tools or request.tool_choice == "none"``, and the request is on
neither the Mistral grammar path nor the harmony path. These tests check
three things. The public setter detaches the tool parser. A detached
DelegatingParser streams the deltas unchanged and never calls the tool
parser. The switch reads the environment as documented. For an end-to-end
SSE diff against a live server, see e2e_stream_tool_parser.py.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


class _RecordingToolParser:
    supports_required_and_named = True

    def __init__(self):
        self.calls = 0

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, *a):
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage

        self.calls += 1
        return DeltaMessage(content=delta_text)


def _parser_with_fake_tools():
    from vllm.parser.abstract_parser import DelegatingParser

    parser = DelegatingParser(tokenizer=None)
    tool = _RecordingToolParser()
    parser.tool_parser = tool
    return parser, tool


DELTAS = ['{"a"', ": 1", ', "b": "<x>"', "}"]


def _stream(parser, request):
    out = []
    for d in DELTAS:
        msg = parser.parse_delta(d, [1], request, prompt_token_ids=[0])
        out.append(None if msg is None else (msg.content, msg.tool_calls))
    return out


def test_detached_tool_parser_passes_content_through():
    request = SimpleNamespace(tool_choice="none", tools=None)
    parser, tool = _parser_with_fake_tools()
    attached = _stream(parser, request)
    assert tool.calls == len(DELTAS)

    parser, tool = _parser_with_fake_tools()
    parser.tool_parser = None  # what serving.py does for no-tools requests
    detached = _stream(parser, request)
    assert tool.calls == 0
    assert [c for c, _ in detached] == DELTAS
    assert all(not tc for _, tc in detached)
    # The fake tool parser echoes content, so both streams must be identical.
    assert detached == attached


def test_switch_reads_environment(monkeypatch):
    from vllm.entrypoints.openai.chat_completion import serving

    monkeypatch.delenv("SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS", raising=False)
    assert serving._sx_skip_tool_parser_without_tools()
    monkeypatch.setenv("SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS", "0")
    assert not serving._sx_skip_tool_parser_without_tools()
    monkeypatch.setenv("SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS", "1")
    assert serving._sx_skip_tool_parser_without_tools()


def test_serving_detaches_only_without_tools():
    """Source-level guard: the detach sits right after the per-stream parser
    construction and is conditioned on the request's tools/tool_choice."""
    from vllm.entrypoints.openai.chat_completion import serving

    src = inspect.getsource(serving.OpenAIServingChat.chat_completion_stream_generator)
    i_build = src.index("self.parser_cls(")
    i_cond = src.index("_sx_skip_tool_parser_without_tools()")
    i_detach = src.index("p.tool_parser = None")
    assert i_build < i_cond < i_detach
    cond = src[i_cond:i_detach]
    assert "not request.tools" in cond and 'request.tool_choice == "none"' in cond
    # The Mistral grammar branch asserts ``tool_parser is not None`` and is
    # taken even for requests without tools; harmony keeps its own flow.
    assert "not is_mistral_grammar_path" in cond
    assert "not self.use_harmony" in cond
    # The detach must come after is_mistral_grammar_path is computed.
    assert src.index("is_mistral_grammar_path = ") < i_cond


def test_mistral_grammar_branch_still_requires_tool_parser():
    """Guards the reason for the exclusion above: if the Mistral grammar
    branch ever stops asserting a tool parser, this test can be relaxed."""
    from vllm.entrypoints.openai.chat_completion import serving

    src = inspect.getsource(serving.OpenAIServingChat.chat_completion_stream_generator)
    i_branch = src.index("elif is_mistral_grammar_path:")
    assert "assert tool_parser is not None" in src[i_branch : i_branch + 600]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
