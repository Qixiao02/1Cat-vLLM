# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for sx_tests/b3-moe-verify (batch 3a, group "moe-verify").

Not a test module. The sibling test_*.py files import it first (pytest puts
this directory on sys.path; script mode runs from it).

Run inside image shixiang/1cat-vllm-v100:1.8.0-dev2-sm70main with the four
changed files of this group bind-mounted over the installed vllm package:

  vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py
  vllm/model_executor/layers/fused_moe/router/fused_topk_router.py
  vllm/model_executor/models/qwen2_moe.py
  vllm/distributed/device_communicators/custom_all_reduce.py

  /opt/venv/bin/python -m pytest -q sx_tests/b3-moe-verify/<file>

It provides: fake vLLM configs that satisfy (or break) the SM70 Qwen3.8 TP4
native-MTP lane contract, pure-decode / uniform-verify / mixed forward
contexts whose metadata mirrors the V2 runner (QSA flash metadata and PLE
short-conv metadata share one persistent query_start_loc view; GDN metadata
carries the spec counters only), the runner's query_start_loc tail semantics,
and a CUDA-graph timer.
"""

from __future__ import annotations

import contextlib
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
B2_MOE_DIR = REPO / "sx_tests" / "b2-moe-m32"


def _prefer_installed_vllm() -> None:
    """Keep ``import vllm`` on the installed (bind-mounted) package.

    ``python -m pytest`` from the source-tree root puts that root on sys.path,
    which would shadow the image's compiled vllm with the uncompiled sources.
    Drop it unless it carries compiled extensions.
    """
    if any((REPO / "vllm").glob("_C*.so")):
        return
    for entry in list(sys.path):
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            continue
        if resolved == REPO:
            sys.path.remove(entry)


_prefer_installed_vllm()

import torch  # noqa: E402

E, H, I, K = 512, 2560, 160, 10
LAYERS = 48
ON_SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)

# Uniform verify widths of the benchmark / tests: (label, tokens, q, live).
# 25 live rows: k = 4, B = 5 pads to the B = 6 (W30) graph of the lane-core
# request sizes, or to a W32 graph (kernel-level: live 25 of 32).
BENCH_WIDTHS = (
    ("W5", 5, 5, 5),
    ("W10", 10, 5, 10),
    ("W15", 15, 5, 15),
    ("W20", 20, 5, 20),
    ("W25(pad30)", 30, 5, 25),
    ("W25(pad32)", 32, 4, 25),
    ("W30", 30, 5, 30),
    ("W32", 32, 4, 32),
)


# ----------------------------------------------------------------------------
# configs
def qwen38_text_config(**over):
    fields = dict(
        hidden_size=2560,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        num_attention_heads=24,
        num_key_value_heads=2,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    fields.update(over)
    return NS(**fields)


def target_model_config(arch: str = "Qwen4ExpForCausalLM", **over):
    return NS(
        hf_text_config=qwen38_text_config(**over),
        architectures=[arch],
        multimodal_config=None,
        dtype=torch.float16,
    )


def draft_model_config():
    """The drafter's own model config (qwen4_exp_mtp); never the contract."""
    return NS(
        hf_text_config=NS(hidden_size=2560, num_hidden_layers=1),
        architectures=["Qwen4ExpMTPModel"],
        multimodal_config=None,
        dtype=torch.float16,
    )


def parallel_config(tp: int = 4, **over):
    fields = dict(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        enable_dbo=False,
        use_ubatching=False,
        ubatch_size=0,
    )
    fields.update(over)
    return NS(**fields)


def mtp_spec(k: int = 4, **over):
    fields = dict(
        method="mtp",
        num_speculative_tokens=k,
        use_qwen4_exp_mtp=lambda: True,
        num_speculative_state_tokens=lambda: k,
        parallel_drafting=False,
        rejection_sample_method="standard",
        target_model_config=target_model_config(),
        target_parallel_config=parallel_config(),
    )
    fields.update(over)
    return NS(**fields)


def lane_config(k: int = 4, *, draft: bool = False, spec=None, max_tokens=8192):
    """A vLLM config inside the MTP lane (target or draft construction)."""
    return NS(
        speculative_config=mtp_spec(k) if spec is None else spec,
        model_config=draft_model_config() if draft else target_model_config(),
        parallel_config=parallel_config(),
        scheduler_config=NS(max_num_batched_tokens=max_tokens),
    )


def nomtp_config(max_tokens=8192):
    return NS(
        speculative_config=None,
        model_config=target_model_config(),
        parallel_config=parallel_config(),
        scheduler_config=NS(max_num_batched_tokens=max_tokens),
    )


# ----------------------------------------------------------------------------
# forward contexts (V2 runner metadata semantics)
def write_live(buf: torch.Tensor, live_tokens: int, q: int = 1) -> None:
    """Runner semantics: cumsum of live requests (q rows each), every later
    entry = live token count (query_start_loc_np[num_reqs + 1:] = num_tokens)."""
    host = (torch.arange(buf.numel(), dtype=torch.int32) * q).clamp_(max=live_tokens)
    buf.copy_(host)


def qsl_buffer(device="cuda", capacity: int = 64) -> torch.Tensor:
    return torch.zeros(capacity + 1, device=device, dtype=torch.int32)


def decode_metadata(qsl_view, tokens: int):
    """Pure decode (no-MTP lane): QSA/PLE share the view, GDN has none."""
    shared = NS(max_query_len=1, query_start_loc=qsl_view)
    ple = NS(
        num_prefills=0, num_prefill_tokens=0, num_decodes=tokens,
        num_decode_tokens=tokens, num_reqs=tokens, query_start_loc=qsl_view,
        num_spec_decodes=0, num_spec_decode_tokens=0,
    )
    gdn = NS(
        num_prefills=0, num_prefill_tokens=0, num_decodes=tokens,
        num_decode_tokens=tokens, num_spec_decodes=0, num_spec_decode_tokens=0,
    )
    return {"layers.3.self_attn": shared, "layers.7.self_attn": shared,
            "layers.0.ple": ple, "layers.0.linear_attn": gdn}


def verify_metadata(qsl_view, tokens: int, q: int, *, spec_reqs=None,
                    spec_tokens=None, max_query=None, plain_decodes=0,
                    prefills=0):
    """Uniform MTP verify step (capture shape: every request has q rows).

    The keyword arguments break uniformity for rejection tests."""
    reqs = tokens // q if spec_reqs is None else spec_reqs
    spec_tok = reqs * q if spec_tokens is None else spec_tokens
    qsa = NS(max_query_len=q if max_query is None else max_query,
             query_start_loc=qsl_view)
    ple = NS(
        num_prefills=prefills, num_prefill_tokens=prefills * 7,
        num_decodes=plain_decodes, num_decode_tokens=plain_decodes,
        num_reqs=reqs, query_start_loc=qsl_view, num_spec_decodes=reqs,
        num_spec_decode_tokens=spec_tok, spec_query_len=q,
    )
    gdn = NS(
        num_prefills=prefills, num_prefill_tokens=prefills * 7,
        num_decodes=plain_decodes, num_decode_tokens=plain_decodes,
        num_spec_decodes=reqs, num_spec_decode_tokens=spec_tok,
    )
    return {"layers.3.self_attn": qsa, "layers.7.self_attn": qsa,
            "layers.0.ple": ple, "layers.0.linear_attn": gdn,
            "layers.1.linear_attn": gdn}


def mixed_metadata(qsl_view, q: int):
    """A prefill chunk plus verify rows (never admitted)."""
    return verify_metadata(qsl_view, 2 * q, q, prefills=1)


@contextlib.contextmanager
def forward_context(moe_module, metadata):
    ctx = NS(attn_metadata=metadata, additional_kwargs={})
    import pytest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe_module, "is_forward_context_available", lambda: True)
        mp.setattr(moe_module, "get_forward_context", lambda: ctx)
        yield ctx


@contextlib.contextmanager
def layer_attrs(layer, **values):
    old = {k: getattr(layer, k) for k in values}
    try:
        for k, v in values.items():
            setattr(layer, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(layer, k, v)


class EnvsProxy:
    """vllm.envs with selected values overridden (monkeypatched as moe.envs)."""

    def __init__(self, real, **overrides):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_over", overrides)

    def __getattr__(self, name):
        over = object.__getattribute__(self, "_over")
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, "_real"), name)


# ----------------------------------------------------------------------------
def graph_ms(fn, samples: int = 60, warmup: int = 5) -> float:
    """Median CUDA-event time of one replay of a graph that runs fn()."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return statistics.median(times)
