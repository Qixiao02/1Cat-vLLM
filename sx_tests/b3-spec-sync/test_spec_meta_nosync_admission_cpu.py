# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_SPEC_META_NOSYNC admission + row-split helper (CPU only).

* The fast path is admitted only for speculative_config.method == "mtp" on
  the exact SM70 Qwen3.8 (Qwen4Exp) TP4 topology (the no-MTP fast-path
  contract with the MTP method); "0" switches it off; the no-MTP lane
  (speculative_config None) never takes it.
* SxSpecRows.spec / non_spec == boolean-mask indexing for random masks.

  /opt/venv/bin/python -m pytest -q sx_tests/b3-spec-sync/test_spec_meta_nosync_admission_cpu.py
"""

from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _spec_meta_common import (  # noqa: E402
    fake_vllm_config,
    make_gdn_builder,
    make_ple_builder,
)

import torch  # noqa: E402

from vllm.v1.attention.backends import gdn_attn  # noqa: E402


def _qwen38_config(method="mtp", tp=4, arch="Qwen4ExpForConditionalGeneration"):
    hf_text_config = NS(
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
    model_config = NS(
        architectures=[arch],
        multimodal_config=NS(language_model_only=True),
        dtype=torch.float16,
        hf_text_config=hf_text_config,
        max_model_len=65536,
    )
    cfg = fake_vllm_config(4, "align", full=True, method=method)
    cfg.model_config = model_config
    cfg.parallel_config = NS(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
    )
    return cfg


@pytest.fixture
def sm70(monkeypatch):
    import vllm.config.vllm as config_vllm

    state = {"sm70": True}
    monkeypatch.setattr(
        config_vllm,
        "_any_participating_device_is_capability",
        lambda cfg, cap: state["sm70"] and tuple(cap) == (7, 0),
    )
    monkeypatch.delenv("SX_OPT_SPEC_META_NOSYNC", raising=False)
    return state


def test_contract_admits_only_mtp_on_qwen38_tp4(sm70, monkeypatch):
    assert gdn_attn.sx_spec_meta_nosync_contract(_qwen38_config())
    assert gdn_attn.sx_spec_meta_nosync_admitted(_qwen38_config())
    assert gdn_attn.sx_spec_meta_nosync_contract(
        _qwen38_config(arch="Qwen4ExpForCausalLM")
    )
    for method in ("eagle", "dflash", "ngram", "draft_model"):
        assert not gdn_attn.sx_spec_meta_nosync_contract(_qwen38_config(method))
    assert not gdn_attn.sx_spec_meta_nosync_contract(_qwen38_config(method=None))
    assert not gdn_attn.sx_spec_meta_nosync_contract(_qwen38_config(tp=2))
    assert not gdn_attn.sx_spec_meta_nosync_contract(
        _qwen38_config(arch="Qwen3_5ForConditionalGeneration")
    )
    sm70["sm70"] = False
    assert not gdn_attn.sx_spec_meta_nosync_contract(_qwen38_config())
    sm70["sm70"] = True
    monkeypatch.setenv("SX_OPT_SPEC_META_NOSYNC", "0")
    assert not gdn_attn.sx_spec_meta_nosync_admitted(_qwen38_config())
    monkeypatch.setenv("SX_OPT_SPEC_META_NOSYNC", "1")
    assert gdn_attn.sx_spec_meta_nosync_admitted(_qwen38_config())


def test_contract_never_raises():
    assert not gdn_attn.sx_spec_meta_nosync_contract(NS())
    assert not gdn_attn.sx_spec_meta_nosync_contract(
        NS(speculative_config=NS(method="mtp"))
    )


@pytest.mark.parametrize("switch", ("1", "0"))
def test_builders_follow_admission(monkeypatch, switch):
    monkeypatch.setenv("SX_OPT_SPEC_META_NOSYNC", switch)
    monkeypatch.setattr(gdn_attn, "sx_spec_meta_nosync_contract", lambda cfg: True)
    expected = switch == "1"
    for mode in ("align", "none"):
        cfg = fake_vllm_config(4, mode, full=True)
        gdn = make_gdn_builder(
            monkeypatch, 4, mode, "cpu", full=True, nosync=None, vllm_config=cfg
        )
        ple = make_ple_builder(4, mode, "cpu", full=True, nosync=None, vllm_config=cfg)
        assert gdn._sx_spec_meta_nosync is expected
        assert ple._sx_spec_meta_nosync is expected


def test_no_mtp_lane_never_admitted(monkeypatch):
    """speculative_config None: builders keep the untouched legacy code."""
    monkeypatch.setattr(gdn_attn, "sx_spec_meta_nosync_contract", lambda cfg: True)
    cfg = fake_vllm_config(0, "align", full=True, method=None)
    gdn = make_gdn_builder(
        monkeypatch, 0, "align", "cpu", full=True, nosync=None, vllm_config=cfg
    )
    ple = make_ple_builder(0, "align", "cpu", full=True, nosync=None, vllm_config=cfg)
    assert gdn.use_spec_decode is False and gdn._sx_spec_meta_nosync is False
    assert ple.use_spec_decode is False and ple._sx_spec_meta_nosync is False


@pytest.mark.parametrize("rows", (1, 2, 3, 5, 8, 17, 24, 31))
def test_row_split_matches_boolean_indexing(rows):
    rng = random.Random(rows)
    for trial in range(64):
        mask = torch.tensor([rng.random() < 0.5 for _ in range(rows)])
        if trial == 0:
            mask[:] = True
        elif trial == 1:
            mask[:] = False
            mask[: max(1, rows // 2)] = True  # spec rows first (verify batch)
        num_spec = int(mask.sum())
        if num_spec == 0:
            continue
        prefix = gdn_attn.SxSpecRows.is_prefix(mask, num_spec)
        assert prefix == bool(torch.equal(mask, torch.arange(rows) < num_spec))
        order_cpu = None if prefix else gdn_attn.SxSpecRows.host_order(mask)
        split = gdn_attn.SxSpecRows(num_spec, prefix, order_cpu, order_cpu)
        table = torch.randint(0, 1000, (rows, 5), dtype=torch.int32)
        counts = torch.randint(1, 6, (rows,), dtype=torch.int32)
        flags = torch.rand(rows) < 0.5
        for tensor in (table, table[:, :3], counts, flags):
            assert torch.equal(split.spec(tensor), tensor[mask])
            assert torch.equal(split.non_spec(tensor), tensor[~mask])
