# SPDX-License-Identifier: Apache-2.0
"""Host-side planners and the metadata-builder hook (no GPU kernels).

Needs the deployed image's vllm import; no GPU kernels are launched.
From the source-tree root:
  /opt/venv/bin/python -m pytest -q sx_tests/qsa/test_qsa_host_metadata.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from _qsa_test_utils import new_ops, patched


def qsl(*values):
    return torch.tensor(values, dtype=torch.int32)


def test_page4_segments_split_decodes_and_remainders():
    ops = new_ops()
    plan = ops._plan_qsa_page4_request_segments
    # 3 decodes + one 784-row prefill.
    segments = plan(qsl(0, 1, 2, 3, 787), 787)
    assert segments == [(0, 3, False), (3, 787, True)]
    assert not ops._qsa_page4_segments_match_baseline(segments, 787)
    # Two prefills with a remainder in between; decodes first.
    segments = plan(qsl(0, 1, 301, 1085), 1085)
    assert segments == [(0, 1, False), (1, 297, True), (297, 301, False),
                        (301, 1085, True)]
    # Padded tail rows are row-wise.
    assert plan(qsl(0, 16), 20) == [(0, 16, True), (16, 20, False)]
    # Only short requests (< 8 rows each) plus padding: everything row-wise.
    assert plan(qsl(0, 1, 2, 9, 16), 64) == [(0, 64, False)]
    # A 16-row request between decodes: two whole groups of its own.
    assert plan(qsl(0, 1, 17, 18), 18) == [(0, 1, False), (1, 17, True),
                                           (17, 18, False)]


def test_page4_segments_baseline_equivalence():
    ops = new_ops()
    plan = ops._plan_qsa_page4_request_segments
    same = ops._qsa_page4_segments_match_baseline
    for rows in (64, 450, 784, 3136, 8192):
        assert same(plan(qsl(0, rows), rows), rows)
    # Several prefills whose lengths are multiples of 8 keep the old split.
    assert same(plan(qsl(0, 784, 1568, 3136), 3136), 3136)


def test_page4_segments_reject_bad_metadata():
    ops = new_ops()
    plan = ops._plan_qsa_page4_request_segments
    assert plan(None, 10) is None
    assert plan(qsl(1, 10), 10) is None
    assert plan(qsl(0, 11), 10) is None
    assert plan(qsl(0, 5, 3), 10) is None
    assert plan(qsl(0), 10) is None
    if torch.cuda.is_available():
        assert plan(qsl(0, 10).cuda(), 10) is None  # never reads device data


def test_host_max_visible():
    ops = new_ops()
    table = torch.zeros(1, 168, dtype=torch.int32)
    fn = ops._qsa_host_max_visible
    assert fn(table, 784, 4, qsl(0, 784), qsl(8192)) == 2048
    assert fn(table, 792, 4, qsl(0, 784), qsl(8195)) == 2048  # padded rows
    assert fn(table, 784, 4, qsl(0, 784), None) is None
    assert fn(table, 784, 4, None, qsl(8192)) is None
    assert fn(table, 784, 4, qsl(0, 784), qsl(100)) is None  # seq < rows
    assert fn(table, 784, 4, qsl(0, 800), qsl(8192)) is None  # rows mismatch
    assert fn(table.expand(2, -1), 784, 4, qsl(0, 784), qsl(8192)) is None
    with patched(ops, _SX_OPT_QSA_HOST_BOUND=False):
        assert fn(table, 784, 4, qsl(0, 784), qsl(8192)) is None


def test_indexer_cublas_segment_planner(monkeypatch):
    ops = new_ops()
    monkeypatch.setattr(
        ops.current_platform, "is_device_capability", lambda capability: capability == 70
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    rows = 23 + 784
    q = torch.empty(rows, 4, 128, dtype=torch.float16)
    cache = torch.empty(8, 196, 1, 128, dtype=torch.float16)
    table = torch.zeros(24, 168, dtype=torch.int32)
    starts = qsl(*range(24), rows)
    seq = torch.tensor([2000] * 23 + [8192], dtype=torch.int32)
    segments = ops._plan_qsa_indexer_cublas_segments(
        q, cache, table, starts, seq, 168 * 196, 512, 4
    )
    assert segments == [(0, 23, None, 0), (23, rows, 23, 2048)]
    # A 2K-context chunk does not reach the cuBLAS work threshold.
    seq_short = torch.tensor([2000] * 23 + [2048], dtype=torch.int32)
    assert ops._plan_qsa_indexer_cublas_segments(
        q, cache, table, starts, seq_short, 168 * 196, 512, 4
    ) is None
    with patched(ops, _SX_OPT_QSA_MIXED_CUBLAS=False):
        assert ops._plan_qsa_indexer_cublas_segments(
            q, cache, table, starts, seq, 168 * 196, 512, 4
        ) is None
    # Metadata that does not describe this batch is ignored.
    assert ops._plan_qsa_indexer_cublas_segments(
        q, cache, table[:23], starts, seq, 168 * 196, 512, 4
    ) is None


def test_decode_gates():
    ops = new_ops()
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)):
        pytest.skip("gate checks read the device capability")
    assert ops._use_sm70_qsa_two_warp_partial(32, 6, 256)
    assert not ops._use_sm70_qsa_two_warp_partial(33, 6, 256)
    with patched(ops, _SX_OPT_QSA_TWO_WARP32=False):
        assert not ops._use_sm70_qsa_two_warp_partial(17, 6, 256)
        assert ops._use_sm70_qsa_two_warp_partial(16, 6, 256)
    for rows in (1, 2, 17, 24, 32):
        for page in (16, 400, 784):
            q = torch.empty(rows, 6, 256, dtype=torch.float16)
            k = torch.empty(1000, page, 1, 256, dtype=torch.float16)
            indices = torch.empty(rows, 2051, dtype=torch.int32)
            assert ops._use_sm70_qsa_resolved_indices(q, k, indices, "auto")
            assert not ops._use_sm70_qsa_resolved_indices(q, k, indices, "fp8_e4m3")
    q = torch.empty(33, 6, 256, dtype=torch.float16)
    k = torch.empty(1000, 784, 1, 256, dtype=torch.float16)
    assert not ops._use_sm70_qsa_resolved_indices(
        q, k, torch.empty(33, 2051, dtype=torch.int32), "auto"
    )


def test_metadata_builder_attaches_host_copies(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

    monkeypatch.setattr(
        FlashAttentionMetadataBuilder,
        "build",
        lambda self, common_prefix_len, common, fast_build=False: SimpleNamespace(),
    )
    builder = object.__new__(qsa_model.Qwen4ExpQSAMetadataBuilder)
    builder.vllm_config = SimpleNamespace(speculative_config=None)
    common = SimpleNamespace(
        query_start_loc_cpu=qsl(0, 1, 785, 785),  # padded request slot
        seq_lens_cpu_upper_bound=qsl(100, 8192),
        num_reqs=2,
    )
    metadata = builder.build(0, common)
    starts, seq_lens = qsa_model._sx_qsa_host_metadata(metadata)
    assert starts.tolist() == [0, 1, 785]
    assert seq_lens.tolist() == [100, 8192]

    spec_builder = object.__new__(qsa_model.Qwen4ExpQSAMetadataBuilder)
    spec_builder.vllm_config = SimpleNamespace(speculative_config=object())
    assert qsa_model._sx_qsa_host_metadata(spec_builder.build(0, common)) == (
        None,
        None,
    )
    monkeypatch.setattr(qsa_model, "_SX_OPT_QSA_HOST_METADATA", False)
    fresh = object.__new__(qsa_model.Qwen4ExpQSAMetadataBuilder)
    fresh.vllm_config = SimpleNamespace(speculative_config=None)
    assert qsa_model._sx_qsa_host_metadata(fresh.build(0, common)) == (None, None)
