# SPDX-License-Identifier: Apache-2.0
"""design_1 [MTP-4] on the GPU: batch-1 QSA host metadata in the MTP lane.

The MTP-lane metadata builder attaches exact host query starts and host
sequence lengths in which every request that may carry an optimistic upper
bound (<= 1 + k query rows: verify rows, decodes) is -1. With it:
  * QSA_HOST_BOUND: a single-request prompt chunk (target + draft prefill)
    selects bit-identically to the dev2 device path, without a host sync
    (torch.cuda.set_sync_debug_mode("error"));
  * QSA_MIXED_CUBLAS: in a mixed step of k=4 verify requests + a prompt
    chunk, the prompt rows equal the request run alone through dev2 (the
    single-request cuBLAS path) and the verify rows equal dev2's mixed
    result (unchanged Triton scorer, full-batch tile profile);
    optimistic verify lengths never reach a planner;
  * QSA_MIXED_GROUPS: in a page4 mixed step, prompt rows equal the request
    run alone through dev2 and the 5-row verify rows equal dev2 row-wise XQA
    on those rows (per-row independent).

GPU: ONE V100 (SM70).
  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-qsa-mtp/test_host_metadata_mtp.py
"""

from __future__ import annotations

import contextlib
import os
import statistics
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.is_sm70(), reason="requires SM70 GPU")


def _masked_seq_lens(layout: C.VerifyLayout, k: int, optimism: int = 3) -> torch.Tensor:
    """What the MTP-lane builder attaches: optimistic upper bounds for verify
    requests (as the async scheduler produces) masked to -1."""
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_model

    upper = torch.tensor(
        [
            seq_len + (optimism if rows <= k + 1 else 0)
            for rows, seq_len in zip(layout.request_rows, layout.seq_lens)
        ],
        dtype=torch.int32,
    )
    starts = torch.tensor(layout.starts, dtype=torch.int32)
    return qsa_model._sx_qsa_exact_prefill_seq_lens(starts, upper, k + 1)


@contextlib.contextmanager
def _no_host_sync():
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("context", (2048, 8192, 32768))
@pytest.mark.parametrize("chunk", (784, 1568, 7840))
def test_single_request_prefill_host_bound(context, chunk):
    if chunk > context:
        pytest.skip("chunk longer than the context")
    new, dev2 = C.new_ops(), C.dev2_ops()
    layout = C.verify_layout([chunk], [context])
    case = C.build_indexer_case(layout, seed=chunk + context)
    lane = C.mtp_lane()
    seq_lens_cpu = _masked_seq_lens(layout, k=4)
    assert seq_lens_cpu.tolist() == [context]  # a prompt chunk stays exact
    expected = C.run_select(dev2, case)  # device-side .item() bound
    kwargs = dict(
        query_start_loc_cpu=case.query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        sx_mtp_lane=lane,
    )
    C.run_select(new, case, **kwargs)  # JIT warmup outside the sync check
    out = torch.empty_like(expected)
    with _no_host_sync():
        C.run_select(new, case, out=out, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(expected, out), C.first_mismatch(expected, out)


@pytest.mark.parametrize("num_verify,q_len", [(1, 5), (8, 5), (16, 3), (24, 5)])
def test_mixed_step_cublas(num_verify, q_len):
    new, dev2 = C.new_ops(), C.dev2_ops()
    generator = torch.Generator().manual_seed(num_verify * 10 + q_len)
    verify_seq = C.jittered_contexts(generator, num_verify, 4096)
    prompt_rows, prompt_seq = 1568, 8192
    layout = C.verify_layout(
        [q_len] * num_verify + [prompt_rows], verify_seq + [prompt_seq]
    )
    case = C.build_indexer_case(layout, seed=7 + num_verify)
    seq_lens_cpu = _masked_seq_lens(layout, k=q_len - 1)
    assert seq_lens_cpu.tolist() == [-1] * num_verify + [prompt_seq]
    lane = C.mtp_lane()
    cublas_spy = C.CallSpy(new._qsa_gather_single_request_keys)
    with C.patched(new, _qsa_gather_single_request_keys=cublas_spy):
        actual = C.run_select(
            new,
            case,
            query_start_loc_cpu=case.query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            sx_mtp_lane=lane,
        )
    assert cublas_spy.calls == 1  # only the prompt chunk is scored by cuBLAS
    verify_rows = num_verify * q_len
    # Verify rows: dev2 mixed result (same Triton scorer and tile profile).
    mixed = C.run_select(dev2, case)
    assert torch.equal(mixed[:verify_rows], actual[:verify_rows])
    # Prompt rows: the request run alone through dev2 (single-request cuBLAS).
    alone_layout = C.verify_layout([prompt_rows], [prompt_seq])
    alone = C.IndexerCase(
        layout=alone_layout,
        q=case.q[verify_rows:].contiguous(),
        cache=case.cache,
        table=case.table[num_verify:].contiguous(),
        token_to_req=torch.zeros(prompt_rows, dtype=torch.int32, device="cuda"),
        positions=case.positions[verify_rows:].contiguous(),
        seq_lens=case.seq_lens[num_verify:].contiguous(),
        query_start_loc_cpu=torch.tensor([0, prompt_rows], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([prompt_seq], dtype=torch.int32),
    )
    expected_prompt = C.run_select(dev2, alone)
    torch.cuda.synchronize()
    assert torch.equal(expected_prompt, actual[verify_rows:]), C.first_mismatch(
        expected_prompt, actual[verify_rows:]
    )


@pytest.mark.parametrize(
    "num_verify,q_len,prompt_rows",
    [(0, 5, 1568), (0, 5, 7840), (8, 5, 1568), (24, 2, 784), (12, 8, 3136)],
)
def test_host_bound_check_mode_agrees(num_verify, q_len, prompt_rows):
    """SX_OPT_QSA_HOST_BOUND_CHECK (debug: one sync per request) raises if a
    width planned from the MTP-lane host lengths differs from the device
    value, for the prompt chunk alone (QSA_HOST_BOUND) and inside mixed
    verify steps (QSA_MIXED_CUBLAS), k in {1, 4, 7}."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    generator = torch.Generator().manual_seed(num_verify * 100 + q_len + prompt_rows)
    verify_seq = C.jittered_contexts(generator, num_verify, 4096)
    prompt_seq = 8192 + prompt_rows
    layout = C.verify_layout(
        [q_len] * num_verify + [prompt_rows], verify_seq + [prompt_seq]
    )
    case = C.build_indexer_case(layout, seed=31 + num_verify + prompt_rows)
    seq_lens_cpu = _masked_seq_lens(layout, k=q_len - 1)
    assert seq_lens_cpu.tolist() == [-1] * num_verify + [prompt_seq]
    with C.patched(new, _SX_OPT_QSA_HOST_BOUND_CHECK=True):
        actual = C.run_select(
            new,
            case,
            query_start_loc_cpu=case.query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            sx_mtp_lane=C.mtp_lane(),
        )
    torch.cuda.synchronize()
    if num_verify == 0:
        # Single request: bitwise equal to dev2's device-bound path.
        assert torch.equal(C.run_select(dev2, case), actual)


def test_mixed_step_masked_lengths_keep_device_path():
    """If verify requests ever qualified for the cuBLAS planner (tiny
    VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS), the masked lengths keep the whole
    batch on dev2's device path (bit for bit)."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    generator = torch.Generator().manual_seed(11)
    layout = C.verify_layout(
        [5] * 8 + [1568], C.jittered_contexts(generator, 8, 4096) + [8192]
    )
    case = C.build_indexer_case(layout, seed=11)
    seq_lens_cpu = _masked_seq_lens(layout, k=4)
    with C.patched(new, _SM70_INDEXER_CUBLAS_MIN_ROWS=4):
        actual = C.run_select(
            new,
            case,
            query_start_loc_cpu=case.query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            sx_mtp_lane=C.mtp_lane(),
        )
    expected = C.run_select(dev2, case)
    torch.cuda.synchronize()
    assert torch.equal(expected, actual)


def test_mixed_step_page4_groups():
    """QSA_MIXED_GROUPS with 5-row verify requests + a 784-row prompt chunk."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        pytest.skip("Flash-V100 is not importable")
    if not new._qsa_grouped_page4_supported(flash_attn_v100_cuda, "auto"):
        pytest.skip("grouped page4 ABI unavailable")
    generator = torch.Generator().manual_seed(21)
    num_verify, q_len, prompt_rows, prompt_seq = 8, 5, 784, 8192
    layout = C.verify_layout(
        [q_len] * num_verify + [prompt_rows],
        C.jittered_contexts(generator, num_verify, 4096) + [prompt_seq],
    )
    index_case = C.build_indexer_case(layout, seed=21)
    indices = C.run_select(dev2, index_case)
    case = C.build_attention_case(layout, indices, seed=21)
    segments_spy = C.CallSpy(new._qsa_sparse_paged_attention_sm70_page4_segments)
    actual = torch.full_like(case.q, float("nan"))
    with C.patched(new, _qsa_sparse_paged_attention_sm70_page4_segments=segments_spy):
        C.run_attention(
            new,
            case,
            actual,
            query_start_loc_cpu=torch.tensor(layout.starts, dtype=torch.int32),
            sx_mtp_lane=C.mtp_lane(),
        )
    assert segments_spy.calls == 1
    verify_rows = num_verify * q_len

    def sub_case(rows: slice, requests: slice, remap: bool):
        token_to_req = case.token_to_req[rows].clone()
        if remap:
            token_to_req -= requests.start
        return C.AttentionCase(
            layout=layout,
            q=case.q[rows].contiguous(),
            gate=case.gate[rows].contiguous(),
            kv=case.kv,
            k=case.k,
            v=case.v,
            indices=case.indices[rows].contiguous(),
            table=case.table[requests].contiguous(),
            token_to_req=token_to_req,
            positions=case.positions[rows].contiguous(),
            seq_lens=case.seq_lens[requests].contiguous(),
            page_size=case.page_size,
        )

    # Prompt rows == the request run alone through dev2 (784 rows = 98 groups).
    prompt = sub_case(slice(verify_rows, None), slice(num_verify, None), True)
    expected_prompt = torch.full_like(prompt.q, float("nan"))
    C.run_attention(dev2, prompt, expected_prompt)
    # Verify rows == dev2 row-wise XQA on those rows (per-row independent).
    verify = sub_case(slice(0, verify_rows), slice(0, None), False)
    expected_verify = torch.full_like(verify.q, float("nan"))
    dev2._qsa_sparse_paged_attention_sm70_xqa_page4_batch(
        verify.q,
        verify.k,
        verify.v,
        verify.indices,
        verify.table,
        verify.token_to_req,
        verify.positions,
        verify.seq_lens,
        expected_verify,
        "auto",
        1.0,
        1.0,
        flash_attn_v100_cuda,
    )
    dev2._qsa_output_gate(expected_verify, verify.gate.view_as(verify.q))
    torch.cuda.synchronize()
    assert C.bitwise_equal(expected_prompt, actual[verify_rows:]), C.first_mismatch(
        expected_prompt, actual[verify_rows:]
    )
    assert C.bitwise_equal(expected_verify, actual[:verify_rows]), C.first_mismatch(
        expected_verify, actual[:verify_rows]
    )


@pytest.mark.skipif(C.no_bench(), reason="SX_TEST_NO_BENCH")
@pytest.mark.parametrize("context", (8192, 32768))
def test_prefill_host_bound_microbench(context):
    """Host time to enqueue one MTP-lane prefill step's 13 QSA selections
    (12 target layers + the draft layer, 784-row chunk) while the GPU is
    still busy: dev2 blocks on .item() per layer, the host bound does not."""
    new, dev2 = C.new_ops(), C.dev2_ops()
    layout = C.verify_layout([C.SCHED_BLOCK], [context])
    cases = [C.build_indexer_case(layout, seed=200 + layer) for layer in range(13)]
    seq_lens_cpu = _masked_seq_lens(layout, k=4)
    lane = C.mtp_lane()
    busy = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    def keep_gpu_busy():
        for _ in range(16):
            busy @ busy

    def run(ops, host: bool):
        for case in cases:
            kwargs = {}
            if host:
                kwargs = dict(
                    query_start_loc_cpu=case.query_start_loc_cpu,
                    seq_lens_cpu=seq_lens_cpu,
                    sx_mtp_lane=lane,
                )
            C.run_select(ops, case, **kwargs)

    results = {}
    for name, ops, host in (("dev2_item", dev2, False), ("mtp_host", new, True)):
        run(ops, host)  # warmup / JIT
        enqueue, wall = [], []
        for _ in range(7):
            torch.cuda.synchronize()
            keep_gpu_busy()
            start = time.perf_counter()
            run(ops, host)
            issued = time.perf_counter()
            torch.cuda.synchronize()
            done = time.perf_counter()
            enqueue.append((issued - start) * 1e3)
            wall.append((done - start) * 1e3)
        results[f"{name}_enqueue_ms"] = statistics.median(enqueue)
        results[f"{name}_wall_ms"] = statistics.median(wall)
    C.report(f"prefill select x13 layers, 784 rows, ctx={context}", results)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
