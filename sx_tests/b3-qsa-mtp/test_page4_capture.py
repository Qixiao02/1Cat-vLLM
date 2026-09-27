# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_QSA_MTP_PAGE4_CAPTURE: QSA XQA/grouped page4 under CUDA-graph capture.

Phase 3 could not capture the k4 verify graph at 24 requests x 5 = 120
tokens: _qsa_xqa_page4_workspace() created its partition-count tensor with
torch.tensor(..., device=cuda) (a synchronous host copy) the first time the
capture stream needed a workspace, and CUDA aborted the capture with
"operation not permitted when stream is capturing". The fix keeps a cached
read-only device constant, gives captured launches their own never-freed
workspaces and, in the MTP lane, reserves them outside capture.

Asserts (GPU, ONE V100, image 1.8.0-dev2 + the patched files):
  * the dev2 copy really fails to capture a 120-token verify batch whose
    rows reach XQA (subprocess, so the aborted capture cannot poison this
    process);
  * the new path captures 120 tokens with grouped page4 on (all 15 groups)
    and off (all 120 rows row-wise XQA), and the production failure widths
    with an XQA remainder (115 = 112 grouped + 3, 110, 105, ..., 65), in
    vLLM's descending capture order;
  * with the MTP-lane reserve no capture allocates a workspace or partition
    constant (dict entries identical before/after, nothing retired);
  * without the reserve (non-MTP callers) capture still works, growth during
    capture retires (never frees) the workspace an earlier graph uses, and
    that graph still replays bit-exactly after the allocator reused memory;
  * every replay (changed q / indices / positions / sequence lengths,
    poisoned output) is bit-identical to the dev2 eager path;
  * eager calls keep the per-stream workspace; eager output == dev2.

  /opt/venv/bin/python -m pytest -q -s sx_tests/b3-qsa-mtp/test_page4_capture.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mtp_common as C  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not C.is_sm70(), reason="requires SM70 GPU")


def _flash_v100_page4():
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        pytest.skip("Flash-V100 is not importable")
    if not hasattr(flash_attn_v100_cuda, "decode_paged_xqa_fwd"):
        pytest.skip("Flash-V100 build has no decode_paged_xqa_fwd")
    return flash_attn_v100_cuda


def _grouped_available() -> bool:
    ops = C.new_ops()
    cuda = _flash_v100_page4()
    return bool(ops._SM70_QSA_GROUPED_PAGE4 and ops._qsa_grouped_page4_supported(cuda, "auto"))


def _verify_attention_case(num_reqs: int, q_len: int = 5, seed: int = 0):
    """Attention inputs of a k = q_len - 1 verify batch (dev2 selections)."""
    _, case = C.verify_case(num_reqs, q_len, 8192, seed=seed)
    return case


def _refresh(case, seed: int) -> None:
    """New inputs into the fixed graph buffers (same shapes)."""
    num_reqs = case.table.shape[0]
    q_len = case.q.shape[0] // num_reqs
    fresh = _verify_attention_case(num_reqs, q_len, seed=seed)
    case.q.copy_(fresh.q)
    case.gate.copy_(fresh.gate)
    case.indices.copy_(fresh.indices)
    case.table.copy_(fresh.table)
    case.positions.copy_(fresh.positions)
    case.seq_lens.copy_(fresh.seq_lens)


def _check_replay(graph, case, out, seeds) -> None:
    """Replay with fresh inputs; compare with dev2 eager on the same route
    (dev2 follows the live module's grouped-page4 setting)."""
    dev2 = C.dev2_ops()
    grouped = C.new_ops()._SM70_QSA_GROUPED_PAGE4
    for seed in seeds:
        _refresh(case, seed)
        out.fill_(float("nan"))
        graph.replay()
        expected = torch.full_like(out, float("nan"))
        with C.patched(dev2, _SM70_QSA_GROUPED_PAGE4=grouped):
            C.run_attention(dev2, case, expected)
        torch.cuda.synchronize()
        assert C.bitwise_equal(expected, out), (seed, C.first_mismatch(expected, out))


def _graph_state(ops):
    def ptrs(entries):
        return {
            key: tuple(
                item.data_ptr() if isinstance(item, torch.Tensor) else item
                for item in value
            )
            for key, value in entries.items()
        }

    return (
        ptrs(ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES),
        ptrs(ops._SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES),
        {key: value.data_ptr() for key, value in ops._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS.items()},
        len(ops._SM70_QSA_PAGE4_GRAPH_RETIRED),
    )


@pytest.fixture(autouse=True)
def _fresh_graph_state():
    """Every test starts without graph workspaces or reservations."""
    ops = C.new_ops()
    ops._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES.clear()
    ops._SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES.clear()
    ops._SM70_QSA_PAGE4_GRAPH_RESERVED.clear()
    ops._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS.clear()
    ops._SM70_QSA_XQA_PAGE4_WORKSPACES.clear()
    ops._SM70_QSA_GROUPED_PAGE4_WORKSPACES.clear()
    yield
    torch.cuda.synchronize()


_DEV2_REPRO = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {here!r})
    import _mtp_common as C
    import torch
    dev2 = C.dev2_ops()
    dev2._SM70_QSA_GROUPED_PAGE4 = {grouped!r}
    _, case = C.verify_case({num_reqs}, 5, 8192, seed=0)
    out = torch.empty_like(case.q)
    try:
        C.capture(lambda: C.run_attention(dev2, case, out))
    except Exception as exc:  # the dev2 failure
        print("DEV2_CAPTURE_FAILED", type(exc).__name__, str(exc)[:300], flush=True)
        sys.exit(3)
    print("DEV2_CAPTURE_OK", flush=True)
    """
)


@pytest.mark.parametrize(
    "num_reqs,grouped", [(24, False), (23, True)], ids=["120-xqa", "115-remainder"]
)
def test_dev2_capture_fails(num_reqs, grouped):
    """Reproduce the phase-3 failure with the dev2 module (subprocess)."""
    _flash_v100_page4()
    if grouped and not _grouped_available():
        pytest.skip("grouped page4 ABI unavailable: 115 rows would not split")
    script = _DEV2_REPRO.format(
        here=os.path.dirname(os.path.abspath(__file__)),
        grouped=grouped,
        num_reqs=num_reqs,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout[-2000:], result.stderr[-2000:])
    assert result.returncode == 3 and "DEV2_CAPTURE_FAILED" in result.stdout, (
        "dev2 capture was expected to fail (torch.tensor during capture)"
    )
    # The CUDA error itself (not the marker) must be the capture violation.
    message = result.stdout.split("DEV2_CAPTURE_FAILED", 1)[1].lower()
    assert "captur" in message or "not permitted" in message, message


@pytest.mark.parametrize("grouped", [False, True], ids=["xqa-all-rows", "grouped"])
def test_capture_120_tokens_reserved(grouped):
    """24 requests x 5 verify rows = 120 tokens, MTP lane with the reserve."""
    _flash_v100_page4()
    new = C.new_ops()
    if grouped and not _grouped_available():
        pytest.skip("grouped page4 ABI unavailable")
    lane = C.mtp_lane(page4_graph_rows=120)
    with C.patched(new, _SM70_QSA_GROUPED_PAGE4=grouped):
        case = _verify_attention_case(24, seed=10)
        out = torch.empty_like(case.q)
        # Eager call first (the profile run / capture warmup): it reserves.
        C.run_attention(new, case, out, sx_mtp_lane=lane)
        torch.cuda.synchronize()
        before = _graph_state(new)
        if grouped:
            assert before[1], "grouped graph workspace not reserved"
        else:
            assert before[0] and before[0][next(iter(before[0]))][0] >= 120
        assert before[2], "partition constant not created outside capture"
        graph = C.capture(lambda: C.run_attention(new, case, out, sx_mtp_lane=lane))
        after = _graph_state(new)
        assert before == after, "capture allocated or replaced a page4 workspace"
        _check_replay(graph, case, out, seeds=(11, 12, 13))


def test_capture_descending_widths_reserved():
    """vLLM's FULL capture order for k4 B24..13 (120, 115, ..., 65): every
    width with an XQA remainder captures from the reserved workspaces."""
    _flash_v100_page4()
    new = C.new_ops()
    lane = C.mtp_lane(page4_graph_rows=120)
    # The startup profile run: one eager page4 call reserves.
    first = _verify_attention_case(24, seed=19)
    C.run_attention(new, first, torch.empty_like(first.q), sx_mtp_lane=lane)
    torch.cuda.synchronize()
    before = _graph_state(new)
    assert before[2], "partition constant not created outside capture"
    pool = torch.cuda.graph_pool_handle()  # vLLM shares one pool
    captured = []
    for num_reqs in range(24, 12, -1):
        case = _verify_attention_case(num_reqs, seed=20 + num_reqs)
        out = torch.empty_like(case.q)
        graph = C.capture(
            lambda case=case, out=out: C.run_attention(
                new, case, out, sx_mtp_lane=lane
            ),
            pool=pool,
        )
        assert _graph_state(new) == before, (
            "workspace changed during capture",
            num_reqs,
        )
        captured.append((graph, case, out))
    # Replays after all captures (earlier graphs must still be intact).
    for graph, case, out in captured:
        _check_replay(graph, case, out, seeds=(30,))


# Verify-graph request counts of the MTP lane (vllm/config/vllm.py
# _SX_MTP_LANE_GRAPH_REQUEST_SIZES) at max_num_seqs 24.
_LANE_GRAPH_REQS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)


@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_capture_lane_graph_sizes_reserved(k):
    """The FULL verify widths the lane really captures for k (every width
    >= 64 rows: k2 72; k3 64/80/96; k4 80/100/120; k7 64..192), captured
    in descending order into one pool after the single eager reserve with
    the config-derived graph rows (24 x (k + 1)): no capture allocates, and
    every graph replays bit-exactly after all captures."""
    _flash_v100_page4()
    if C.quick() and k != 4:
        pytest.skip("SX_TEST_QUICK")
    new = C.new_ops()
    q_len = k + 1
    lane = C.mtp_lane(page4_graph_rows=24 * q_len)
    request_counts = sorted(
        (count for count in _LANE_GRAPH_REQS if count * q_len >= 64), reverse=True
    )
    first = _verify_attention_case(request_counts[0], q_len, seed=100 + k)
    C.run_attention(new, first, torch.empty_like(first.q), sx_mtp_lane=lane)
    torch.cuda.synchronize()
    before = _graph_state(new)
    pool = torch.cuda.graph_pool_handle()
    captured = []
    for num_reqs in request_counts:
        case = _verify_attention_case(num_reqs, q_len, seed=110 + num_reqs)
        out = torch.empty_like(case.q)
        graph = C.capture(
            lambda case=case, out=out: C.run_attention(
                new, case, out, sx_mtp_lane=lane
            ),
            pool=pool,
        )
        assert _graph_state(new) == before, (k, num_reqs)
        captured.append((graph, case, out))
    for graph, case, out in captured:
        _check_replay(graph, case, out, seeds=(120,))


def test_capture_partition_constant_fallback():
    """A capture that finds no cached partition constant (dropped after the
    eager JIT call) records a device fill instead of a host copy, keeps it
    private to that graph, and replays bit-exactly."""
    _flash_v100_page4()
    new = C.new_ops()
    with C.patched(new, _SM70_QSA_GROUPED_PAGE4=False):
        case = _verify_attention_case(13, seed=90)  # 65 rows, all XQA
        out = torch.empty_like(case.q)
        C.run_attention(new, case, out)  # JIT compile outside capture
        torch.cuda.synchronize()
        new._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS.clear()
        retired = len(new._SM70_QSA_PAGE4_GRAPH_RETIRED)
        graph = C.capture(lambda: C.run_attention(new, case, out), warmup=0)
        assert not new._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS
        assert len(new._SM70_QSA_PAGE4_GRAPH_RETIRED) > retired
        _check_replay(graph, case, out, seeds=(91, 92))


def test_capture_without_reserve_retires_on_growth():
    """Non-MTP callers (no reserve): capture allocates from the graph pool
    without host copies; growth retires the old workspace, and the first
    graph still replays bit-exactly after other allocations reuse memory."""
    _flash_v100_page4()
    new = C.new_ops()
    pool = torch.cuda.graph_pool_handle()  # one shared pool, as in vLLM
    with C.patched(new, _SM70_QSA_GROUPED_PAGE4=False):
        small = _verify_attention_case(13, seed=40)  # 65 rows -> XQA capacity 128
        small_out = torch.empty_like(small.q)
        graph_small = C.capture(
            lambda: C.run_attention(new, small, small_out), pool=pool
        )
        retired_before = len(new._SM70_QSA_PAGE4_GRAPH_RETIRED)
        large = _verify_attention_case(40, q_len=4, seed=41)  # 160 rows -> 256
        large_out = torch.empty_like(large.q)
        graph_large = C.capture(
            lambda: C.run_attention(new, large, large_out), pool=pool
        )
        assert len(new._SM70_QSA_PAGE4_GRAPH_RETIRED) > retired_before
        # A later capture in the same pool that fills fresh blocks: had the
        # retired workspace been freed, this graph could own its memory.
        junk_out = []

        def junk():
            junk_out.clear()
            junk_out.extend(
                torch.full((1 << 18,), 7.0, device="cuda") for _ in range(16)
            )

        graph_junk = C.capture(junk, pool=pool)
        graph_junk.replay()
        _check_replay(graph_small, small, small_out, seeds=(42, 43))
        graph_junk.replay()
        _check_replay(graph_large, large, large_out, seeds=(44,))


def test_eager_paths_unchanged():
    """Eager page4 (all rows XQA, grouped + remainder) == dev2 bit for bit,
    with and without the MTP lane, on the default and on a side stream."""
    _flash_v100_page4()
    new, dev2 = C.new_ops(), C.dev2_ops()
    for grouped in (False, True):
        if grouped and not _grouped_available():
            continue
        with C.patched(new, _SM70_QSA_GROUPED_PAGE4=grouped), C.patched(
            dev2, _SM70_QSA_GROUPED_PAGE4=grouped
        ):
            for num_reqs in (13, 23, 24):
                case = _verify_attention_case(num_reqs, seed=60 + num_reqs)
                expected = torch.full_like(case.q, float("nan"))
                C.run_attention(dev2, case, expected)
                for lane in (None, C.mtp_lane(page4_graph_rows=120)):
                    kwargs = {} if lane is None else {"sx_mtp_lane": lane}
                    actual = torch.full_like(case.q, float("nan"))
                    C.run_attention(new, case, actual, **kwargs)
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    side_out = torch.full_like(case.q, float("nan"))
                    with torch.cuda.stream(side):
                        C.run_attention(new, case, side_out, **kwargs)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    assert C.bitwise_equal(expected, actual), (num_reqs, grouped, lane)
                    assert C.bitwise_equal(expected, side_out), (num_reqs, grouped, lane)


def test_partition_constant_is_readonly_and_shared():
    """The cached constant holds num_partitions and is shared by eager
    workspaces of different streams (the kernel never writes it)."""
    _flash_v100_page4()
    new = C.new_ops()
    with C.patched(new, _SM70_QSA_GROUPED_PAGE4=False):
        case = _verify_attention_case(13, seed=70)
        out = torch.empty_like(case.q)
        C.run_attention(new, case, out)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            C.run_attention(new, case, torch.empty_like(out))
        torch.cuda.synchronize()
    constants = new._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS
    assert len(constants) == 1
    (key, constant), = constants.items()
    assert key[1] == 3 and constant.tolist() == [3]  # ceil(2051 / 1024)
    workspaces = list(new._SM70_QSA_XQA_PAGE4_WORKSPACES.values())
    assert len(workspaces) == 2
    assert all(entry[4].data_ptr() == constant.data_ptr() for entry in workspaces)


def test_switch_off_restores_dev2_workspace():
    """SX_OPT_QSA_MTP_PAGE4_CAPTURE=0 runs the verbatim dev2 workspace code."""
    _flash_v100_page4()
    new, dev2 = C.new_ops(), C.dev2_ops()
    with C.patched(new, _SM70_QSA_GROUPED_PAGE4=False, _SX_OPT_QSA_MTP_PAGE4_CAPTURE=False):
        case = _verify_attention_case(13, seed=80)
        expected = torch.full_like(case.q, float("nan"))
        with C.patched(dev2, _SM70_QSA_GROUPED_PAGE4=False):
            C.run_attention(dev2, case, expected)
        actual = torch.full_like(case.q, float("nan"))
        C.run_attention(new, case, actual, sx_mtp_lane=C.mtp_lane(page4_graph_rows=120))
        torch.cuda.synchronize()
        assert C.bitwise_equal(expected, actual)
        assert not new._SM70_QSA_XQA_PAGE4_PARTITION_COUNTS
        assert not new._SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
