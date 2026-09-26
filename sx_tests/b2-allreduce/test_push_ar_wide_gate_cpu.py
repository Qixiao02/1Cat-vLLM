# SPDX-License-Identifier: Apache-2.0
"""b2-allreduce [C4] CPU checks (no GPU): Python gate, size contract, the
admission table this change produces, the covering-grid invariant, and that
the model in _push_ar_common matches constants/branch order in the header.

  /opt/venv/bin/python -m pytest -q sx_tests/b2-allreduce/test_push_ar_wide_gate_cpu.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _push_ar_common as C  # noqa: E402

HEADER = HERE.parents[1] / "csrc" / "custom_all_reduce.cuh"


@pytest.mark.parametrize(
    "value,enabled",
    [(None, True), ("1", True), ("0", False), ("", True), ("false", True)],
)
def test_python_gate_semantics(monkeypatch, value, enabled):
    car = pytest.importorskip("vllm.distributed.device_communicators.custom_all_reduce")
    if value is None:
        monkeypatch.delenv("SX_OPT_PUSH_AR_WIDE", raising=False)
    else:
        monkeypatch.setenv("SX_OPT_PUSH_AR_WIDE", value)
    assert car._sx_push_ar_wide_enabled() is enabled
    # The model used by the GPU tests follows the same semantics.
    assert C.wide_enabled({C.WIDE: value}) is enabled


def test_python_payload_contract_matches_model():
    car = pytest.importorskip("vllm.distributed.device_communicators.custom_all_reduce")
    for nbytes in range(16, 400 * 1024, 16):
        assert car._sx_push_ar_wide_payload(nbytes) == C.wide_bytes(nbytes), nbytes
    admitted = [n for n in range(16, 400 * 1024, 16) if C.wide_bytes(n)]
    assert admitted == [m * C.ROW_BYTES for m in range(2, 33)]


def test_admission_table_new_vs_old():
    new, old = C.ARMS["new"], C.ARMS["old"]
    for op in ("plain", "sum2"):
        for m in range(1, 33):
            n = m * C.ROW_BYTES
            c_new = C.native_push_ctas(op, n, new)
            c_old = C.native_push_ctas(op, n, old)
            assert c_new > 0, (op, m)
            if op == "sum2" and m == 16:
                assert c_new == c_old == 80
            else:
                assert c_new == C.covering_ctas(n), (op, m, c_new)
            if c_old:
                assert c_old == c_new, (op, m)  # established launches unchanged
    old_pull = {
        op: [m for m in range(1, 33) if not C.native_push_ctas(op, m * C.ROW_BYTES, old)]
        for op in ("plain", "sum2")
    }
    assert old_pull["plain"] == list(range(17, 32))
    assert old_pull["sum2"] == [m for m in range(2, 33) if m not in (4, 8, 16)]
    # M24 = 60 CTAs, M32 = 80: within the 80 per-CTA epoch words.
    assert C.native_push_ctas("plain", 24 * C.ROW_BYTES, new) == 60
    assert C.native_push_ctas("sum2", 32 * C.ROW_BYTES, new) == 80


def test_switch_only_touches_row_multiples_and_needs_batch():
    """The switch changes nothing outside the [2..32, 2560] contract, and it
    is inert when the Qwen3.8 batch admission is rolled back."""
    for arm in ("new", "wide_only", "pull"):
        on = dict(C.ARMS[arm], **{C.WIDE: "1"})
        off = dict(C.ARMS[arm], **{C.WIDE: "0"})
        no_batch_on = dict(on, **{C.BATCH: "0"})
        no_batch_off = dict(off, **{C.BATCH: "0"})
        for op in ("plain", "sum2"):
            for n in range(16, 400 * 1024, 16):
                assert C.native_push_ctas(op, n, no_batch_on) == C.native_push_ctas(
                    op, n, no_batch_off
                )
                if not C.wide_bytes(n):
                    assert C.native_push_ctas(op, n, on) == C.native_push_ctas(
                        op, n, off
                    )
    assert dict(C.ARMS["new"], **{C.WIDE: "0"}) == C.ARMS["old"]


@pytest.mark.parametrize("arm", sorted(C.ARMS))
def test_every_push_launch_covers_its_payload(arm):
    """Protocol invariant: pack offset o always maps to CTA o // 128.

    Every push grid must cover its payload with one pack per thread (the only
    grid-stride launch is the established 320-KiB/80-CTA one, whose second
    pass touches offsets no other size uses), and stay within the 80 epoch
    words.
    """
    env = C.ARMS[arm]
    for op in ("plain", "sum2"):
        for n in range(16, 400 * 1024, 16):
            ctas = C.native_push_ctas(op, n, env)
            if not ctas:
                continue
            assert ctas <= C.PUSH_MAX_BLOCKS, (arm, op, n)
            if n != C.M32_5120:
                assert ctas >= C.covering_ctas(n), (arm, op, n, ctas)
            assert n <= C.PUSH_MAX_BYTES


def test_wide_blocks_override_never_shrinks_the_grid():
    for m in range(2, 33):
        n = m * C.ROW_BYTES
        minimum = C.covering_ctas(n)
        for raw in ("1", str(minimum - 1), str(minimum), "80", "81", "x", ""):
            env = dict(C.ARMS["new"], **{C.WIDE_BLOCKS: raw})
            ctas = C.native_push_ctas("sum2", n, env)
            assert minimum <= ctas <= C.PUSH_MAX_BLOCKS


def _header() -> str:
    if not HEADER.exists():
        pytest.skip(f"source header not available at {HEADER}")
    return HEADER.read_text()


def test_model_constants_match_header():
    text = _header()

    def const(name):
        match = re.search(rf"{name}\s*=\s*([^;]+);", text)
        assert match, name
        return match.group(1).strip()

    assert const("constexpr int kSm70Tp4PushAllreduceBlocks") == str(C.PUSH_MAX_BLOCKS)
    assert const("constexpr int kSm70Tp4PushAllreduceThreads") == str(C.PUSH_THREADS)
    assert const("constexpr int kSm70Tp4PushAllreduceEpochs") == str(C.PUSH_EPOCHS)
    assert const("constexpr size_t kSm70Tp4PushAllreduceMaxBytes") == (
        "kSm70Tp4PushAllreduceM32Bytes"
    )
    assert "32 * kSm70GemmaRmsNormHiddenSize * sizeof(half)" in text
    assert "constexpr int kSm70GemmaRmsNormHiddenSize = 5120;" in text
    assert "kSm70Tp4PushAllreduceQwen38RowBytes = 2560 * sizeof(half)" in text
    assert "kSm70Tp4PushAllreduceQwen38WideMinRows = 2;" in text
    assert "kSm70Tp4PushAllreduceQwen38WideMaxRows = 32;" in text
    assert 'std::getenv("SX_OPT_PUSH_AR_WIDE")' in text
    assert 'std::strcmp(raw, "0") != 0' in text


def test_header_branch_order_matches_model():
    text = _header()
    start = text.index("inline int sm70_tp4_push_allreduce_blocks(")
    body = text[start : text.index("\n}\n", start)]
    order = [
        body.index("VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES"),
        body.index("VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY"),
        body.index("kSm70Tp4PushAllreduce8KiBBytes"),
        body.index("kSm70Tp4PushAllreduceQwen4ExpBytes"),
        body.index("VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH\""),
        body.index("sm70_tp4_push_allreduce_wide_bytes(bytes)"),
        body.index("VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5"),
    ]
    assert order == sorted(order)
    sum2 = text[text.index("void allreduce_sum2(") :]
    sum2 = sum2[: sum2.index("sm70_cross_device_reduce_sum2_1stage_push<")]
    assert "sm70_tp4_push_allreduce_wide_bytes(bytes)" in sum2
    assert "sm70_tp4_push_allreduce_blocks(bytes)" in sum2
