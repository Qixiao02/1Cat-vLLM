# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the b2-allreduce ([C4] SX_OPT_PUSH_AR_WIDE) tests.

``native_push_ctas`` is an independent Python model of the native admission in
csrc/custom_all_reduce.cuh (sm70_tp4_push_allreduce_blocks plus the sum2
admission in CustomAllreduce::allreduce_sum2). The GPU admission probe checks it
against the CTA count the real kernel used (epoch words that flipped), so the
model and the C++ must agree for every arm/op/size the tests exercise.
"""

from __future__ import annotations

KIB = 1024
ROW_BYTES = 2560 * 2  # Qwen3.8 hidden row, FP16
ROW_SIZES = tuple(ROW_BYTES * m for m in range(1, 33))  # 5..160 KiB, 5-KiB steps

# Native constants (csrc/custom_all_reduce.cuh).
PUSH_MAX_BLOCKS = 80
PUSH_THREADS = 128
PACK_BYTES_PER_BLOCK = PUSH_THREADS * 16  # 2 KiB
PUSH_EPOCHS = 2
PUSH_WORLD = 4
PUSH_MAX_BYTES = 32 * 5120 * 2  # 320 KiB per-rank slot
SIGNAL_BYTES = ((PUSH_MAX_BLOCKS * 4 + 127) // 128) * 128  # 384
GENERIC_BUFFER_BYTES = SIGNAL_BYTES + PUSH_EPOCHS * PUSH_WORLD * PUSH_MAX_BYTES
SENTINEL_I16 = 0x7F7F

M8_5120 = 8 * 5120 * 2  # 80 KiB, always push (80 CTAs)
M16_5120 = 16 * 5120 * 2  # 160 KiB (CONCURRENCY)
M32_5120 = 32 * 5120 * 2  # 320 KiB (CONCURRENCY, grid-stride 2)
B8K = 4096 * 2
QWEN4EXP_M1 = 2560 * 2
QWEN38_M4 = 4 * 2560 * 2
QWEN38_M8 = 8 * 2560 * 2
MTP5 = 5 * 2560 * 2

SMALL = "VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES"
CONC = "VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY"
BATCH = "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH"
BATCH_BLOCKS = "VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH_BLOCKS"
SUM2_M1 = "VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1"
MTP5_ENV = "VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5"
WIDE = "SX_OPT_PUSH_AR_WIDE"
WIDE_BLOCKS = "SX_OPT_PUSH_AR_WIDE_BLOCKS"
CONTROLLED = (SMALL, CONC, BATCH, BATCH_BLOCKS, SUM2_M1, MTP5_ENV, WIDE, WIDE_BLOCKS)

# Every arm sets all controlled variables (None = unset).
_BASE = {BATCH_BLOCKS: None, WIDE_BLOCKS: None, MTP5_ENV: "0"}
ARMS: dict[str, dict[str, str | None]] = {
    # 1.8.0-dev1 production defaults + this change (envs.py publishes 1/1).
    "new": {**_BASE, SMALL: "1", CONC: "1", BATCH: "1", SUM2_M1: "1", WIDE: "1"},
    # 1.8.0-dev1 production defaults (previous behaviour).
    "old": {**_BASE, SMALL: "1", CONC: "1", BATCH: "1", SUM2_M1: "1", WIDE: "0"},
    # Everything the native admission lets us turn off. 5 KiB, 8 KiB and
    # 80 KiB regular collectives are hard-wired push and stay push here.
    "pull": {**_BASE, SMALL: "0", CONC: "0", BATCH: "0", SUM2_M1: "0", WIDE: "0"},
    # Wide branch alone: the regular collective reaches the new geometry for
    # every row multiple (production reaches <=80 KiB through SMALL_MESSAGES).
    "wide_only": {**_BASE, SMALL: "0", CONC: "0", BATCH: "1", SUM2_M1: "0", WIDE: "1"},
}


def _is1(env, name):
    return env.get(name) == "1"


def _unset_or1(env, name):
    value = env.get(name)
    return value is None or value == "1"


def wide_enabled(env) -> bool:
    value = env.get(WIDE)
    return value is None or value != "0"


def wide_bytes(nbytes: int) -> bool:
    return nbytes % ROW_BYTES == 0 and 2 * ROW_BYTES <= nbytes <= 32 * ROW_BYTES


def covering_ctas(nbytes: int) -> int:
    return (nbytes + PACK_BYTES_PER_BLOCK - 1) // PACK_BYTES_PER_BLOCK


def _strict_int(raw):
    # The tests only pass plain decimal digits (strtol-compatible).
    return int(raw) if raw.isascii() and raw.isdigit() else None


def _push_blocks(nbytes: int, env, allow_generic: bool) -> int:
    if (
        allow_generic
        and _is1(env, SMALL)
        and 0 < nbytes <= M8_5120
        and nbytes % 16 == 0
    ):
        return covering_ctas(nbytes)
    if nbytes == M8_5120 or (_is1(env, CONC) and nbytes in (M16_5120, M32_5120)):
        return PUSH_MAX_BLOCKS
    if nbytes == B8K:
        return 4
    if nbytes == QWEN4EXP_M1:
        return 3
    batch = _unset_or1(env, BATCH)
    if batch and nbytes in (QWEN38_M4, QWEN38_M8):
        if env.get(BATCH_BLOCKS) is not None:
            raise NotImplementedError("tests keep the legacy BLOCKS override unset")
        return 10 if nbytes == QWEN38_M4 else 20
    if batch and wide_enabled(env) and wide_bytes(nbytes):
        minimum = covering_ctas(nbytes)
        raw = env.get(WIDE_BLOCKS)
        if raw:
            parsed = _strict_int(raw)
            if parsed is not None and minimum <= parsed <= PUSH_MAX_BLOCKS:
                return parsed
        return minimum
    if nbytes == MTP5 and _is1(env, MTP5_ENV):
        return 13
    return 0


def native_push_ctas(op: str, nbytes: int, env) -> int:
    """CTAs of the native push launch for a captured FP16 TP4 call; 0 = pull."""
    if op == "plain":
        return _push_blocks(nbytes, env, True)
    if op != "sum2":
        raise ValueError(op)
    admitted = (
        (
            _unset_or1(env, BATCH)
            and (
                nbytes in (QWEN38_M4, QWEN38_M8, M8_5120)
                or (wide_enabled(env) and wide_bytes(nbytes))
            )
        )
        or (_is1(env, MTP5_ENV) and nbytes == MTP5)
        or (nbytes == QWEN4EXP_M1 and _unset_or1(env, SUM2_M1))
    )
    return _push_blocks(nbytes, env, False) if admitted else 0
