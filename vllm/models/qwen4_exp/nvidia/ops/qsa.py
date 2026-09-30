# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen4Exp weight-free QSA path."""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import regex as re
import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast as fp8_e4m3fn_bits_to_fp32,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_SM70_QSA_TOPK_LIBRARY = os.getenv("VLLM_SM70_QSA_TOPK_LIBRARY")
if _SM70_QSA_TOPK_LIBRARY is not None:
    torch.ops.load_library(_SM70_QSA_TOPK_LIBRARY)
    _topk_version = getattr(
        torch.ops._C_qsa_sm70, "decode_specialization_version", None
    )
    if _topk_version is not None:
        logger.info(
            "SM70 QSA source-overlay decode specialization version %d.", _topk_version()
        )

if hasattr(torch.ops._C_qsa_sm70, "qsa_lexicographic_topk"):

    @torch.library.register_fake("_C_qsa_sm70::qsa_lexicographic_topk")
    def _qsa_lexicographic_topk_sidecar_fake(
        logits: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        topk: int,
    ) -> None:
        del logits, lengths, output, topk
        return None


if hasattr(torch.ops._C_qsa_sm70, "qsa_lexicographic_topk_decode_rows"):

    @torch.library.register_fake("_C_qsa_sm70::qsa_lexicographic_topk_decode_rows")
    def _qsa_lexicographic_topk_decode_rows_sidecar_fake(
        logits: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        topk: int,
    ) -> None:
        del logits, lengths, output, topk
        return None


_SM70_INDEXER_CUBLAS = os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS", "1") == "1"
_SM70_INDEXER_SCORE_TILE_BYTES = (
    int(os.getenv("VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB", "64")) * 1024 * 1024
)
_SM70_INDEXER_CUBLAS_MIN_ROWS = int(
    os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS", "512")
)
_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS = int(
    os.getenv("VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS", str(1024**2))
)
_SM70_QSA_XQA_PAGE4 = os.getenv("VLLM_SM70_QSA_XQA_PAGE4", "1") == "1"
_SM70_QSA_XQA_PAGE4_MIN_ROWS = int(
    # Operator crossover on SM70 is around 48 rows for the fixed QSA width.
    # Use a conservative 64-row workload gate rather than coupling the route
    # to a particular server's max_num_batched_tokens setting.
    os.getenv("VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS", "64")
)
_SM70_QSA_XQA_PAGE4_PARTITION = 1024
_SM70_QSA_XQA_PAGE4_PAGES = 513
_SM70_QSA_XQA_PAGE4_MARKER = 1 << 30
_SM70_QSA_GROUPED_PAGE4 = os.getenv("VLLM_SM70_QSA_GROUPED_PAGE4", "1") == "1"
_SM70_QSA_GROUPED_PAD_FIX = os.getenv("VLLM_SM70_QSA_GROUPED_PAD_FIX", "1") == "1"
_SM70_QSA_GROUPED_PAGE4_QUERIES = 8
_SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES = (
    _SM70_QSA_XQA_PAGE4_PAGES * _SM70_QSA_GROUPED_PAGE4_QUERIES + 56
)
_SM70_QSA_XQA_PAGE4_WORKSPACES: dict[
    tuple[int, int, int, int, int, bool],
    tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_SM70_QSA_GROUPED_PAGE4_WORKSPACES: dict[
    tuple[int, int],
    tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}
_SM70_QSA_GROUPED_PAGE4_ABI_CACHE: tuple[object, int] | None = None
# SX_OPT_QSA_MTP_PAGE4_CAPTURE state (see the batch 3a switch block below).
# Read-only XQA partition-count constants, one per (device, partitions).
_SM70_QSA_XQA_PAGE4_PARTITION_COUNTS: dict[tuple[int, int], torch.Tensor] = {}
# Workspaces used while a CUDA graph is being captured. Like the old
# capture-stream entries they are shared by every captured graph (graphs
# replay one at a time on the model stream), but they are never freed: a
# replaced entry moves to _SM70_QSA_PAGE4_GRAPH_RETIRED because an earlier
# graph still holds its addresses.
_SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES: dict[
    tuple[int, int, int, int, bool],
    tuple[int, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES: dict[
    tuple[int, int],
    tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_SM70_QSA_PAGE4_GRAPH_RETIRED: list[object] = []
_SM70_QSA_PAGE4_GRAPH_RESERVED: set[tuple] = set()


# ---------------------------------------------------------------------------
# ShiXiang batch-1 QSA optimisation switches (group "qsa").
#
# Every switch is read once at import, defaults to ON, and "0" restores the
# exact baseline code path (the old launch sequence is kept verbatim behind
# each switch):
#
#   SX_OPT_QSA_HOST_BOUND      Single-request cuBLAS indexer: take the score
#                              width from the host seq_lens copy instead of
#                              int(all_visible.max().item()) (one blocking
#                              cudaStreamSynchronize per QSA layer). Bitwise.
#   SX_OPT_QSA_HOST_BOUND_CHECK  Debug only, default "0". "1" additionally runs
#                              the old .item() and raises if it disagrees
#                              (single-request host bound and every
#                              SX_OPT_QSA_MIXED_CUBLAS per-request width).
#   SX_OPT_QSA_TWO_WARP32      Two-warp split-K sparse partial for M <= 32
#                              (was M <= 16). Bitwise (design_1 [C1]).
#   SX_OPT_QSA_RESOLVED_ROWS   Exact physical-address resolver for every decode
#                              width 1 <= M <= 32 and any int32-safe page size
#                              (was M == 1, page 400). Bitwise (design_4 [MR4]).
#   SX_OPT_QSA_SCORER_STRIDE   OPT-IN ("1"; default "0" after V100 validation
#                              showed it slower). Paged indexer scorer walks the visible tiles with
#                              a bounded grid-stride loop for 2 <= M <= 32
#                              instead of a max_model_len/4-wide grid of mostly
#                              empty CTAs. Bitwise (design_1 [C7]).
#   SX_OPT_QSA_MIXED_GROUPS    Grouped page4 sparse attention: never put rows of
#                              different requests into one 8-query group. Decode
#                              rows of a mixed step go to the row-wise XQA page4
#                              kernel instead of forming 8-request groups that
#                              one CTA must walk serially (the 3.6 vs 1.5 ms per
#                              layer mixed-step straggler). NOT bitwise versus
#                              the old mixed step; prefill rows become bitwise
#                              equal to the same request run alone. Only used
#                              when an old 8-row group would mix at least
#                              SX_OPT_QSA_MIXED_MIN_GROUP_REQUESTS (default 3)
#                              requests; otherwise the old launch runs.
#   SX_OPT_QSA_MIXED_CUBLAS    Indexer: inside a mixed batch, score each large
#                              prefill request with the single-request cuBLAS
#                              path (design_3 [P2]). NOT bitwise versus the old
#                              mixed step for that request (Triton FMA scorer
#                              vs cuBLAS HMMA); bitwise equal to the request run
#                              alone. Other rows stay bitwise unchanged.
#
# Batch 2 (group "small-native"):
#
#   SX_OPT_QSA_TOPK_ROWS       Exact lexicographic block top-k: run the
#                              decode-specialised selector (coarse radix byte +
#                              shared-memory candidate refinement) for every
#                              row of a 2 <= M <= 32 batch, grid = M, instead of
#                              the generic four-full-scan kernel (the decode
#                              kernel used to serve M == 1 only). Bitwise: the
#                              same selected ids and order as the generic
#                              kernel and as the M == 1 kernel on each row
#                              (design_4 [MR5]). Needs the rebuilt
#                              _C_stable_libtorch (op
#                              qsa_lexicographic_topk_decode_rows) or a
#                              validation sidecar built from this source;
#                              without the op the old launch runs.
#
# The host metadata these paths need (query_start_loc / seq_lens CPU copies)
# is attached by Qwen4ExpQSAMetadataBuilder in ../qsa.py when no speculative
# decoding is configured, and (batch 3a, SX_OPT_QSA_MTP_HOST_METADATA) in the
# admitted MTP lane, where every sequence length that may be an optimistic
# upper bound is replaced by -1 (which every planner below rejects); without
# it every path falls back to the baseline behaviour.
# ---------------------------------------------------------------------------
def _sx_opt_enabled(name: str) -> bool:
    return os.environ.get(f"SX_OPT_{name}", "1") != "0"


_SX_OPT_QSA_HOST_BOUND = _sx_opt_enabled("QSA_HOST_BOUND")
_SX_OPT_QSA_HOST_BOUND_CHECK = (
    os.environ.get("SX_OPT_QSA_HOST_BOUND_CHECK", "0") == "1"
)
_SX_OPT_QSA_TWO_WARP32 = _sx_opt_enabled("QSA_TWO_WARP32")
_SX_OPT_QSA_RESOLVED_ROWS = _sx_opt_enabled("QSA_RESOLVED_ROWS")
# Validation (opt180dev1, V100): the grid-stride scorer measured 2.5-5x SLOWER than
# the old rectangular grid (M24 x12 layers: 1.0 -> 4.0 ms at <=8K, 4.6 -> 22.5 ms at
# <=128K), so it is opt-in ("1"); the default keeps the baseline launch.
_SX_OPT_QSA_SCORER_STRIDE = os.environ.get("SX_OPT_QSA_SCORER_STRIDE", "0") == "1"
_SX_OPT_QSA_MIXED_GROUPS = _sx_opt_enabled("QSA_MIXED_GROUPS")
_SX_OPT_QSA_MIXED_CUBLAS = _sx_opt_enabled("QSA_MIXED_CUBLAS")
_SX_OPT_QSA_TOPK_ROWS = _sx_opt_enabled("QSA_TOPK_ROWS")
# Decode widths served by the two-warp partial and the address resolver.
_SX_QSA_DECODE_MAX_ROWS = 32
# The strided scorer targets roughly one wave of two-warp CTAs.
_SX_QSA_SCORER_TARGET_CTAS = 2560
# A mixed step with more routing segments than this keeps the old grouping.
_SX_QSA_MIXED_MAX_SEGMENTS = 16
# Validation (opt180dev1, V100): the request-aligned routing only pays off when some
# old 8-row group mixes at least this many requests (decode rows of different
# requests). Prefill-only multi-request steps, whose old groups mix at most 2
# requests, keep the single grouped launch, which measured faster (two 1570-row
# prefills: 5.14 vs 5.97 ms/layer; 450+784 rows: 2.22 vs 2.26 ms/layer).
try:
    _SX_QSA_MIXED_MIN_GROUP_REQUESTS = int(
        os.environ.get("SX_OPT_QSA_MIXED_MIN_GROUP_REQUESTS", "3")
    )
except ValueError:
    _SX_QSA_MIXED_MIN_GROUP_REQUESTS = 3


# ---------------------------------------------------------------------------
# ShiXiang batch 3a (group "qsa-mtp"): the speculative MTP lane, i.e.
# speculative_config.method == "mtp" on the admitted Qwen3.8 SM70 TP4
# topology (../qsa.py _sx_qsa_mtp_lane_contract). Every switch defaults to ON
# and "0" restores the 1.8.0-dev2 behaviour. The MTP-lane options reach the
# ops as an explicit ``sx_mtp_lane`` argument built once per layer; without it
# (every other configuration, including the no-MTP production lane) the
# decode-row gates below are the 1.8.0-dev2 ones.
#
#   SX_OPT_QSA_MTP_PAGE4_CAPTURE   Page4 workspaces under CUDA-graph capture.
#                                  The XQA partition count is a cached,
#                                  read-only device constant (the old
#                                  torch.tensor() host copy aborted FULL
#                                  capture: "operation not permitted when
#                                  stream is capturing" at k4 verify widths
#                                  >= 64 with an XQA remainder, e.g. 23 x 5).
#                                  Captured graphs use their own workspaces,
#                                  which are never freed (a graph keeps their
#                                  addresses) and, in the MTP lane, are
#                                  reserved outside capture by the first eager
#                                  page4 call. Addresses only: bitwise.
#   SX_OPT_QSA_MTP_PAGE4_GRAPH_ROWS  Optional int (read in ../qsa.py): the
#                                  widest FULL graph the reserve covers
#                                  (default derived from the config; 0 = no
#                                  reserve, capture then allocates from the
#                                  graph pool without host copies).
#   SX_OPT_QSA_MTP_DECODE_ROWS     Decode-row kernels (two-warp split-K
#                                  partial, resolved physical rows, decode
#                                  top-k rows) for verify widths up to
#                                  SX_OPT_QSA_MTP_DECODE_MAX_ROWS (default 63,
#                                  clamped to [16, 63]; XQA page4 serves >= 64
#                                  rows). Each verify row carries its own
#                                  request and causal query position, so the
#                                  per-row kernels need no change. Bitwise:
#                                  M33..63 compile to the M32 constexpr set
#                                  (BLOCK_N 16, 8 splits) and the rows top-k
#                                  is the per-row decode selector
#                                  (design_1 [MTP-9]).
#   SX_OPT_QSA_MTP_{TWO_WARP,RESOLVED,TOPK}_MAX_ROWS
#                                  Optional per-kernel caps (default the
#                                  decode max, clamped to [16, 63]) so a
#                                  measured crossover can be admitted without
#                                  a code change.
#   SX_OPT_QSA_MTP_HOST_METADATA   (../qsa.py) host query starts and exact
#                                  prefill sequence lengths in the MTP lane,
#                                  restoring QSA_HOST_BOUND / MIXED_CUBLAS /
#                                  MIXED_GROUPS there (design_1 [MTP-4]).
# ---------------------------------------------------------------------------
_SX_OPT_QSA_MTP_PAGE4_CAPTURE = _sx_opt_enabled("QSA_MTP_PAGE4_CAPTURE")
# XQA page4 takes over at _SM70_QSA_XQA_PAGE4_MIN_ROWS (64) rows.
_SX_QSA_MTP_DECODE_ROWS_LIMIT = 63
_SX_QSA_MTP_DECODE_ROWS_FLOOR = 16


def _sx_env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    value = default
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r; using %d.", name, raw, default)
    return max(low, min(high, value))


class SxQsaMtpLane(NamedTuple):
    """QSA options of a layer whose config satisfies the MTP-lane contract.

    Built once per layer by ../qsa.py (sx_qsa_mtp_lane_options); passed to
    the ops as ``sx_mtp_lane``. ``None`` keeps the 1.8.0-dev2 gates.
    """

    two_warp_max_rows: int
    resolved_max_rows: int
    topk_max_rows: int
    # Widest FULL CUDA graph whose page4 workspaces are reserved outside
    # capture (0 = no reserve).
    page4_graph_rows: int


def sx_qsa_mtp_lane_options(page4_graph_rows: int) -> SxQsaMtpLane:
    """MTP-lane QSA options from the SX_OPT_QSA_MTP_* switches."""

    if _sx_opt_enabled("QSA_MTP_DECODE_ROWS"):
        decode_max = _sx_env_int(
            "SX_OPT_QSA_MTP_DECODE_MAX_ROWS",
            _SX_QSA_MTP_DECODE_ROWS_LIMIT,
            _SX_QSA_MTP_DECODE_ROWS_FLOOR,
            _SX_QSA_MTP_DECODE_ROWS_LIMIT,
        )

        def cap(kind: str) -> int:
            return _sx_env_int(
                f"SX_OPT_QSA_MTP_{kind}_MAX_ROWS",
                decode_max,
                _SX_QSA_MTP_DECODE_ROWS_FLOOR,
                _SX_QSA_MTP_DECODE_ROWS_LIMIT,
            )

        two_warp, resolved, topk = cap("TWO_WARP"), cap("RESOLVED"), cap("TOPK")
    else:
        two_warp = resolved = topk = _SX_QSA_DECODE_MAX_ROWS
    graph_rows = max(0, int(page4_graph_rows)) if _SX_OPT_QSA_MTP_PAGE4_CAPTURE else 0
    return SxQsaMtpLane(two_warp, resolved, topk, graph_rows)


def _sx_decode_rows_cap(sx_mtp_lane: SxQsaMtpLane | None, field: str) -> int:
    if sx_mtp_lane is None:
        return _SX_QSA_DECODE_MAX_ROWS
    return int(getattr(sx_mtp_lane, field))


def _host_int_list(tensor: torch.Tensor | None, length: int | None = None):
    """Return a CPU int tensor as a Python list, or None if it is unusable.

    Never touches a device tensor, so it cannot introduce a host sync.
    """
    if (
        tensor is None
        or not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.ndim != 1
        or (length is not None and tensor.shape[0] != length)
    ):
        return None
    return [int(value) for value in tensor.tolist()]


def _qsa_grouped_page4_abi_version(flash_attn_v100_cuda) -> int:
    """Return the grouped-page4 ABI without probing it on the hot path.

    New Flash-V100 builds expose an explicit version. Wheels predating that
    capability query are recognized conservatively from pybind's generated
    signature: ABI v1 has arguments 0..8, while ABI v2 has arguments 0..11.
    An unknown binding is treated as unsupported instead of risking a server
    crash on the first large prefill.
    """
    global _SM70_QSA_GROUPED_PAGE4_ABI_CACHE
    cached = _SM70_QSA_GROUPED_PAGE4_ABI_CACHE
    if cached is not None and cached[0] is flash_attn_v100_cuda:
        return cached[1]

    version = 0
    capability = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_abi_version", None)
    if callable(capability):
        try:
            version = int(capability())
        except (RuntimeError, TypeError, ValueError):
            version = 0
    else:
        binding = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_fwd", None)
        doc = getattr(binding, "__doc__", "") or ""
        argument_ids = [int(match) for match in re.findall(r"\barg(\d+):", doc)]
        if argument_ids:
            highest_argument = max(argument_ids)
            if highest_argument >= 11:
                version = 2
            elif highest_argument >= 8:
                version = 1

    _SM70_QSA_GROUPED_PAGE4_ABI_CACHE = (flash_attn_v100_cuda, version)
    return version


def _qsa_grouped_page4_supported(
    flash_attn_v100_cuda,
    kv_cache_dtype: str,
) -> bool:
    forward = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_fwd", None)
    planner = getattr(flash_attn_v100_cuda, "grouped_sparse_page4_plan_fwd", None)
    if not callable(forward) or not callable(planner):
        return False
    abi_version = _qsa_grouped_page4_abi_version(flash_attn_v100_cuda)
    return abi_version >= 2 or (
        abi_version == 1 and kv_cache_dtype in ("auto", "float16")
    )


@triton.jit(do_not_specialize=["num_requests"])
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


@triton.jit(do_not_specialize=["num_requests", "col_programs"])
def _qsa_mqa_paged_strided_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    col_programs,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    """Grid-stride twin of _qsa_mqa_paged_kernel with TILES_PER_PROG=1.

    Program (row, p) scores tiles p, p + col_programs, ... below the row's
    visible bound. The per-tile body is copied verbatim from the original
    kernel, so every written score is bitwise identical; only the number of
    CTAs that exit without work changes.
    """
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    last_tile = tl.minimum(tl.cdiv(visible, BLOCK_N), tl.cdiv(num_columns, BLOCK_N))
    first_tile = tl.program_id(1)
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if first_tile >= last_tile:
        return

    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(first_tile, last_tile, col_programs, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


@triton.jit
def _qsa_visible_blocks_kernel(
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    rows,
    num_requests,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row, mask=row < rows, other=-1)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row, mask=row < rows, other=-1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(row < rows) & valid_request,
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    tl.store(
        visible_blocks_ptr + row,
        tl.where(valid_request, tl.maximum(visible, 0), 0),
        mask=row < rows,
    )


@triton.jit
def _qsa_gather_single_request_keys_kernel(
    cache_ptr,
    page_table_ptr,
    keys_ptr,
    valid_ptr,
    stride_cache_page,
    stride_cache_token,
    stride_cache_dim,
    stride_table_page,
    stride_keys_row,
    columns,
    num_pages,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    column = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    logical_page = column // PAGE_SIZE
    page_offset = column % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr + logical_page * stride_table_page,
        mask=(column < columns) & (logical_page < PAGE_TABLE_WIDTH),
        other=-1,
    )
    valid = (column < columns) & (physical_page >= 0) & (physical_page < num_pages)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    key = tl.load(
        cache_ptr
        + safe_page * stride_cache_page
        + page_offset * stride_cache_token
        + dims * stride_cache_dim,
        mask=valid & (dims < HEAD_DIM),
        other=0.0,
    )
    tl.store(
        keys_ptr + column * stride_keys_row + dims,
        key,
        mask=(column < columns) & (dims < HEAD_DIM),
    )
    tl.store(valid_ptr + column, valid, mask=column < columns)


@triton.jit
def _qsa_relu_headsum_visible_kernel(
    score_ptr,
    visible_ptr,
    key_valid_ptr,
    logits_ptr,
    stride_score_row,
    stride_score_column,
    stride_logits_row,
    width,
    column_start,
    score_divisor,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    absolute_columns = column_start + columns
    visible = tl.load(visible_ptr + row)
    key_valid = tl.load(
        key_valid_ptr + absolute_columns,
        mask=columns < width,
        other=0,
    ).to(tl.int1)
    live = (columns < width) & (absolute_columns < visible) & key_valid
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in range(NUM_HEADS):
        values = tl.load(
            score_ptr
            + (row * NUM_HEADS + head) * stride_score_row
            + columns * stride_score_column,
            mask=columns < width,
            other=0.0,
        ).to(tl.float32)
        score += tl.maximum(values, 0.0)
    tl.store(
        logits_ptr + row * stride_logits_row + absolute_columns,
        tl.where(live, score / score_divisor, -float("inf")),
        mask=columns < width,
    )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _qsa_xqa_page4_table_kernel(
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    encoded_pages_ptr,
    xqa_sequence_lengths_ptr,
    stride_indices_row,
    stride_table_req,
    stride_encoded_row,
    rows,
    num_cache_blocks,
    num_requests,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    COMPLETE_PAGES: tl.constexpr,
    OUTPUT_PAGES: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    PHYSICAL_PAGE_STRIDE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_PAGES)
    request = tl.load(token_to_req_ptr + row)
    request_is_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=request_is_valid,
        other=0,
    )
    # Padded graph rows use position -1. Clamp malformed or stale positions to
    # the request's live sequence so they cannot expose a synthetic tail page.
    visible_tokens = tl.minimum(
        tl.maximum(query_position + 1, 0),
        sequence_length,
    )
    complete_pages = tl.minimum(
        tl.minimum(visible_tokens // 4, sequence_length // 4),
        COMPLETE_PAGES,
    )
    tail_count = visible_tokens - (visible_tokens // 4) * 4
    is_complete = slots < complete_pages
    selected_token = tl.load(
        indices_ptr + row * stride_indices_row + slots * 4,
        mask=(row < rows) & is_complete,
        other=-1,
    )
    tail_token = (visible_tokens // 4) * 4
    selected_tail_token = tl.load(
        indices_ptr + row * stride_indices_row + complete_pages * 4,
        mask=(row < rows) & (tail_count > 0),
        other=-1,
    )
    tail_is_valid = (
        (tail_count > 0)
        & (selected_tail_token == tail_token)
        & (selected_tail_token < sequence_length)
    )
    is_tail = (slots == complete_pages) & tail_is_valid
    logical_token = tl.where(is_tail, selected_tail_token, selected_token)
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token - logical_page * PAGE_SIZE
    valid = (
        (row < rows)
        & request_is_valid
        & (logical_token >= 0)
        & (logical_token < sequence_length)
        & (logical_page < PAGE_TABLE_WIDTH)
        & (is_complete | is_tail)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    physical_microblock = (
        tl.maximum(physical_page, 0) * PHYSICAL_PAGE_STRIDE + page_offset // 4
    )
    # Sort by logical token, not allocator-dependent physical page ID. Keep
    # the partial causal page after all complete pages and invalid slots last.
    logical_key = safe_token.to(tl.int64) << 31
    encoded = tl.where(
        valid & is_complete,
        logical_key | physical_microblock.to(tl.int64),
        tl.where(
            valid & is_tail,
            (1 << 62) | logical_key | physical_microblock.to(tl.int64),
            9223372036854775807,
        ),
    )
    tl.store(
        encoded_pages_ptr + row * stride_encoded_row + slots,
        encoded,
        mask=(row < rows) & (slots < OUTPUT_PAGES),
    )
    tl.store(
        xqa_sequence_lengths_ptr + row,
        complete_pages * 4 + tl.where(tail_is_valid, tail_count, 0),
        mask=row < rows,
    )


@triton.jit
def _qsa_resolve_physical_indices_kernel(
    indices_ptr,
    table_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_indices_row,
    stride_table_req,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    token = tl.load(
        indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1
    )
    safe_token = tl.maximum(token, 0)
    page = safe_token // PAGE_SIZE
    valid = (
        (columns < TOPK)
        & (request >= 0)
        & (request < num_requests)
        & (token >= 0)
        & (page < PAGE_TABLE_WIDTH)
    )
    physical = tl.load(
        table_ptr + safe_request * stride_table_req + page, mask=valid, other=-1
    )
    valid &= (physical >= 0) & (physical < num_cache_blocks)
    # Keep logical order, duplicates and invalid slots. Never sort by page.
    slot = physical.to(tl.int64) * PAGE_SIZE + safe_token % PAGE_SIZE
    tl.store(
        output_ptr + row * TOPK + columns,
        tl.where(valid, slot, -1),
        mask=columns < TOPK,
    )


@triton.jit(do_not_specialize=["num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    k_scale,
    v_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_E4M3: tl.constexpr,
    RESOLVED_INDICES: tl.constexpr = False,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    qk_scale_log2 = softmax_scale_log2
    if KV_E4M3:
        # Keep the exactly decoded E4M3 values in the dot inputs and apply the
        # scalar K dequantization factor once to the accumulated QK scores.
        qk_scale_log2 *= k_scale

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0)
        if RESOLVED_INDICES:
            physical_page = logical_page
        else:
            valid &= logical_page < PAGE_TABLE_WIDTH
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if KV_E4M3:
            keys = fp8_e4m3fn_bits_to_fp32(keys).to(query.dtype)
            values = fp8_e4m3fn_bits_to_fp32(values).to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= qk_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        if KV_E4M3:
            # V dequantization is linear, so apply its scalar after the
            # normalized FP32 accumulation instead of to every loaded value.
            normalized_output *= v_scale
        if output_gate_ptr is not None:
            # Preserve the compiled path's rounded attention output before
            # evaluating the sigmoid gate and final product in FP32.
            normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
            output_gate = tl.load(
                output_gate_ptr
                + row * stride_output_gate_row
                + (first_head + head_offsets[:, None]) * stride_output_gate_head
                + dim_offsets[None, :],
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
            normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(
                output_gate
            )
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    v_scale,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    KV_E4M3: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    if KV_E4M3:
        # Apply the V scale once after combining all independently normalized
        # splits. Scaling partials earlier would repeat this work per split.
        merged *= v_scale
    if output_gate_ptr is not None:
        merged = merged.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + head * stride_output_gate_head
            + dim_offsets
        ).to(tl.float32)
        merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _qsa_output_gate_kernel(
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    HEAD_DIM: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    output = tl.load(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets
    ).to(tl.float32)
    gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        output * tl.sigmoid(gate),
    )


def _qsa_output_gate(output: torch.Tensor, output_gate: torch.Tensor) -> None:
    _qsa_output_gate_kernel[(output.shape[0], output.shape[1])](
        output,
        output_gate,
        output.stride(0),
        output.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        HEAD_DIM=output.shape[2],
        num_warps=4,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
    *,
    launch_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache.

    ``launch_rows`` (internal) selects the tile profile as if the call had that
    many rows. The mixed-batch indexer uses it so rows scored in a smaller
    sub-call keep exactly the tile shape they had in the full-batch call.
    """

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires CUDA and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    profile_rows = q.shape[0] if launch_rows is None else launch_rows
    is_sm70 = current_platform.is_device_capability(70)
    sm70_single_token = profile_rows == 1 and is_sm70
    # On V100 the GB300 decode tile leaves the 128-d scorer badly
    # under-occupied. A 32-column, two-warp tile preserves the selected QSA
    # blocks while exposing enough independent CTAs for the single-row path.
    BLOCK_N = 32 if sm70_single_token else 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if profile_rows <= 32 else 8
    if (
        _SX_OPT_QSA_SCORER_STRIDE
        and is_sm70
        and launch_rows is None
        and 2 <= q.shape[0] <= _SX_QSA_DECODE_MAX_ROWS
    ):
        # Decode widths: the rectangular grid is sized by the model capacity
        # (max_model_len / ratio columns) although a row only has
        # cdiv(visible, BLOCK_N) live tiles. Walk those tiles with a bounded
        # grid-stride loop; the per-tile arithmetic is unchanged (bitwise).
        assert tiles_per_program == 1 and BLOCK_N == 64
        col_programs = min(
            triton.cdiv(columns, BLOCK_N),
            max(1, triton.cdiv(_SX_QSA_SCORER_TARGET_CTAS, q.shape[0])),
        )
        _qsa_mqa_paged_strided_kernel[(q.shape[0], col_programs)](
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            visible_blocks,
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(3),
            page_table.stride(0),
            page_table.stride(1),
            logits.stride(0),
            q.shape[0],
            columns,
            k_cache.shape[0],
            page_table.shape[0],
            float(score_divisor),
            col_programs,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            STAGES=2,
            MAX_N=MAX_N,
            COMPRESS_RATIO=compress_ratio,
            num_warps=2,
        )
        return logits, visible_blocks
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


def _qsa_indexer_cublas_shape_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> bool:
    """Whether the exact Qwen3.8 single-request index shape can use cuBLAS."""

    return (
        q.dtype == torch.float16
        and k_cache.dtype == torch.float16
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and k_cache.ndim == 4
        and k_cache.shape[2:] == (1, 128)
        and page_table.ndim == 2
        and page_table.shape[0] == 1
    )


def _use_sm70_qsa_indexer_cublas(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
) -> bool:
    return (
        _SM70_INDEXER_CUBLAS
        and current_platform.is_device_capability(70)
        and q.shape[0] >= _SM70_INDEXER_CUBLAS_MIN_ROWS
        and _qsa_indexer_cublas_shape_supported(q, k_cache, page_table)
    )


def _qsa_indexer_cublas_work_supported(rows: int, columns: int) -> bool:
    return rows * columns >= _SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS


def _use_sm70_qsa_lexicographic_topk(topk: int) -> bool:
    """Use the exact, deterministic selector for Volta QSA."""

    return topk == 512 and current_platform.is_device_capability(70)


def _sm70_qsa_lexicographic_topk_op():
    """Prefer an opt-in source-validation fragment over the wheel op."""

    sidecar = torch.ops._C_qsa_sm70
    if hasattr(sidecar, "qsa_lexicographic_topk"):
        return sidecar.qsa_lexicographic_topk
    return torch.ops._C.qsa_lexicographic_topk


def _sm70_qsa_lexicographic_topk_rows_op():
    """Decode-specialised selector for every row (grid = rows), or None.

    Follows the same source as _sm70_qsa_lexicographic_topk_op: when a
    validation sidecar is loaded, only its own rows op is used (an older
    sidecar build has none, so the old launch stays), never the wheel's rows
    op mixed with a sidecar M == 1 op.
    """

    sidecar = torch.ops._C_qsa_sm70
    if hasattr(sidecar, "qsa_lexicographic_topk"):
        return getattr(sidecar, "qsa_lexicographic_topk_decode_rows", None)
    return getattr(torch.ops._C, "qsa_lexicographic_topk_decode_rows", None)


def _sx_qsa_topk_rows_op(rows: int, max_rows: int | None = None):
    """SX_OPT_QSA_TOPK_ROWS admission: 2 <= rows <= 32 and the op is built.

    M == 1 keeps the existing launch (already the decode kernel); larger row
    counts (prefill chunks) keep the generic kernel. ``max_rows`` (MTP lane,
    SX_OPT_QSA_MTP_DECODE_ROWS) raises the 32-row limit for verify widths;
    the op has no row limit (grid = rows, one CTA per row).
    """

    limit = _SX_QSA_DECODE_MAX_ROWS if max_rows is None else max_rows
    if not _SX_OPT_QSA_TOPK_ROWS or not 2 <= rows <= limit:
        return None
    return _sm70_qsa_lexicographic_topk_rows_op()


def _qsa_visible_blocks(
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    rows = query_positions.shape[0]
    visible = torch.empty(rows, dtype=torch.int32, device=query_positions.device)
    if rows:
        _qsa_visible_blocks_kernel[(rows,)](
            token_to_req,
            query_positions,
            sequence_lengths,
            visible,
            rows,
            sequence_lengths.shape[0],
            COMPRESS_RATIO=compress_ratio,
            num_warps=1,
        )
    return visible


def _qsa_gather_single_request_keys(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty(
        (columns, k_cache.shape[3]), dtype=k_cache.dtype, device=k_cache.device
    )
    valid = torch.empty(columns, dtype=torch.uint8, device=k_cache.device)
    if columns:
        _qsa_gather_single_request_keys_kernel[(columns,)](
            k_cache,
            page_table[0],
            keys,
            valid,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(3),
            page_table.stride(1),
            keys.stride(0),
            columns,
            k_cache.shape[0],
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            HEAD_DIM=k_cache.shape[3],
            BLOCK_D=triton.next_power_of_2(k_cache.shape[3]),
            num_warps=4,
        )
    return keys, valid


def _qsa_mqa_cublas(
    q: torch.Tensor,
    keys: torch.Tensor,
    key_valid: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one request with Volta Tensor Cores and a fused QSA epilogue."""

    rows, num_heads, head_dim = q.shape
    columns = keys.shape[0]
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    if not rows or not columns:
        return logits, visible

    q2 = q.reshape(rows * num_heads, head_dim)
    bytes_per_column = max(1, q2.shape[0] * torch.float32.itemsize)
    tile_columns = max(
        256,
        min(columns, _SM70_INDEXER_SCORE_TILE_BYTES // bytes_per_column),
    )
    tile_columns = max(256, tile_columns // 256 * 256)
    for column_start in range(0, columns, tile_columns):
        column_stop = min(column_start + tile_columns, columns)
        width = column_stop - column_start
        # FP16 inputs, FP32 accumulation/output. This still selects Volta HMMA
        # through cuBLAS while avoiding an FP16 round trip before top-k.
        scores = torch.mm(
            q2,
            keys[column_start:column_stop].t(),
            out_dtype=torch.float32,
        )
        _qsa_relu_headsum_visible_kernel[(rows, triton.cdiv(width, 256))](
            scores,
            visible,
            key_valid,
            logits,
            scores.stride(0),
            scores.stride(1),
            logits.stride(0),
            width,
            column_start,
            math.sqrt(head_dim),
            NUM_HEADS=num_heads,
            BLOCK_N=256,
            num_warps=4,
        )
    return logits, visible


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the causal tail of the open group."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA expansion requires Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def _qsa_host_max_visible(
    page_table: torch.Tensor,
    rows: int,
    compress_ratio: int,
    query_start_loc_cpu: torch.Tensor | None,
    seq_lens_cpu: torch.Tensor | None,
) -> int | None:
    """Host value of int(_qsa_visible_blocks(...).max()) for one request.

    The QSA metadata builder writes position = seq_len - query_len + i for the
    i-th mapped row of a request and -1 for unmapped (padded) rows, so the last
    mapped row has position seq_len - 1. The per-row visible bound
    min((pos + 1) // r, seq_len // r) therefore peaks at seq_len // r, and
    padded rows contribute 0. Returns None when the host copies are missing or
    do not describe exactly this single-request batch; callers then keep the
    old device-side max (one host sync).
    """

    if (
        not _SX_OPT_QSA_HOST_BOUND
        or page_table.shape[0] != 1
        or seq_lens_cpu is None
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    seq_lens = _host_int_list(seq_lens_cpu, 1)
    query_starts = _host_int_list(query_start_loc_cpu, 2)
    if seq_lens is None or query_starts is None:
        return None
    sequence_length = seq_lens[0]
    mapped_rows = query_starts[1] - query_starts[0]
    if (
        query_starts[0] != 0
        or not 1 <= mapped_rows <= rows
        or sequence_length < mapped_rows
    ):
        return None
    return sequence_length // compress_ratio


def _plan_qsa_indexer_cublas_segments(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc_cpu: torch.Tensor | None,
    seq_lens_cpu: torch.Tensor | None,
    capacity_columns: int,
    block_topk: int,
    compress_ratio: int,
) -> list[tuple[int, int, int | None, int]] | None:
    """Split a multi-request indexer batch into cuBLAS and Triton row ranges.

    A request qualifies for the single-request cuBLAS scorer under exactly the
    conditions (and with exactly the score width) the old code applies when the
    request runs alone. Returns None, i.e. keep the old all-Triton path, when no
    request qualifies or the host metadata does not describe this batch.
    Segments are (start, end, request or None, score_columns).
    """

    rows = q.shape[0]
    num_requests = page_table.shape[0]
    if (
        not _SX_OPT_QSA_MIXED_CUBLAS
        or num_requests <= 1
        or rows < _SM70_INDEXER_CUBLAS_MIN_ROWS
        or query_start_loc_cpu is None
        or seq_lens_cpu is None
        or not _SM70_INDEXER_CUBLAS
        or not current_platform.is_device_capability(70)
        or not _qsa_indexer_cublas_shape_supported(q, k_cache, page_table[:1])
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    query_starts = _host_int_list(query_start_loc_cpu, num_requests + 1)
    seq_lens = _host_int_list(seq_lens_cpu, num_requests)
    if query_starts is None or seq_lens is None:
        return None
    if query_starts[0] != 0 or query_starts[-1] > rows:
        return None
    segments: list[tuple[int, int, int | None, int]] = []
    cursor = 0
    for request in range(num_requests):
        start, end = query_starts[request], query_starts[request + 1]
        if end < start:
            return None
        request_rows = end - start
        if request_rows < _SM70_INDEXER_CUBLAS_MIN_ROWS:
            continue
        sequence_length = seq_lens[request]
        if sequence_length < request_rows:
            return None
        score_columns = min(
            capacity_columns,
            max(block_topk, sequence_length // compress_ratio),
        )
        if not _qsa_indexer_cublas_work_supported(request_rows, score_columns):
            continue
        if cursor < start:
            segments.append((cursor, start, None, 0))
        segments.append((start, end, request, score_columns))
        cursor = end
    if not segments:
        return None
    if cursor < rows:
        segments.append((cursor, rows, None, 0))
    return segments


def _qsa_select_rows(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor,
    score_columns: int,
    contiguous_keys: torch.Tensor | None,
    key_valid: torch.Tensor | None,
    all_visible: torch.Tensor | None,
    launch_rows: int | None = None,
    topk_max_rows: int | None = None,
) -> None:
    """Chunked score -> top-k -> expand loop (verbatim baseline body).

    ``topk_max_rows`` (MTP lane) is the SX_OPT_QSA_TOPK_ROWS admission limit;
    None keeps the 32-row limit.
    """

    rows = q.shape[0]
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(score_columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = torch.empty(
        (chunk_rows, block_topk), dtype=torch.int32, device=q.device
    )
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        if contiguous_keys is not None and key_valid is not None:
            assert all_visible is not None
            logits, visible_blocks = _qsa_mqa_cublas(
                q[row_slice],
                contiguous_keys,
                key_valid,
                all_visible[row_slice],
            )
        else:
            logits, visible_blocks = qsa_mqa_paged(
                q[row_slice],
                k_cache,
                page_table,
                token_to_req[row_slice],
                query_positions[row_slice],
                sequence_lengths,
                compress_ratio,
                launch_rows=launch_rows,
            )
        blocks = blocks_buffer[: row_end - row_start]
        use_cooperative_topk = (
            blocks.shape[0] <= 32
            and logits.stride(0) % 4 == 0
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        if _use_sm70_qsa_lexicographic_topk(block_topk):
            logger.info_once(
                "Using exact SM70 QSA lexicographic top-k "
                "(score descending, block index ascending)."
            )
            rows_topk_op = _sx_qsa_topk_rows_op(blocks.shape[0], topk_max_rows)
            if rows_topk_op is not None:
                logger.info_once(
                    "Using exact SM70 QSA decode-specialised lexicographic "
                    "top-k for every row of 2 <= M <= %d batches "
                    "(SX_OPT_QSA_TOPK_ROWS).",
                    _SX_QSA_DECODE_MAX_ROWS,
                )
                if blocks.shape[0] > _SX_QSA_DECODE_MAX_ROWS:
                    logger.info_once(
                        "Using exact SM70 QSA decode-specialised lexicographic "
                        "top-k for MTP verify widths up to %d rows "
                        "(SX_OPT_QSA_MTP_DECODE_ROWS).",
                        topk_max_rows,
                    )
                rows_topk_op(
                    logits,
                    visible_blocks,
                    blocks,
                    block_topk,
                )
            else:
                _sm70_qsa_lexicographic_topk_op()(
                    logits,
                    visible_blocks,
                    blocks,
                    block_topk,
                )
        else:
            topk_op = (
                torch.ops._C.cooperative_topk
                if use_cooperative_topk
                else torch.ops._C.persistent_topk
            )
            topk_op(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                score_columns,
            )
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )


def _qsa_select_mixed_batch(
    segments: list[tuple[int, int, int | None, int]],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor,
    topk_max_rows: int | None = None,
) -> None:
    rows = q.shape[0]
    capacity_columns = page_table.shape[1] * k_cache.shape[1]
    for start, end, request, score_columns in segments:
        rows_slice = slice(start, end)
        if request is None:
            # Unchanged paged Triton scorer; keep the full-batch tile profile
            # so these rows stay bitwise equal to the old mixed-batch result.
            _qsa_select_rows(
                q[rows_slice],
                k_cache,
                page_table,
                token_to_req[rows_slice],
                query_positions[rows_slice],
                sequence_lengths,
                token_topk,
                compress_ratio,
                out[rows_slice],
                capacity_columns,
                None,
                None,
                None,
                launch_rows=rows,
                topk_max_rows=topk_max_rows,
            )
            continue
        # No per-call arguments: info_once de-duplicates on them, and every
        # new mixed-step shape would otherwise log another line.
        logger.info_once(
            "Using SM70 QSA indexer prefill cuBLAS path per request inside "
            "mixed batches (SX_OPT_QSA_MIXED_CUBLAS)."
        )
        page_row = page_table[request : request + 1]
        # Same launches, shapes and score width as the request run alone.
        all_visible = _qsa_visible_blocks(
            token_to_req[rows_slice],
            query_positions[rows_slice],
            sequence_lengths,
            compress_ratio,
        )
        if _SX_OPT_QSA_HOST_BOUND_CHECK:
            # Debug only (one host sync per request): the width planned from
            # the host seq_lens copy must equal the width the request would
            # get alone from the device-side visible maximum.
            device_max_visible = int(all_visible.max().item())
            device_score_columns = min(
                capacity_columns,
                max(token_topk // compress_ratio, device_max_visible),
            )
            if device_score_columns != score_columns:
                raise RuntimeError(
                    "QSA host score width disagrees with the device value for "
                    f"mixed-batch request {request} "
                    f"({score_columns} != {device_score_columns})"
                )
        # Memory-safety net for the host-derived width (no-op when exact).
        all_visible.clamp_(max=score_columns)
        contiguous_keys, key_valid = _qsa_gather_single_request_keys(
            k_cache, page_row, score_columns
        )
        _qsa_select_rows(
            q[rows_slice],
            k_cache,
            page_row,
            token_to_req[rows_slice],
            query_positions[rows_slice],
            sequence_lengths,
            token_topk,
            compress_ratio,
            out[rows_slice],
            score_columns,
            contiguous_keys,
            key_valid,
            all_visible,
            topk_max_rows=topk_max_rows,
        )


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    *,
    query_start_loc_cpu: torch.Tensor | None = None,
    seq_lens_cpu: torch.Tensor | None = None,
    sx_mtp_lane: SxQsaMtpLane | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices without host synchronization.

    ``query_start_loc_cpu`` / ``seq_lens_cpu`` are optional host copies of the
    batch's query starts and sequence lengths. With them the single-request
    cuBLAS path needs no device->host sync, and large prefill requests inside
    a mixed batch can use the cuBLAS scorer too. Without them the baseline
    behaviour is kept. A negative host sequence length means "unknown" (MTP
    lane: a request whose host length may be an optimistic upper bound) and
    keeps the device-side path for that request.

    ``sx_mtp_lane`` (MTP lane only) widens the decode top-k rows admission to
    the verify widths; None keeps the 1.8.0-dev2 limit.
    """

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    capacity_columns = page_table.shape[1] * k_cache.shape[1]
    score_columns = capacity_columns
    block_topk = token_topk // compress_ratio

    mixed_segments = _plan_qsa_indexer_cublas_segments(
        q,
        k_cache,
        page_table,
        query_start_loc_cpu,
        seq_lens_cpu,
        capacity_columns,
        block_topk,
        compress_ratio,
    )
    topk_max_rows = (
        None if sx_mtp_lane is None else _sx_decode_rows_cap(sx_mtp_lane, "topk_max_rows")
    )
    if mixed_segments is not None:
        _qsa_select_mixed_batch(
            mixed_segments,
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            token_topk,
            compress_ratio,
            out,
            topk_max_rows=topk_max_rows,
        )
        return out

    contiguous_keys: torch.Tensor | None = None
    key_valid: torch.Tensor | None = None
    all_visible: torch.Tensor | None = None
    if _use_sm70_qsa_indexer_cublas(q, k_cache, page_table):
        all_visible = _qsa_visible_blocks(
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
        )
        # cuBLAS has a rectangular host-side launch shape, unlike the paged
        # Triton kernel's device-side early exit. Bound it to this chunk's live
        # prefix so early-context prefill does not multiply the unused 140K
        # model-capacity tail. The bound comes from the host seq_lens copy when
        # available; otherwise it is one scalar sync per QSA layer.
        max_visible = _qsa_host_max_visible(
            page_table,
            rows,
            compress_ratio,
            query_start_loc_cpu,
            seq_lens_cpu,
        )
        host_bound = max_visible is not None
        if max_visible is None:
            max_visible = int(all_visible.max().item())
        elif _SX_OPT_QSA_HOST_BOUND_CHECK:
            device_max_visible = int(all_visible.max().item())
            if device_max_visible != max_visible:
                raise RuntimeError(
                    "QSA host visible bound disagrees with the device value "
                    f"({max_visible} != {device_max_visible})"
                )
        score_columns = min(
            capacity_columns,
            max(block_topk, max_visible),
        )
        if host_bound and _qsa_indexer_cublas_work_supported(rows, score_columns):
            # Memory-safety net: top-k reads visible_blocks columns of a
            # score_columns-wide buffer. With an exact host bound every row
            # already satisfies visible <= score_columns (no-op, bitwise).
            all_visible.clamp_(max=score_columns)
        if _qsa_indexer_cublas_work_supported(rows, score_columns):
            logger.info_once(
                "Using SM70 QSA indexer prefill cuBLAS path "
                "(single-request FP16, rows=%d, score_tile_mib=%d).",
                rows,
                _SM70_INDEXER_SCORE_TILE_BYTES // (1024 * 1024),
            )
            # Gather this request's paged MQA keys once, then reuse them across
            # every bounded logits chunk below. Generic, short-work and
            # multi-request batches keep the paged Triton fallback.
            contiguous_keys, key_valid = _qsa_gather_single_request_keys(
                k_cache, page_table, score_columns
            )
        else:
            score_columns = capacity_columns
            all_visible = None
    _qsa_select_rows(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
        out,
        score_columns,
        contiguous_keys,
        key_valid,
        all_visible,
        topk_max_rows=topk_max_rows,
    )
    return out


def _qsa_xqa_page4_shape_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    return (
        query_positions is not None
        and sequence_lengths is not None
        and q.dtype == torch.float16
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in (torch.float16, torch.uint8)
        and q.device
        == k_cache.device
        == v_cache.device
        == logical_indices.device
        == block_table.device
        == token_to_req.device
        == query_positions.device
        == sequence_lengths.device
        and q.ndim == 3
        and q.shape[1:] == (6, 256)
        and q.stride(2) == 1
        and k_cache.ndim == 4
        and v_cache.shape == k_cache.shape
        and k_cache.shape[2:] == (1, 256)
        and k_cache.shape[1] % 4 == 0
        and k_cache.stride(3) == v_cache.stride(3) == 1
        and k_cache.stride(1) == v_cache.stride(1) == 256
        and k_cache.stride(0) == v_cache.stride(0)
        and k_cache.stride(0) in (k_cache.shape[1] * 256, 2 * k_cache.shape[1] * 256)
        and logical_indices.shape == (q.shape[0], 2051)
        and logical_indices.dtype == torch.int32
        and logical_indices.stride(1) == 1
        and block_table.ndim == 2
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and token_to_req.shape == (q.shape[0],)
        and token_to_req.dtype == torch.int32
        and token_to_req.stride(0) == 1
        and query_positions.shape == (q.shape[0],)
        and query_positions.dtype == torch.int64
        and query_positions.stride(0) == 1
        and sequence_lengths.shape == (block_table.shape[0],)
        and sequence_lengths.dtype == torch.int32
        and sequence_lengths.stride(0) == 1
        and k_cache.shape[0] * (k_cache.stride(0) // (4 * 256))
        < _SM70_QSA_XQA_PAGE4_MARKER
    )


def _use_sm70_qsa_xqa_page4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    return (
        _SM70_QSA_XQA_PAGE4
        and current_platform.is_device_capability(70)
        and (
            q.shape[0] >= _SM70_QSA_XQA_PAGE4_MIN_ROWS
            or (k_cache.dtype == torch.uint8 and q.shape[0] > 16)
        )
        and _qsa_xqa_page4_shape_supported(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
        )
    )


def _qsa_xqa_page4_block_table(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    num_cache_blocks: int,
    page_size: int,
    physical_page_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if physical_page_stride is None:
        physical_page_stride = page_size // 4
    rows = logical_indices.shape[0]
    encoded_pages = torch.empty(
        (rows, _SM70_QSA_XQA_PAGE4_PAGES),
        dtype=torch.int64,
        device=logical_indices.device,
    )
    xqa_sequence_lengths = torch.empty(
        (rows,), dtype=torch.int32, device=logical_indices.device
    )
    _qsa_xqa_page4_table_kernel[(rows,)](
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        encoded_pages,
        xqa_sequence_lengths,
        logical_indices.stride(0),
        block_table.stride(0),
        encoded_pages.stride(0),
        rows,
        num_cache_blocks,
        block_table.shape[0],
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        COMPLETE_PAGES=2048 // 4,
        OUTPUT_PAGES=_SM70_QSA_XQA_PAGE4_PAGES,
        BLOCK_PAGES=1024,
        PHYSICAL_PAGE_STRIDE=physical_page_stride,
        num_warps=4,
    )
    sorted_pages = torch.sort(encoded_pages, dim=1).values
    physical_pages = torch.bitwise_and(
        sorted_pages,
        _SM70_QSA_XQA_PAGE4_MARKER - 1,
    ).to(torch.int32)
    return physical_pages, xqa_sequence_lengths


def _qsa_device_index(device: torch.device) -> int:
    return device.index if device.index is not None else -1


def _qsa_xqa_page4_partition_count(
    device: torch.device,
    num_partitions: int,
) -> torch.Tensor:
    """Device int32 [1] partition count read (never written) by the XQA kernel.

    Created once per (device, partitions) outside CUDA-graph capture: the old
    per-workspace torch.tensor() is a synchronous host copy, which is not
    permitted while a stream is being captured. If a capture reaches this
    before any eager call created the constant, a fill kernel recorded into
    that graph writes the value on every replay; that tensor is private to
    the call (another graph must not read it before this one replays) and is
    kept alive for the graph.
    """

    key = (_qsa_device_index(device), num_partitions)
    constant = _SM70_QSA_XQA_PAGE4_PARTITION_COUNTS.get(key)
    if constant is not None:
        return constant
    if torch.cuda.is_current_stream_capturing():
        logger.warning_once(
            "SM70 QSA page4 XQA partition count created during CUDA graph "
            "capture; recording a device fill instead of a host copy "
            "(SX_OPT_QSA_MTP_PAGE4_CAPTURE)."
        )
        constant = torch.full((1,), num_partitions, dtype=torch.int32, device=device)
        _SM70_QSA_PAGE4_GRAPH_RETIRED.append(constant)
        return constant
    constant = torch.tensor([num_partitions], dtype=torch.int32, device=device)
    _SM70_QSA_XQA_PAGE4_PARTITION_COUNTS[key] = constant
    return constant


def _qsa_xqa_page4_alloc(
    q: torch.Tensor,
    capacity: int,
    num_partitions: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporary_output = torch.empty(
        (capacity, q.shape[1], num_partitions, q.shape[2]),
        # The XQA kernel keeps E4M3 partials in FP32 and rejects FP16 scratch.
        dtype=torch.float32 if kv_cache_dtype == "fp8_e4m3" else torch.float16,
        device=q.device,
    )
    max_logits = torch.empty(
        (capacity, q.shape[1], num_partitions),
        dtype=torch.float32,
        device=q.device,
    )
    exp_sums = torch.empty_like(max_logits)
    return temporary_output, max_logits, exp_sums


def _qsa_xqa_page4_graph_workspace(
    q: torch.Tensor,
    num_partitions: int,
    rows: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scratch of captured XQA launches: shared by all graphs, never freed."""

    key = (
        _qsa_device_index(q.device),
        q.shape[1],
        q.shape[2],
        num_partitions,
        kv_cache_dtype == "fp8_e4m3",
    )
    workspace = _SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES.get(key)
    if workspace is None or workspace[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            logger.warning_once(
                "SM70 QSA page4 XQA graph workspace allocated during CUDA graph "
                "capture (not reserved beforehand); it comes from the graph "
                "pool without host copies and is never freed "
                "(SX_OPT_QSA_MTP_PAGE4_CAPTURE)."
            )
        if workspace is not None:
            # A graph captured earlier still addresses the smaller buffers.
            _SM70_QSA_PAGE4_GRAPH_RETIRED.append(workspace)
        capacity = 1 << (rows - 1).bit_length()
        workspace = (
            capacity,
            *_qsa_xqa_page4_alloc(q, capacity, num_partitions, kv_cache_dtype),
        )
        _SM70_QSA_XQA_PAGE4_GRAPH_WORKSPACES[key] = workspace
    _, temporary_output, max_logits, exp_sums = workspace
    return temporary_output[:rows], max_logits[:rows], exp_sums[:rows]


def _qsa_xqa_page4_workspace_dev2(
    q: torch.Tensor,
    num_partitions: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """1.8.0-dev2 workspace (SX_OPT_QSA_MTP_PAGE4_CAPTURE=0).

    Verbatim except for the scratch dtype, which follows the KV dtype.
    """
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    e4m3_output = kv_cache_dtype == "fp8_e4m3"
    key = (
        device_index,
        stream_id,
        q.shape[1],
        q.shape[2],
        num_partitions,
        e4m3_output,
    )
    workspace = _SM70_QSA_XQA_PAGE4_WORKSPACES.get(key)
    rows = q.shape[0]
    if workspace is None or workspace[0] < rows:
        capacity = 1 << (rows - 1).bit_length()
        temporary_output = torch.empty(
            (capacity, q.shape[1], num_partitions, q.shape[2]),
            dtype=torch.float32 if e4m3_output else torch.float16,
            device=q.device,
        )
        max_logits = torch.empty(
            (capacity, q.shape[1], num_partitions),
            dtype=torch.float32,
            device=q.device,
        )
        exp_sums = torch.empty_like(max_logits)
        active_num_partitions = torch.tensor(
            [num_partitions], dtype=torch.int32, device=q.device
        )
        workspace = (
            capacity,
            temporary_output,
            max_logits,
            exp_sums,
            active_num_partitions,
        )
        _SM70_QSA_XQA_PAGE4_WORKSPACES[key] = workspace
    _, temporary_output, max_logits, exp_sums, active_num_partitions = workspace
    return (
        temporary_output[:rows],
        max_logits[:rows],
        exp_sums[:rows],
        active_num_partitions,
    )


def _qsa_xqa_page4_workspace(
    q: torch.Tensor,
    num_partitions: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scratch + partition count for one row-wise XQA page4 launch.

    Eager launches keep the per-stream workspace (it may grow and free the
    smaller one: no graph holds it). Captured launches use the graph
    workspace. The partition count is the cached device constant in both.
    The kernel writes every scratch element it later reads, so buffer
    addresses never change results. FP16 and E4M3 K/V need different scratch
    dtypes and never share a workspace.
    """

    if not _SX_OPT_QSA_MTP_PAGE4_CAPTURE:
        return _qsa_xqa_page4_workspace_dev2(q, num_partitions, kv_cache_dtype)
    rows = q.shape[0]
    if torch.cuda.is_current_stream_capturing():
        temporary_output, max_logits, exp_sums = _qsa_xqa_page4_graph_workspace(
            q, num_partitions, rows, kv_cache_dtype
        )
        return (
            temporary_output,
            max_logits,
            exp_sums,
            _qsa_xqa_page4_partition_count(q.device, num_partitions),
        )
    device_index = _qsa_device_index(q.device)
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        device_index,
        stream_id,
        q.shape[1],
        q.shape[2],
        num_partitions,
        kv_cache_dtype == "fp8_e4m3",
    )
    workspace = _SM70_QSA_XQA_PAGE4_WORKSPACES.get(key)
    if workspace is None or workspace[0] < rows:
        capacity = 1 << (rows - 1).bit_length()
        workspace = (
            capacity,
            *_qsa_xqa_page4_alloc(q, capacity, num_partitions, kv_cache_dtype),
            _qsa_xqa_page4_partition_count(q.device, num_partitions),
        )
        _SM70_QSA_XQA_PAGE4_WORKSPACES[key] = workspace
    _, temporary_output, max_logits, exp_sums, active_num_partitions = workspace
    return (
        temporary_output[:rows],
        max_logits[:rows],
        exp_sums[:rows],
        active_num_partitions,
    )


def _qsa_grouped_page4_alloc(
    q: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped_pages = torch.empty(
        (capacity, _SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES),
        dtype=torch.int32,
        device=q.device,
    )
    token_masks = torch.empty(
        (capacity, _SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES),
        dtype=torch.uint32,
        device=q.device,
    )
    grouped_sequence_lengths = torch.empty(
        (capacity,), dtype=torch.int32, device=q.device
    )
    lse = torch.empty(
        (capacity * _SM70_QSA_GROUPED_PAGE4_QUERIES, q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    return grouped_pages, token_masks, grouped_sequence_lengths, lse


def _qsa_grouped_page4_graph_workspace(
    q: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped planner/forward scratch of captured launches (never freed)."""

    key = (_qsa_device_index(q.device), q.shape[1])
    workspace = _SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES.get(key)
    if workspace is None or workspace[0] < groups:
        if torch.cuda.is_current_stream_capturing():
            logger.warning_once(
                "SM70 QSA grouped page4 graph workspace allocated during CUDA "
                "graph capture (not reserved beforehand); it comes from the "
                "graph pool and is never freed (SX_OPT_QSA_MTP_PAGE4_CAPTURE)."
            )
        if workspace is not None:
            _SM70_QSA_PAGE4_GRAPH_RETIRED.append(workspace)
        capacity = 1 << (groups - 1).bit_length()
        workspace = (capacity, *_qsa_grouped_page4_alloc(q, capacity))
        _SM70_QSA_GROUPED_PAGE4_GRAPH_WORKSPACES[key] = workspace
    _, grouped_pages, token_masks, grouped_sequence_lengths, lse = workspace
    return (
        grouped_pages[:groups],
        token_masks[:groups],
        grouped_sequence_lengths[:groups],
        lse[: groups * _SM70_QSA_GROUPED_PAGE4_QUERIES],
    )


def _qsa_grouped_page4_workspace(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = q.shape[0] // _SM70_QSA_GROUPED_PAGE4_QUERIES
    if _SX_OPT_QSA_MTP_PAGE4_CAPTURE and torch.cuda.is_current_stream_capturing():
        grouped_pages, token_masks, grouped_sequence_lengths, lse = (
            _qsa_grouped_page4_graph_workspace(q, groups)
        )
        return grouped_pages, token_masks, grouped_sequence_lengths, lse[: q.shape[0]]
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    key = (device_index, stream_id)
    workspace = _SM70_QSA_GROUPED_PAGE4_WORKSPACES.get(key)
    if workspace is None or workspace[0] < groups:
        capacity = 1 << (groups - 1).bit_length()
        grouped_pages = torch.empty(
            (capacity, _SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES),
            dtype=torch.int32,
            device=q.device,
        )
        token_masks = torch.empty(
            (capacity, _SM70_QSA_GROUPED_PAGE4_OUTPUT_PAGES),
            dtype=torch.uint32,
            device=q.device,
        )
        grouped_sequence_lengths = torch.empty(
            (capacity,), dtype=torch.int32, device=q.device
        )
        lse = torch.empty(
            (capacity * _SM70_QSA_GROUPED_PAGE4_QUERIES, q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
        workspace = (
            capacity,
            grouped_pages,
            token_masks,
            grouped_sequence_lengths,
            lse,
        )
        _SM70_QSA_GROUPED_PAGE4_WORKSPACES[key] = workspace
    _, grouped_pages, token_masks, grouped_sequence_lengths, lse = workspace
    return (
        grouped_pages[:groups],
        token_masks[:groups],
        grouped_sequence_lengths[:groups],
        lse[: q.shape[0]],
    )


def _qsa_page4_reserve_graph_workspaces(
    q: torch.Tensor,
    kv_cache_dtype: str,
    selection_width: int,
    grouped_enabled: bool,
    graph_rows: int,
) -> None:
    """Allocate, outside capture, what the MTP lane's FULL graphs will use.

    Mirrors the capture-time routing of _qsa_sparse_paged_attention_sm70_xqa_page4
    (mixed-groups routing never runs under capture): with the grouped kernel
    a captured launch hands at most 7 remainder rows to XQA and at most
    graph_rows // 8 groups to the grouped kernel; the E4M3 fallback slices
    XQA into 16-row launches; otherwise XQA takes every row. Runs on the
    first eager page4 call of the lane (the startup profile run, and the
    eager warmup that precedes every capture), so captures do not allocate.
    Idempotent.
    """

    key = (
        _qsa_device_index(q.device),
        q.shape[1],
        q.shape[2],
        kv_cache_dtype,
        selection_width,
        bool(grouped_enabled),
        graph_rows,
    )
    if key in _SM70_QSA_PAGE4_GRAPH_RESERVED:
        return
    partition_size = (
        256 if kv_cache_dtype == "fp8_e4m3" else _SM70_QSA_XQA_PAGE4_PARTITION
    )
    num_partitions = math.ceil(selection_width / partition_size)
    group = _SM70_QSA_GROUPED_PAGE4_QUERIES
    if grouped_enabled:
        xqa_rows = min(graph_rows, group - 1)
        groups = graph_rows // group
    elif kv_cache_dtype == "fp8_e4m3":
        xqa_rows = min(graph_rows, 16)
        groups = 0
    else:
        xqa_rows = graph_rows
        groups = 0
    if xqa_rows > 0:
        _qsa_xqa_page4_partition_count(q.device, num_partitions)
        _qsa_xqa_page4_graph_workspace(q, num_partitions, xqa_rows, kv_cache_dtype)
    if groups > 0:
        _qsa_grouped_page4_graph_workspace(q, groups)
    _SM70_QSA_PAGE4_GRAPH_RESERVED.add(key)
    logger.info_once(
        "Reserved SM70 QSA page4 CUDA-graph workspaces outside capture "
        "(graph_rows=%d, xqa_rows=%d, grouped_groups=%d; "
        "SX_OPT_QSA_MTP_PAGE4_CAPTURE).",
        graph_rows,
        xqa_rows,
        groups,
    )


def _qsa_xqa_page4_physical_kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    microblock_stride = 4 * q.shape[2]
    if k_cache.stride(0) == k_cache.shape[1] * q.shape[2]:
        microblocks_per_cache_block = k_cache.shape[1] // 4
        physical_k_cache = k_cache.view(
            k_cache.shape[0] * microblocks_per_cache_block,
            4,
            1,
            q.shape[2],
        )
        physical_v_cache = v_cache.view_as(physical_k_cache)
    else:
        # The local FlashAttention ABI interleaves K and V inside every
        # physical cache block. The virtual page IDs carry that doubled block
        # stride, while this narrow view exposes a four-token page stride.
        physical_shape = (k_cache.shape[0], 4, 1, q.shape[2])
        physical_strides = (
            microblock_stride,
            q.shape[2],
            q.shape[2],
            1,
        )
        physical_k_cache = k_cache.as_strided(physical_shape, physical_strides)
        physical_v_cache = v_cache.as_strided(physical_shape, physical_strides)
    return physical_k_cache, physical_v_cache


def _qsa_grouped_page4_forward(
    flash_attn_v100_cuda,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    grouped_pages: torch.Tensor,
    token_masks: torch.Tensor,
    grouped_sequence_lengths: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    forward_args = (
        q,
        k_cache,
        v_cache,
        out,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        lse,
        softmax_scale,
    )
    abi_version = _qsa_grouped_page4_abi_version(flash_attn_v100_cuda)
    if abi_version >= 2:
        flash_attn_v100_cuda.grouped_sparse_page4_fwd(
            *forward_args,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return

    # ABI v1 only supports the original FP16 K/V contract. The route
    # eligibility check rejects quantized K/V before the planner runs.
    assert abi_version == 1 and kv_cache_dtype in ("auto", "float16")
    flash_attn_v100_cuda.grouped_sparse_page4_fwd(*forward_args)


def _qsa_sparse_paged_attention_sm70_grouped_page4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    flash_attn_v100_cuda,
) -> torch.Tensor:
    grouped_pages, token_masks, grouped_sequence_lengths, lse = (
        _qsa_grouped_page4_workspace(q)
    )
    physical_page_stride = k_cache.stride(0) // (4 * q.shape[2])
    flash_attn_v100_cuda.grouped_sparse_page4_plan_fwd(
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        k_cache.shape[1],
        physical_page_stride,
        k_cache.shape[0],
    )
    if _SM70_QSA_GROUPED_PAD_FIX and kv_cache_dtype == "fp8_e4m3":
        # The planner pads each category to a multiple of 8 with (physical
        # microblock 0 = null block, mask 0) and counts the padding in
        # seq_len. The forward loads page 0's K/V for those padded rows and
        # sets P=0, but 0 * NaN survives the P@V MMA when page 0 holds FP16
        # GDN state whose bytes decode to E4M3 NaN. Repoint every mask==0
        # (padding) entry at this group's first real microblock (column 0;
        # real entries always carry a nonzero mask). Only E4M3 needs it: FP16
        # K/V read back from the null block are finite. torch.where + copy_
        # have no host sync, so the launch stays CUDA-graph capturable.
        grouped_pages.copy_(
            torch.where(token_masks == 0, grouped_pages[:, :1], grouped_pages)
        )
    physical_k_cache, physical_v_cache = _qsa_xqa_page4_physical_kv(q, k_cache, v_cache)
    _qsa_grouped_page4_forward(
        flash_attn_v100_cuda,
        q,
        physical_k_cache,
        physical_v_cache,
        out,
        grouped_pages,
        token_masks,
        grouped_sequence_lengths,
        lse,
        q.shape[2] ** -0.5,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    logger.info_once(
        "Using SM70 grouped QSA Flash-V100 page4 prefill route (rows=%d, groups=%d).",
        q.shape[0],
        q.shape[0] // _SM70_QSA_GROUPED_PAGE4_QUERIES,
    )
    return out


def _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    flash_attn_v100_cuda,
) -> torch.Tensor:
    virtual_block_table, xqa_sequence_lengths = _qsa_xqa_page4_block_table(
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        k_cache.shape[0],
        k_cache.shape[1],
        k_cache.stride(0) // (4 * q.shape[2]),
    )
    # Generic E4M3 G6 XQA supports P256 for virtual page4 caches. Its P1024
    # specialization is restricted to the page1568 layout.
    partition_size = (
        256 if kv_cache_dtype == "fp8_e4m3" else _SM70_QSA_XQA_PAGE4_PARTITION
    )
    num_partitions = math.ceil(logical_indices.shape[1] / partition_size)
    temporary_output, max_logits, exp_sums, active_num_partitions = (
        _qsa_xqa_page4_workspace(q, num_partitions, kv_cache_dtype)
    )
    physical_k_cache, physical_v_cache = _qsa_xqa_page4_physical_kv(q, k_cache, v_cache)
    flash_attn_v100_cuda.decode_paged_xqa_fwd(
        q,
        physical_k_cache,
        physical_v_cache,
        out,
        virtual_block_table,
        xqa_sequence_lengths,
        temporary_output,
        max_logits,
        exp_sums,
        active_num_partitions,
        q.shape[2] ** -0.5,
        partition_size,
        num_partitions,
        kv_cache_dtype,
        k_scale,
        v_scale,
        -1,
        -1,
        0,
    )
    logger.info_once(
        "Using SM70 QSA Flash-V100 XQA page4 prefill route (rows=%d, partitions=%d).",
        q.shape[0],
        num_partitions,
    )
    return out


def _plan_qsa_page4_request_segments(
    query_start_loc_cpu: torch.Tensor | None,
    rows: int,
) -> list[tuple[int, int, bool]] | None:
    """Plan grouped/row-wise page4 ranges that never mix requests in a group.

    Each request contributes floor(len / 8) * 8 grouped rows starting at its
    first row, and its remaining rows (all rows of a decode or short request)
    go to the row-wise XQA kernel. Adjacent ranges of the same kind merge;
    merged grouped ranges still only contain whole 8-row groups of a single
    request because every grouped range length is a multiple of 8. Rows after
    the last mapped token (padding) are row-wise. Returns None if the host
    query starts do not describe this batch.
    """

    query_starts = _host_int_list(query_start_loc_cpu)
    if query_starts is None or len(query_starts) < 2:
        return None
    if query_starts[0] != 0 or query_starts[-1] > rows:
        return None
    group = _SM70_QSA_GROUPED_PAGE4_QUERIES
    segments: list[tuple[int, int, bool]] = []

    def add(start: int, end: int, grouped: bool) -> None:
        if end <= start:
            return
        if segments and segments[-1][2] == grouped and segments[-1][1] == start:
            segments[-1] = (segments[-1][0], end, grouped)
        else:
            segments.append((start, end, grouped))

    for request in range(len(query_starts) - 1):
        start, end = query_starts[request], query_starts[request + 1]
        if end < start:
            return None
        grouped_end = start + (end - start) // group * group
        add(start, grouped_end, True)
        add(grouped_end, end, False)
    add(query_starts[-1], rows, False)
    return segments


def _qsa_page4_old_group_max_requests(
    query_start_loc_cpu: torch.Tensor | None,
    rows: int,
) -> int:
    """Most requests that share one 8-row group under the old whole-batch split.

    A request boundary b splits the old group b // 8 unless it is 8-aligned;
    rows after the last mapped token (padding) count as one more "request".
    Host-only; returns 0 when the host query starts are unusable.
    """

    query_starts = _host_int_list(query_start_loc_cpu)
    if not query_starts:
        return 0
    group = _SM70_QSA_GROUPED_PAGE4_QUERIES
    grouped_rows = rows // group * group
    interior: dict[int, int] = {}
    for boundary in query_starts[1:]:
        if 0 < boundary < grouped_rows and boundary % group:
            interior[boundary // group] = interior.get(boundary // group, 0) + 1
    return 1 + max(interior.values(), default=0)


def _qsa_page4_segments_match_baseline(
    segments: list[tuple[int, int, bool]],
    rows: int,
) -> bool:
    """Whether the per-request plan equals the old whole-batch split."""

    grouped_rows = rows // _SM70_QSA_GROUPED_PAGE4_QUERIES * (
        _SM70_QSA_GROUPED_PAGE4_QUERIES
    )
    baseline = []
    if grouped_rows:
        baseline.append((0, grouped_rows, True))
    if grouped_rows < rows:
        baseline.append((grouped_rows, rows, False))
    return segments == baseline


def _qsa_sparse_paged_attention_sm70_page4_segments(
    segments: list[tuple[int, int, bool]],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    flash_attn_v100_cuda,
) -> torch.Tensor:
    """Run request-aligned grouped ranges plus one row-wise XQA launch."""

    # No per-call arguments: info_once de-duplicates on them.
    logger.info_once(
        "Using SM70 QSA request-aligned grouped page4 routing for mixed "
        "batches (SX_OPT_QSA_MIXED_GROUPS)."
    )
    rowwise = [(start, end) for start, end, grouped in segments if not grouped]
    for start, end, grouped in segments:
        if not grouped:
            continue
        # The planner and grouped kernel are CUDA; row offsets need no
        # particular alignment and every group holds rows of one request.
        _qsa_sparse_paged_attention_sm70_grouped_page4(
            q[start:end],
            k_cache,
            v_cache,
            logical_indices[start:end],
            block_table,
            token_to_req[start:end],
            query_positions[start:end],
            sequence_lengths,
            out[start:end],
            kv_cache_dtype,
            k_scale,
            v_scale,
            flash_attn_v100_cuda,
        )
    if not rowwise:
        return out
    if len(rowwise) == 1 and rowwise[0][0] % 4 == 0:
        # A single range at a 4-row-aligned offset keeps the 16-byte pointer
        # alignment of every metadata slice, so the Triton table kernel reuses
        # its existing specialisation. Row-wise XQA is per-row independent.
        start, end = rowwise[0]
        _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
            q[start:end],
            k_cache,
            v_cache,
            logical_indices[start:end],
            block_table,
            token_to_req[start:end],
            query_positions[start:end],
            sequence_lengths,
            out[start:end],
            kv_cache_dtype,
            k_scale,
            v_scale,
            flash_attn_v100_cuda,
        )
        return out
    # Several short ranges (decodes plus per-request remainders): gather them
    # into fresh contiguous buffers, launch the row-wise kernel once and
    # scatter the rows back. Device-side copies only; no host sync.
    rowwise_q = torch.cat([q[start:end] for start, end in rowwise])
    rowwise_out = torch.empty_like(rowwise_q)
    _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
        rowwise_q,
        k_cache,
        v_cache,
        torch.cat([logical_indices[start:end] for start, end in rowwise]),
        block_table,
        torch.cat([token_to_req[start:end] for start, end in rowwise]),
        torch.cat([query_positions[start:end] for start, end in rowwise]),
        sequence_lengths,
        rowwise_out,
        kv_cache_dtype,
        k_scale,
        v_scale,
        flash_attn_v100_cuda,
    )
    offset = 0
    for start, end in rowwise:
        out[start:end].copy_(rowwise_out[offset : offset + end - start])
        offset += end - start
    return out


def _qsa_sparse_paged_attention_sm70_xqa_page4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    out: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    query_start_loc_cpu: torch.Tensor | None = None,
    sx_page4_graph_rows: int = 0,
) -> torch.Tensor | None:
    try:
        from flash_attn_v100.flash_attn_interface import flash_attn_v100_cuda
    except ImportError:
        logger.warning_once(
            "SM70 QSA page4 XQA route is unavailable because Flash-V100 "
            "could not be imported; using Triton sparse attention."
        )
        return None
    if not hasattr(flash_attn_v100_cuda, "decode_paged_xqa_fwd"):
        logger.warning_once(
            "SM70 QSA page4 XQA route is unavailable in this Flash-V100 build; "
            "using Triton sparse attention."
        )
        return None

    grouped_enabled = _SM70_QSA_GROUPED_PAGE4 and _qsa_grouped_page4_supported(
        flash_attn_v100_cuda, kv_cache_dtype
    )
    if (
        sx_page4_graph_rows > 0
        and _SX_OPT_QSA_MTP_PAGE4_CAPTURE
        and not torch.cuda.is_current_stream_capturing()
    ):
        # MTP lane: FULL verify graphs reach page4 at >= 64 rows. Reserve
        # their workspaces now so capture neither allocates nor copies.
        _qsa_page4_reserve_graph_workspaces(
            q,
            kv_cache_dtype,
            logical_indices.shape[1],
            grouped_enabled,
            sx_page4_graph_rows,
        )
    if (
        grouped_enabled
        and _SX_OPT_QSA_MIXED_GROUPS
        and query_start_loc_cpu is not None
        and kv_cache_dtype in ("auto", "float16")
        and not torch.cuda.is_current_stream_capturing()
    ):
        # The grouped kernel runs one CTA per 8-row group and that CTA walks
        # the union of the 8 rows' selections. Rows of one prefill request
        # share most pages, but 8 decode rows of 8 different requests have
        # disjoint selections (8x the pages), so in a mixed step those groups
        # become the kernel's critical path. Group rows per request instead.
        segments = _plan_qsa_page4_request_segments(query_start_loc_cpu, q.shape[0])
        if (
            segments is not None
            and len(segments) <= _SX_QSA_MIXED_MAX_SEGMENTS
            and not _qsa_page4_segments_match_baseline(segments, q.shape[0])
            and _qsa_page4_old_group_max_requests(query_start_loc_cpu, q.shape[0])
            >= _SX_QSA_MIXED_MIN_GROUP_REQUESTS
        ):
            return _qsa_sparse_paged_attention_sm70_page4_segments(
                segments,
                q,
                k_cache,
                v_cache,
                logical_indices,
                block_table,
                token_to_req,
                query_positions,
                sequence_lengths,
                out,
                kv_cache_dtype,
                k_scale,
                v_scale,
                flash_attn_v100_cuda,
            )
    if grouped_enabled:
        grouped_rows = (
            q.shape[0] // _SM70_QSA_GROUPED_PAGE4_QUERIES
        ) * _SM70_QSA_GROUPED_PAGE4_QUERIES
        if grouped_rows:
            _qsa_sparse_paged_attention_sm70_grouped_page4(
                q[:grouped_rows],
                k_cache,
                v_cache,
                logical_indices[:grouped_rows],
                block_table,
                token_to_req[:grouped_rows],
                query_positions[:grouped_rows],
                sequence_lengths,
                out[:grouped_rows],
                kv_cache_dtype,
                k_scale,
                v_scale,
                flash_attn_v100_cuda,
            )
        if grouped_rows == q.shape[0]:
            return out

        logger.info_once(
            "Splitting a non-grouped page4 batch across grouped/XQA routes "
            "(rows=%d, grouped_rows=%d, kv_cache_dtype=%s).",
            q.shape[0],
            grouped_rows,
            kv_cache_dtype,
        )
        _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
            q[grouped_rows:],
            k_cache,
            v_cache,
            logical_indices[grouped_rows:],
            block_table,
            token_to_req[grouped_rows:],
            query_positions[grouped_rows:],
            sequence_lengths,
            out[grouped_rows:],
            kv_cache_dtype,
            k_scale,
            v_scale,
            flash_attn_v100_cuda,
        )
        return out

    if kv_cache_dtype == "fp8_e4m3" and q.shape[0] > 16:
        # The generic E4M3 XQA kernel accepts at most 16 query rows. Scheduler
        # iterations can mix a large prefill (or catch-up chunk) with decode
        # rows. If an older Flash-V100 build lacks the quantized grouped ABI,
        # retain correctness by slicing the work into supported XQA batches.
        # This avoids both an invalid B>16 launch and the larger Triton split-K
        # fallback workspace.
        logger.info_once(
            "Splitting an E4M3 page4 batch across supported XQA launches "
            "because this Flash-V100 build lacks the quantized grouped ABI "
            "(rows=%d).",
            q.shape[0],
        )
        for row_start in range(0, q.shape[0], 16):
            row_end = min(row_start + 16, q.shape[0])
            _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
                q[row_start:row_end],
                k_cache,
                v_cache,
                logical_indices[row_start:row_end],
                block_table,
                token_to_req[row_start:row_end],
                query_positions[row_start:row_end],
                sequence_lengths,
                out[row_start:row_end],
                kv_cache_dtype,
                k_scale,
                v_scale,
                flash_attn_v100_cuda,
            )
        return out

    return _qsa_sparse_paged_attention_sm70_xqa_page4_batch(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        out,
        kv_cache_dtype,
        k_scale,
        v_scale,
        flash_attn_v100_cuda,
    )


def _use_sm70_qsa_resolved_indices(
    q, k_cache, indices, kv_cache_dtype, *, max_rows: int | None = None
):
    """Admit only the measured checkpoint-FP16 TP4 decode cache geometry.

    Baseline: M == 1 with 400-token pages. With SX_OPT_QSA_RESOLVED_ROWS the
    same address-only rewrite covers every decode width 1 <= M <= 32 and any
    page size whose physical token slots fit in int32 (e.g. the 784-token
    pages of the mamba-align deployment). The resolver only replaces the
    partial kernel's dependent page-table load with a precomputed physical
    slot: logical order, duplicates and invalid slots are preserved, so the
    attention arithmetic and its output are bitwise unchanged. ``max_rows``
    (MTP lane, SX_OPT_QSA_MTP_DECODE_ROWS) widens the 32-row limit to the
    verify widths; the resolver is per row (its own request's page table).
    """
    if _SX_OPT_QSA_RESOLVED_ROWS:
        rows = q.shape[0] if len(q.shape) == 3 else 0
        page_size = k_cache.shape[1] if len(k_cache.shape) == 4 else 0
        limit = _SX_QSA_DECODE_MAX_ROWS if max_rows is None else max_rows
        return bool(
            current_platform.is_device_capability(70)
            and 1 <= rows <= limit
            and q.shape[1:] == (6, 256)
            and q.dtype == k_cache.dtype == torch.float16
            and page_size > 0
            and k_cache.shape[2:] == (1, 256)
            and k_cache.shape[0] * page_size < 2**31
            and indices.shape == (rows, 2051)
            and indices.dtype == torch.int32
            and kv_cache_dtype in ("auto", "float16")
        )
    return bool(
        current_platform.is_device_capability(70)
        and q.shape == (1, 6, 256)
        and q.dtype == k_cache.dtype == torch.float16
        and k_cache.shape[1:] == (400, 1, 256)
        and k_cache.shape[0] * 400 < 2**31
        and indices.shape == (1, 2051)
        and indices.dtype == torch.int32
        and kv_cache_dtype in ("auto", "float16")
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    output_gate: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    *,
    query_start_loc_cpu: torch.Tensor | None = None,
    sx_mtp_lane: SxQsaMtpLane | None = None,
) -> torch.Tensor:
    """Run sparse GQA over paged FP16/BF16 or calibrated E4M3 K/V.

    ``query_start_loc_cpu`` (optional host copy of the batch's query starts)
    lets the SM70 grouped page4 prefill route keep every 8-row group inside a
    single request; without it the baseline grouping is used.

    ``sx_mtp_lane`` (MTP lane only) widens the decode-row gates (two-warp
    partial, resolved rows) to the verify widths and reserves the page4 CUDA
    graph workspaces; None keeps the 1.8.0-dev2 gates.
    """

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires CUDA and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QSA sparse attention requires FP16 or BF16 queries")
    kv_e4m3 = kv_cache_dtype in ("fp8", "fp8_e4m3")
    if kv_e4m3:
        if k_cache.dtype != torch.uint8 or v_cache.dtype != torch.uint8:
            raise ValueError("QSA E4M3 K/V caches must use uint8 storage")
        if not math.isfinite(k_scale) or not math.isfinite(v_scale):
            raise ValueError("QSA E4M3 K/V scales must be finite")
        if k_scale <= 0.0 or v_scale <= 0.0:
            raise ValueError("QSA E4M3 K/V scales must be positive")
    else:
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError(f"Unsupported QSA K/V cache dtype: {kv_cache_dtype}")
        if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
            raise ValueError("QSA unquantized K/V caches must match query dtype")
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    output_gate_view = output_gate.view_as(q) if output_gate is not None else None
    if output_gate_view is not None:
        if output_gate_view.dtype != q.dtype or output_gate_view.device != q.device:
            raise ValueError("QSA output gate must match the query dtype and device")
        if output_gate_view.stride(2) != 1:
            raise ValueError("QSA output gate must be contiguous in head dimension")
    if not q.shape[0]:
        return out

    if _use_sm70_qsa_xqa_page4(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
    ):
        assert query_positions is not None and sequence_lengths is not None
        xqa_output = _qsa_sparse_paged_attention_sm70_xqa_page4(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            out,
            "fp8_e4m3" if kv_e4m3 else "auto",
            k_scale,
            v_scale,
            query_start_loc_cpu=query_start_loc_cpu,
            sx_page4_graph_rows=(
                0 if sx_mtp_lane is None else int(sx_mtp_lane.page4_graph_rows)
            ),
        )
        if xqa_output is not None:
            if output_gate_view is not None:
                _qsa_output_gate(xqa_output, output_gate_view)
            return xqa_output

    resolved_indices = _use_sm70_qsa_resolved_indices(
        q,
        k_cache,
        logical_indices,
        kv_cache_dtype,
        max_rows=(
            None
            if sx_mtp_lane is None
            else _sx_decode_rows_cap(sx_mtp_lane, "resolved_max_rows")
        ),
    )
    if resolved_indices and _SX_OPT_QSA_RESOLVED_ROWS:
        resolved_rows, resolved_topk = logical_indices.shape
        # A fresh contiguous buffer: the resolver writes at row * TOPK while
        # the partial kernel reads with this tensor's own row stride.
        physical_indices = torch.empty(
            (resolved_rows, resolved_topk),
            dtype=logical_indices.dtype,
            device=logical_indices.device,
        )
        _qsa_resolve_physical_indices_kernel[
            (resolved_rows, triton.cdiv(resolved_topk, 256))
        ](
            logical_indices,
            block_table,
            token_to_req,
            physical_indices,
            logical_indices.stride(0),
            block_table.stride(0),
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=resolved_topk,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            BLOCK=256,
            num_warps=4,
        )
        logical_indices = physical_indices
    elif resolved_indices:
        physical_indices = torch.empty_like(logical_indices)
        _qsa_resolve_physical_indices_kernel[(1, triton.cdiv(2051, 256))](
            logical_indices,
            block_table,
            token_to_req,
            physical_indices,
            logical_indices.stride(0),
            block_table.stride(0),
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=2051,
            PAGE_SIZE=400,
            PAGE_TABLE_WIDTH=block_table.shape[1],
            BLOCK=256,
            num_warps=4,
        )
        logical_indices = physical_indices

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    block_n, target_splits, partial_warps = _qsa_sparse_launch_profile(
        base_programs,
        block_m,
        not current_platform.has_device_capability(80),
    )

    if _use_sm70_qsa_two_warp_partial(
        q.shape[0],
        group_size,
        head_dim,
        max_rows=(
            None
            if sx_mtp_lane is None
            else _sx_decode_rows_cap(sx_mtp_lane, "two_warp_max_rows")
        ),
    ):
        # Exact Qwen4Exp TP4 decode family. Two warps preserve the existing
        # split/merge arithmetic and cut the partial-kernel time on V100.
        partial_warps = 2
        if q.shape[0] > _SX_QSA_DECODE_MAX_ROWS:
            logger.info_once(
                "Using SM70 QSA two-warp sparse partial for MTP verify widths "
                "up to %d rows (SX_OPT_QSA_MTP_DECODE_ROWS).",
                _sx_decode_rows_cap(sx_mtp_lane, "two_warp_max_rows"),
            )

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0) if output_gate_view is not None else 0,
        output_gate_view.stride(1) if output_gate_view is not None else 0,
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        k_scale,
        v_scale,
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        KV_E4M3=kv_e4m3,
        RESOLVED_INDICES=resolved_indices,
        num_warps=partial_warps,
        num_stages=2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0) if output_gate_view is not None else 0,
        output_gate_view.stride(1) if output_gate_view is not None else 0,
        q.shape[0],
        v_scale,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        KV_E4M3=kv_e4m3,
        num_warps=2,
        num_stages=1,
    )
    return out


def _qsa_sparse_launch_profile(
    base_programs: int,
    block_m: int,
    is_pre_ampere: bool,
) -> tuple[int, int, int]:
    """Return BLOCK_N, target splits, and warps for sparse QSA."""
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    if is_pre_ampere and block_n == 64:
        # Pre-Ampere: the 64-column tile at D=256 does not fit Turing's
        # 64 KiB shared-memory limit (Triton OutOfResources -- the kernel
        # cannot launch on SM75 at all), and two warps serialize the D=256
        # tensor-core work on V100. A 16-column tile with four warps
        # launches on both and measured 1.16-2.6x faster than the best
        # previously runnable profile across the 64..2048-row prefill
        # regimes (V100-PCIE-32GB and Quadro RTX 8000, see #441).
        partial_warps = 4
        block_n = 16
    return block_n, target_splits, partial_warps


def _use_sm70_qsa_two_warp_partial(
    num_query_tokens: int,
    group_size: int,
    head_dim: int,
    *,
    max_rows: int | None = None,
) -> bool:
    """Gate the bitwise small-batch SM70 sparse-QSA launch policy.

    With SX_OPT_QSA_TWO_WARP32 the gate covers M <= 32 (baseline M <= 16).
    M17..31 compile to the same constexpr set as M16 (BLOCK_N 16, 32 splits);
    M32 is the 8-split variant. The per-tile code is identical for every trip
    count, so the two-warp result equals the four-warp one bitwise.
    ``max_rows`` (MTP lane, SX_OPT_QSA_MTP_DECODE_ROWS) replaces the 32-row
    limit for verify widths: M33..63 use the M32 constexpr set (8 splits),
    only the grid grows.
    """
    if _SX_OPT_QSA_TWO_WARP32:
        max_query_tokens = _SX_QSA_DECODE_MAX_ROWS if max_rows is None else max_rows
    else:
        max_query_tokens = 16
    return bool(
        0 < num_query_tokens <= max_query_tokens
        and group_size == 6
        and head_dim == 256
        and current_platform.is_device_capability(70)
    )


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
