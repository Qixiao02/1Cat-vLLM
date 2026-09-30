# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import math
import os
from typing import ClassVar, cast

import torch
from torch import nn

from vllm import envs
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    _sm70_dump_qwen_layer_tensor,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from .indexer_qsa import QSAIndexer

logger = init_logger(__name__)

# SX_OPT_QSA_HOST_METADATA (default "1"; "0" = baseline): attach host copies of
# query_start_loc and seq_lens to the QSA main-attention metadata so the QSA
# ops can size launches without a device->host sync (see the SX_OPT_QSA_*
# switches in ops/qsa.py). Attached without speculative decoding, where
# seq_lens_cpu_upper_bound is exact for every row, and (batch 3a,
# SX_OPT_QSA_MTP_HOST_METADATA, default "1") in the admitted MTP lane, where
# only the requests that may carry optimistic lengths are masked (see
# _sx_qsa_exact_prefill_seq_lens).
_SX_OPT_QSA_HOST_METADATA = os.environ.get("SX_OPT_QSA_HOST_METADATA", "1") != "0"
_SX_OPT_QSA_MTP_HOST_METADATA = (
    os.environ.get("SX_OPT_QSA_MTP_HOST_METADATA", "1") != "0"
)
_SX_QSA_QUERY_START_LOC_CPU = "sx_qsa_query_start_loc_cpu"
_SX_QSA_SEQ_LENS_CPU = "sx_qsa_seq_lens_cpu"
# The Qwen4Exp MTP proposer runs 1..7 sequential draft steps.
_SX_QSA_MTP_MAX_K = 7


def _sx_qsa_local_mtp_lane_contract(vllm_config: object) -> bool:
    """Fallback MTP-lane contract when vllm/config/vllm.py lacks the shared one.

    The same checks as _is_sm70_qwen38_mtp_lane_contract: the SX_OPT_MTP_LANE
    master switch, the Qwen4Exp MTP proposer with 1 <= k <= 7 sequential
    drafts (as many state tokens as drafts, i.e. no tree verify), standard
    rejection sampling, and the no-MTP dual-compile topology (Qwen3.8 FP16
    TP4) on the target model config. Malformed configs fail closed.
    """

    # Same master switch as vllm/config/vllm.py::_sx_mtp_lane_enabled, so
    # SX_OPT_MTP_LANE=0 also disables the QSA MTP-lane items when only the
    # QSA files are overlaid on an image without the shared contract.
    if os.environ.get("SX_OPT_MTP_LANE", "1").strip() == "0":
        return False
    spec = getattr(vllm_config, "speculative_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if spec is None or parallel_config is None:
        return False
    try:
        use_qwen4_exp_mtp = getattr(spec, "use_qwen4_exp_mtp", None)
        if not callable(use_qwen4_exp_mtp) or not use_qwen4_exp_mtp():
            return False
        num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0)
        state_tokens_fn = getattr(spec, "num_speculative_state_tokens", None)
        state_tokens = int(state_tokens_fn()) if callable(state_tokens_fn) else num_spec
    except Exception:  # noqa: BLE001 - partial configs fail closed
        return False
    if (
        getattr(spec, "method", None) != "mtp"
        or not 1 <= num_spec <= _SX_QSA_MTP_MAX_K
        or state_tokens != num_spec
        or getattr(spec, "parallel_drafting", False)
        or getattr(spec, "rejection_sample_method", "standard") != "standard"
    ):
        return False
    from vllm.config.vllm import _is_sm70_qwen38_nomtp_dual_compile_contract

    return any(
        candidate is not None
        and _is_sm70_qwen38_nomtp_dual_compile_contract(
            candidate, None, parallel_config
        )
        for candidate in (
            getattr(vllm_config, "model_config", None),
            getattr(spec, "target_model_config", None),
        )
    )


def _sx_qsa_mtp_lane_contract(vllm_config: object) -> bool:
    """Whether the QSA MTP-lane fast paths apply to this configuration.

    speculative_config.method == "mtp" with the Qwen4Exp MTP drafter on the
    admitted Qwen3.8 TP4 topology, on SM70: the shared
    _is_sm70_qwen38_mtp_lane_contract (which also honours SX_OPT_MTP_LANE)
    when vllm/config/vllm.py provides it, otherwise the same local checks.
    """

    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "mtp":
        return False
    try:
        from vllm.config.vllm import _is_sm70_qwen38_mtp_lane_contract
    except ImportError:
        admitted = _sx_qsa_local_mtp_lane_contract(vllm_config)
    else:
        admitted = _is_sm70_qwen38_mtp_lane_contract(
            getattr(vllm_config, "model_config", None),
            spec,
            getattr(vllm_config, "parallel_config", None),
        )
    return bool(admitted) and current_platform.is_device_capability(70)


def _sx_qsa_mtp_verify_rows(vllm_config: object) -> int:
    """Query rows of one MTP verify request: 1 + num_speculative_tokens."""

    spec = getattr(vllm_config, "speculative_config", None)
    return 1 + int(getattr(spec, "num_speculative_tokens", 0) or 0)


def _sx_qsa_mtp_page4_graph_rows(vllm_config: object) -> int:
    """Widest FULL CUDA graph of the MTP lane (target verify / draft prefill).

    SX_OPT_QSA_MTP_PAGE4_GRAPH_ROWS overrides it (0 = no pre-capture reserve).
    Otherwise the upper bound of every capture-size rewrite: the configured
    maximum, the listed sizes and max_num_seqs * (1 + k), capped by
    max_num_batched_tokens. 0 without CUDA graphs.
    """

    override = os.environ.get("SX_OPT_QSA_MTP_PAGE4_GRAPH_ROWS", "").strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "enforce_eager", False):
        return 0
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    has_full = getattr(mode, "has_full_cudagraphs", None)
    if callable(has_full) and not has_full():
        return 0
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    candidates = [
        int(getattr(compilation_config, "max_cudagraph_capture_size", 0) or 0),
        int(getattr(scheduler_config, "max_num_seqs", 0) or 0)
        * _sx_qsa_mtp_verify_rows(vllm_config),
    ]
    candidates += [
        int(size)
        for size in (getattr(compilation_config, "cudagraph_capture_sizes", None) or ())
    ]
    rows = max(candidates)
    max_tokens = int(getattr(scheduler_config, "max_num_batched_tokens", 0) or 0)
    if max_tokens > 0:
        rows = min(rows, max_tokens)
    return max(0, rows)


def _sx_qsa_exact_prefill_seq_lens(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    verify_rows: int,
) -> torch.Tensor:
    """Host sequence lengths with every possibly optimistic entry set to -1.

    With speculative decoding the scheduler advances num_computed_tokens as if
    every draft of the previous step had been accepted, so
    seq_lens_cpu_upper_bound (num_computed + num_scheduled) is only an upper
    bound for a request that carried drafts. Such a request schedules at most
    1 + num_speculative_tokens query rows (its verify rows). A request with
    more rows is a prompt or recompute chunk whose previous step, if any, was
    a chunk without drafts, so its host length is exact; the QSA ops only
    read host lengths of requests with >= 512 rows. -1 marks "unknown": the
    width planners in ops/qsa.py reject seq_len < rows and keep the device
    path for that request. Host arrays only (a few microseconds per step);
    never touches the device and never modifies the runner's buffers.
    """

    starts = query_start_loc_cpu.numpy()
    exact = seq_lens_cpu.numpy().copy()
    exact[(starts[1:] - starts[:-1]) <= verify_rows] = -1
    return torch.from_numpy(exact)


def _sx_qsa_host_metadata(
    metadata: object,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return (
        getattr(metadata, _SX_QSA_QUERY_START_LOC_CPU, None),
        getattr(metadata, _SX_QSA_SEQ_LENS_CPU, None),
    )


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if _SX_OPT_QSA_HOST_METADATA and self._sx_host_metadata_allowed():
            query_start_loc_cpu = getattr(
                common_attn_metadata, "query_start_loc_cpu", None
            )
            seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
            num_reqs = int(getattr(common_attn_metadata, "num_reqs", -1))
            if (
                isinstance(query_start_loc_cpu, torch.Tensor)
                and isinstance(seq_lens_cpu, torch.Tensor)
                and query_start_loc_cpu.device.type == "cpu"
                and seq_lens_cpu.device.type == "cpu"
                and num_reqs >= 1
                and query_start_loc_cpu.shape[0] >= num_reqs + 1
                and seq_lens_cpu.shape[0] >= num_reqs
            ):
                # Host tensors built for this step only; never read on device.
                query_start_loc_cpu = query_start_loc_cpu[: num_reqs + 1]
                seq_lens_cpu = seq_lens_cpu[:num_reqs]
                verify_rows = getattr(self, "_sx_host_metadata_verify_rows", 0)
                if verify_rows:
                    # MTP lane: keep only the exact (prefill) lengths.
                    seq_lens_cpu = _sx_qsa_exact_prefill_seq_lens(
                        query_start_loc_cpu, seq_lens_cpu, verify_rows
                    )
                setattr(metadata, _SX_QSA_QUERY_START_LOC_CPU, query_start_loc_cpu)
                setattr(metadata, _SX_QSA_SEQ_LENS_CPU, seq_lens_cpu)
        return metadata

    def _sx_host_metadata_allowed(self) -> bool:
        allowed = getattr(self, "_sx_host_metadata_allowed_cache", None)
        if allowed is None:
            vllm_config = getattr(self, "vllm_config", None)
            spec = getattr(vllm_config, "speculative_config", None)
            verify_rows = 0
            if vllm_config is None:
                allowed = False
            elif spec is None:
                allowed = True
            else:
                allowed = bool(
                    _SX_OPT_QSA_MTP_HOST_METADATA
                    and _sx_qsa_mtp_lane_contract(vllm_config)
                )
                if allowed:
                    verify_rows = _sx_qsa_mtp_verify_rows(vllm_config)
            self._sx_host_metadata_verify_rows = verify_rows
            self._sx_host_metadata_allowed_cache = allowed
        return allowed


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # QSA chooses its split-K reduction depth from the runtime batch
        # shape, so it cannot inherit FlashAttention's stronger guarantee.
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_gate: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        query_start_loc_cpu: torch.Tensor | None = None,
        sx_mtp_lane=None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        # This tree's FlashAttention cache ABI keeps K/V on dimension 1:
        # [num_blocks, 2, block_size, num_kv_heads, head_size].
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen4Exp QSA requires FP16/BF16 queries")
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            if key_cache.dtype != torch.uint8 or value_cache.dtype != torch.uint8:
                raise RuntimeError("Qwen4Exp QSA E4M3 cache must use uint8 storage")
        elif key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise RuntimeError("Qwen4Exp QSA FP16/BF16 cache must match query dtype")

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_metadata: dict[str, torch.Tensor] = {}
        if query_positions is not None:
            qsa_metadata["query_positions"] = query_positions[:num_tokens]
        if sequence_lengths is not None:
            qsa_metadata["sequence_lengths"] = sequence_lengths
        if output_gate is not None:
            qsa_metadata["output_gate"] = output_gate[:num_tokens]
        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            query_start_loc_cpu=query_start_loc_cpu,
            sx_mtp_lane=sx_mtp_lane,
            **qsa_metadata,
        )
        return output


def _verify_e4m3_kv_requirements(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> None:
    """Gate for the QSA E4M3 main KV cache.

    E4M3 is qualified on SM70 with FP16 activations and TP4 only. Speculative
    decoding also quantizes the drafter's K/V and needs its own calibrated
    scales, so it stays rejected unless envs.VLLM_QWEN4EXP_QSA_E4M3_MTP opts
    in.
    """
    if cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
        return
    if not current_platform.is_device_capability(70):
        raise NotImplementedError("Qwen4Exp QSA E4M3 phase 1 requires SM70")
    if model_config.dtype != torch.float16:
        raise NotImplementedError("Qwen4Exp QSA E4M3 phase 1 requires FP16 activations")
    if vllm_config.parallel_config.tensor_parallel_size != 4:
        raise NotImplementedError("Qwen4Exp QSA E4M3 phase 1 requires TP4")
    if (
        vllm_config.speculative_config is not None
        and not envs.VLLM_QWEN4EXP_QSA_E4M3_MTP
    ):
        raise NotImplementedError("Qwen4Exp QSA E4M3 phase 1 requires MTP0")


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False
    # The paged indexer and sparse attention switch launch profiles after 32
    # query rows. Advertise the first row count in the wider profile so the
    # generic MRV2 warmup can compile it before serving traffic.
    kernel_warmup_prefill_token_counts = (33,)

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("Qwen4Exp QSA requires FP16 or BF16")
        if cache_config.cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache"
            )
        _verify_e4m3_kv_requirements(vllm_config, model_config, cache_config)
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True
        qkv_quant_config = quant_config
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            qkv_quant_config = None

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=qkv_quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3") and (
            cache_config.calculate_kv_scales
        ):
            raise ValueError(
                "QSA calibrated E4M3 forbids calculate_kv_scales; "
                "load an offline scale overlay instead"
            )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_dtype not in ("fp8", "fp8_e4m3") and (
            self.kv_cache_torch_dtype != model_config.dtype
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA main cache dtype must match the model dtype"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)
        self._qsa_kv_scales_finalized = self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
        )
        if not self._qsa_kv_scales_finalized:
            # Keep checkpoint loading state separate from runtime scales.
            # Negative is deliberately invalid and the slots are deleted once
            # validation copies them into the runtime buffers.
            self.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
            self.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._set_topk_indices_buffer(
            max_tokens=max_tokens,
            topk_indices_buffer=topk_indices_buffer,
        )
        # Batch 3a (SX_OPT_QSA_MTP_*): MTP-lane QSA options, None elsewhere
        # (the no-MTP lane keeps the 1.8.0-dev2 gates exactly).
        self._sx_qsa_mtp_lane = None
        if _sx_qsa_mtp_lane_contract(vllm_config):
            from .ops.qsa import sx_qsa_mtp_lane_options

            self._sx_qsa_mtp_lane = sx_qsa_mtp_lane_options(
                _sx_qsa_mtp_page4_graph_rows(vllm_config)
            )
            if self._sx_qsa_mtp_lane.two_warp_max_rows > 32 or (
                self._sx_qsa_mtp_lane.topk_max_rows > 32
            ):
                # Rows 33..63 now run the decode-row kernels; also warm the
                # 16-divisible Triton specialisation (48 rows).
                self.kernel_warmup_prefill_token_counts = (33, 48)
            logger.info_once(
                "Qwen4Exp QSA MTP lane: decode-row caps two_warp=%d "
                "resolved=%d topk=%d, page4 graph rows=%d, host metadata=%s "
                "(SX_OPT_QSA_MTP_*).",
                self._sx_qsa_mtp_lane.two_warp_max_rows,
                self._sx_qsa_mtp_lane.resolved_max_rows,
                self._sx_qsa_mtp_lane.topk_max_rows,
                self._sx_qsa_mtp_lane.page4_graph_rows,
                _SX_OPT_QSA_HOST_METADATA and _SX_OPT_QSA_MTP_HOST_METADATA,
            )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def _set_topk_indices_buffer(
        self,
        *,
        max_tokens: int,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        if topk_indices_buffer is None:
            self.register_buffer(
                "topk_indices_buffer",
                torch.empty(
                    max_tokens,
                    self.indexer.output_width,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
            return

        expected_width = self.indexer.output_width
        if (
            topk_indices_buffer.dtype != torch.int32
            or topk_indices_buffer.ndim != 2
            or topk_indices_buffer.shape[0] < max_tokens
            or topk_indices_buffer.shape[1] != expected_width
        ):
            raise ValueError(
                "QSA shared top-k buffer must have dtype int32 and shape "
                f"[{max_tokens} or more, {expected_width}], got "
                f"dtype={topk_indices_buffer.dtype}, "
                f"shape={tuple(topk_indices_buffer.shape)}"
            )
        self.topk_indices_buffer = topk_indices_buffer

    def adopt_default_kv_scales(self) -> None:
        """Use the module's own unit scales when the checkpoint has none.

        The calibrated overlay exists to keep FP8 E4M3 K/V inside range; a
        checkpoint that was never calibrated has no such overlay, so the layer
        keeps the 1.0 defaults set at construction instead of the -1.0 loading
        sentinel and is marked finalized so nothing re-validates it.
        """
        set_default_quant_scales(self, register_buffer=False)
        if hasattr(self, "k_scale"):
            del self.k_scale
        if hasattr(self, "v_scale"):
            del self.v_scale
        self._qsa_kv_scales_finalized = True

    def validate_loaded_kv_scales(self) -> None:
        if self.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
            return
        if self._qsa_kv_scales_finalized:
            raise RuntimeError(
                f"QSA E4M3 scales already finalized for {self.layer_name}"
            )
        if not hasattr(self, "k_scale") or not hasattr(self, "v_scale"):
            raise RuntimeError(
                f"QSA E4M3 loading slots are unavailable for {self.layer_name}"
            )

        scales = {
            "K": float(self.k_scale.item()),
            "V": float(self.v_scale.item()),
        }
        invalid = [
            name
            for name, value in scales.items()
            if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(
                f"QSA E4M3 calibrated scales are required for {self.layer_name} "
                f"(invalid: {', '.join(invalid)}). Refusing to start."
            )
        self._k_scale.copy_(scales["K"])
        self._v_scale.copy_(scales["V"])
        self._k_scale_float = scales["K"]
        self._v_scale_float = scales["V"]
        del self.k_scale
        del self.v_scale
        self._qsa_kv_scales_finalized = True

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split, normalize, and rotate Q/K using this tree's Qwen3 API."""
        q_gate, key, value = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        token_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*token_shape, self.num_heads, 2 * self.head_dim)
        query, gate = torch.chunk(q_gate, 2, dim=-1)
        query = self.q_norm(query).reshape(*token_shape, self.q_size)
        key = self.k_norm(
            key.view(*token_shape, self.num_kv_heads, self.head_dim)
        ).reshape(*token_shape, self.kv_size)
        query, key = self.rotary_emb(positions, query, key)
        return query, key, value, gate.reshape(*token_shape, self.q_size)

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ) -> None:
        if not self._qsa_kv_scales_finalized:
            raise RuntimeError(
                f"QSA E4M3 scales were not finalized for {self.layer_name}"
            )
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        if os.getenv("VLLM_QSA_KV_CALIBRATION_DIR"):
            from .ops.qsa_kv_calibration import observe_qsa_kv

            observe_qsa_kv(
                self.indexer.layer_id,
                key[:num_tokens],
                value[:num_tokens],
            )
        # Optional host copies attached by Qwen4ExpQSAMetadataBuilder. None
        # (spec decode outside the MTP lane, other metadata builders, switch
        # off) keeps the baseline paths; the ops ignore them while a CUDA
        # graph is captured.
        query_start_loc_cpu, seq_lens_cpu = _sx_qsa_host_metadata(main_metadata)
        sx_mtp_lane = getattr(self, "_sx_qsa_mtp_lane", None)
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            sx_mtp_lane=sx_mtp_lane,
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        selected = _sm70_dump_qwen_layer_tensor(
            "qsa_selected_indices",
            self.indexer.layer_id,
            "qsa",
            selected,
        )
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            output_gate=output_gate,
            query_positions=side_metadata.logical_positions,
            sequence_lengths=side_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            sx_mtp_lane=sx_mtp_lane,
        )
        _sm70_dump_qwen_layer_tensor(
            "qsa_core_out",
            self.indexer.layer_id,
            "qsa",
            output[:num_tokens],
        )

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                gate,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        projected_output, _ = self.o_proj(flat_output)
        if output is not None:
            output.copy_(projected_output)
        return projected_output


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
        output_gate,
    )


def qwen4_exp_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gate: torch.Tensor | None,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, output_gate, layer_name


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_with_output",
]
