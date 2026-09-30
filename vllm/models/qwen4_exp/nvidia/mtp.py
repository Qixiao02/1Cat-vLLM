# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).

SX batch-3 "draft" group (design_4 MTP-K1, design_1 MTP-7 / MTP-8)
-------------------------------------------------------------------
Only under the exact SM70 TP4 FP16 Qwen3.8 Flash-Next MTP contract
(``_sx_mtp_draft_contract``: speculative method "mtp", TP4, PP1, no EP, the
same topology check as the no-MTP fast paths) the drafter

* arms the exact-shape draft MoE tile table in fused_moe.py
  (``SX_OPT_MTP_DRAFT_TILES``, default on; ``0`` keeps 1Cat's M1/M5 tile and
  the 0.0.3 tile elsewhere);
* installs the existing checkpoint-FP16 SM70 routes on its own layer
  (``SX_OPT_MTP_DRAFT_GEMV``, default on; ``0`` = previous behaviour):
  the M=1 row GEMV (sm70_fp16_gemv.Qwen38SM70FP16LinearMethod) on
  qkv / o / indexer / router / HC down and the fused FP16 HC route on the
  attn and mlp HC modules.  Both engage where the drafter has decode-graph
  semantics (single-compile lane, or the batch-3a native-MTP lane, which
  keeps the drafter on them), at M = 1: draft decode of one request with
  split draft graphs (VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=1, set by the
  MTP lane; otherwise one request pads to k+1 rows).  M > 1 stays on
  F.linear inside the opaque ops, because the rows kernels additionally
  require the target's FULL decode-graph capture.  The shared-expert gate is
  left to qwen2_moe (SX_OPT_MTP_MOE_ROUTES admits its rows for the drafter).
  Not installed with online QPN8 or VLLM_BATCH_INVARIANT; an explicit
  VLLM_SM70_QWEN38_FP16_GEMV=0 / VLLM_SM70_QWEN38_FUSED_HC_FP16=0 also keeps
  the drafter off that kernel family.

All of this changes draft proposals only.  The target model and the
rejection sampler are untouched, so the output distribution is unchanged.

FP8 draft experts: Swift 1.5 uses the routed-experts-only NVFP4 recipe with
``mtp.*`` in exclude_modules, so its MTP experts are unquantized and the
online FP8 route (mtp_fp8_experts.MTPFp8SM70MoEMethod) applies unchanged.
Enable it with ``"mtp_expert_quantization": "fp8"`` in --speculative-config,
e.g. ``{"method": "mtp", "num_speculative_tokens": 4,
"mtp_expert_quantization": "fp8"}``.  It keeps 960 MiB packed FP8 + 15 MiB
scales instead of 1200 MiB FP16 per rank (1Cat measured -892 MiB/rank whole
GPU with an NVFP4 target) and is a draft-only precision change.  The draft
MoE then runs Fp8SM70MoEMethod, so the Triton tile table does not apply.
"""

import os
from collections.abc import Iterable
from types import SimpleNamespace

import regex as re
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from .hyperconnection import GatedResidual, HyperConnectionConfig

logger = init_logger(__name__)

try:
    from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
except ModuleNotFoundError as exc:
    # The Blackwell-only CuTe DSL helper is absent from the 1Cat SM70 tree.
    if exc.name != "vllm.model_executor.kernels.linear.cute_dsl":
        raise

    def enable_qwen4_exp_low_latency_gemm(
        module: nn.Module, dtype: torch.dtype
    ) -> None:
        del module, dtype


from .model import (
    _HC_WEIGHTS_MAPPER,
    _QWEN3_5_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices."""
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _validate_mtp_expert_weights_loaded(
    model: nn.Module,
    loaded_weights: set[str],
) -> None:
    """Reject a silently incomplete Qwen4Exp MTP routed-expert load."""

    required_suffixes = (
        ".mlp.experts.w13_weight",
        ".mlp.experts.w2_weight",
    )
    required = {
        name for name, _ in model.named_parameters() if name.endswith(required_suffixes)
    }
    missing = required - loaded_weights
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            "Qwen4Exp MTP routed-expert checkpoint weights were not loaded: "
            f"{missing_names}. Check fused/per-expert checkpoint mappings."
        )


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    from vllm.model_executor.layers.quantization.sm70_turbomind import (
        is_exact_sm70_cuda_platform,
    )

    from .mtp_fp8_experts import MTPExpertFp8Config, checkpoint_fp8_prefixes

    online_fp8 = getattr(speculative_config, "mtp_expert_quantization", None) == "fp8"
    checkpoint_prefixes = set()
    if draft_quant_config is not None and is_exact_sm70_cuda_platform():
        config = draft_vllm_config.model_config.hf_text_config
        prefixes = {
            f"mtp.layers.{mtp_start_layer_idx + index}.mlp.experts"
            for index in range(getattr(config, "mtp_num_hidden_layers", 1))
        }
        checkpoint_prefixes = checkpoint_fp8_prefixes(draft_quant_config, prefixes)
    if online_fp8 or checkpoint_prefixes:
        if (
            not is_exact_sm70_cuda_platform()
            or draft_vllm_config.model_config.dtype != torch.float16
            or draft_quant_config is None
            or draft_quant_config.get_name()
            not in ("awq", "modelopt_fp4", "modelopt_mixed", "fp8")
            or draft_vllm_config.parallel_config.pipeline_parallel_size != 1
            or draft_vllm_config.parallel_config.enable_expert_parallel
            or speculative_config.rejection_sample_method != "standard"
        ):
            raise ValueError(
                "MTP FP8 experts require SM70, FP16, an AWQ/ModelOpt/FP8 draft "
                "checkpoint, tensor parallelism without PP or EP, and standard "
                "rejection sampling"
            )
        draft_quant_config = MTPExpertFp8Config(
            draft_quant_config,
            checkpoint_prefixes,
            quantize_unquantized=online_fp8,
        )
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


def _sx_mtp_draft_contract(vllm_config) -> bool:
    """Exact SM70 TP4 FP16 Qwen3.8 Flash-Next drafter (speculative "mtp").

    ``vllm_config`` is the config Qwen4ExpMTP is constructed with (target
    model config plus the speculative config).  The topology part reuses the
    no-MTP fast-path contract of sm70_fp16_gemv; that helper also rejects any
    speculative config because it guards the target lane, so it is evaluated
    on a view without one.
    """
    try:
        spec = vllm_config.speculative_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        if spec is None or getattr(spec, "method", None) != "mtp":
            return False
        if model_config.dtype != torch.float16:
            return False
        if (
            int(parallel_config.tensor_parallel_size) != 4
            or int(parallel_config.pipeline_parallel_size) != 1
            or bool(getattr(parallel_config, "enable_expert_parallel", False))
        ):
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability((7, 0))
    ):
        return False

    from .sm70_fp16_gemv import _exact_runtime_contract

    view = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=None,
    )
    return _exact_runtime_contract(view)


def _sx_install_mtp_draft_fp16_routes(
    model: nn.Module, *, gemv: bool = True, hc: bool = True
) -> dict[str, int]:
    """MTP-8: put the drafter's own projections on the existing SM70 routes.

    Same selection rules as enable_qwen38_sm70_fp16_gemv /
    enable_qwen38_sm70_fp16_fused_hc (role suffix + exact shape, unquantized
    FP16 weights, base HC modules with a combine), applied independently of
    the no-MTP env defaults and without touching the target model.  The
    module-global fused-HC counter (MR9a combine-norm tile of the *target*
    decode graph) is deliberately left alone.  ``gemv`` / ``hc`` select the
    two route families (the caller maps the global kill switches onto them).
    """
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )

    from .sm70_fp16_gemv import _ROLE_PLANS, Qwen38SM70FP16LinearMethod, _plan_for
    from .sm70_fp16_hc import _HC_COUNT, _HC_DIM, _HC_HIDDEN, _HC_RANK

    counts = {"gemv": 0, "hc": 0}
    roles: list[str] = []
    for child in model.modules() if gemv else ():
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.dtype != torch.float16
        ):
            continue
        prefix = str(getattr(child, "prefix", ""))
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        if _plan_for(prefix, shape) is None:
            continue
        child.quant_method = Qwen38SM70FP16LinearMethod()
        counts["gemv"] += 1
        roles.extend(
            suffix.rsplit(".", 1)[-1]
            for suffix, _, _ in _ROLE_PLANS
            if prefix.endswith(suffix)
        )

    for child in model.modules() if hc else ():
        down = getattr(child, "input_mix_weight_down_block_inject", None)
        up = getattr(child, "input_mix_weight_up", None)
        down_weight = getattr(down, "weight", None)
        up_weight = getattr(up, "weight", None)
        if not (
            getattr(child, "use_combine", False)
            and getattr(child, "lora_rank", None) == _HC_RANK
            and getattr(child, "hc_count", None) == _HC_COUNT
            and getattr(child, "hidden_size", None) == _HC_DIM
            and isinstance(down_weight, torch.Tensor)
            and isinstance(up_weight, torch.Tensor)
            and tuple(down_weight.shape) == (_HC_RANK + _HC_COUNT + 12, _HC_HIDDEN)
            and tuple(up_weight.shape) == (_HC_HIDDEN, _HC_RANK)
            and down_weight.dtype == torch.float16
            and up_weight.dtype == torch.float16
            # The HC down projection itself has a GEMV plan (installed above,
            # as enable_qwen38_sm70_fp16_gemv does for the target); the fused
            # route only reads the FP16 weights.
            and isinstance(getattr(down, "quant_method", None), UnquantizedLinearMethod)
            and isinstance(getattr(up, "quant_method", None), UnquantizedLinearMethod)
        ):
            continue
        child._sm70_qwen38_fp16_fused_hc = True
        counts["hc"] += 1

    # The shared-expert gate is not touched here: its exact M == 1 gate is
    # already active in the drafter, and the multi-row gate for draft rows is
    # admitted by qwen2_moe under the MTP-lane switch (SX_OPT_MTP_MOE_ROUTES).
    logger.info(
        "SX MTP draft SM70 FP16 routes (SX_OPT_MTP_DRAFT_GEMV): %d M=1 GEMV "
        "projections %s, %d fused HC modules.",
        counts["gemv"],
        sorted(set(roles)),
        counts["hc"],
    )
    if not counts["gemv"] and not counts["hc"]:
        logger.warning(
            "SX_OPT_MTP_DRAFT_GEMV matched the runtime but found no draft "
            "projection with an exact SM70 FP16 plan."
        )
    return counts


def _sx_prepare_mtp_draft_sm70(model: nn.Module, vllm_config) -> dict[str, int]:
    """Arm the draft MoE tile table and install the draft FP16 routes."""
    if not _sx_mtp_draft_contract(vllm_config):
        return {}
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        arm_sm70_mtp_draft_moe_tiles,
    )

    arm_sm70_mtp_draft_moe_tiles(True)
    if os.environ.get("SX_OPT_MTP_DRAFT_GEMV", "1").strip() == "0":
        return {}
    if envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8:
        # Online QPN8 repacks the unquantized projections at load time.
        logger.info_once(
            "SX_OPT_MTP_DRAFT_GEMV skipped: VLLM_SM70_QWEN4_EXP_ONLINE_QPN8 is set."
        )
        return {}
    if envs.VLLM_BATCH_INVARIANT:
        # UnquantizedLinearMethod keeps linear_batch_invariant for the draft;
        # the opaque GEMV / fused-HC ops would bypass it (M=1 kernels, plain
        # F.linear otherwise), so a draft row would depend on the batch width.
        logger.info_once("SX_OPT_MTP_DRAFT_GEMV skipped: VLLM_BATCH_INVARIANT is set.")
        return {}
    # An explicit "0" of the global kernel-family switches (the target lane's
    # rollback knobs) also keeps the drafter off that family; unset means the
    # draft default (on), since the no-MTP lane defaults them only when absent.
    gemv = _sx_env_not_forced_off("VLLM_SM70_QWEN38_FP16_GEMV")
    hc = _sx_env_not_forced_off("VLLM_SM70_QWEN38_FUSED_HC_FP16")
    if not gemv and not hc:
        logger.info_once(
            "SX_OPT_MTP_DRAFT_GEMV skipped: VLLM_SM70_QWEN38_FP16_GEMV=0 and "
            "VLLM_SM70_QWEN38_FUSED_HC_FP16=0."
        )
        return {}
    return _sx_install_mtp_draft_fp16_routes(model, gemv=gemv, hc=hc)


def _sx_env_not_forced_off(name: str) -> bool:
    return os.environ.get(name, "").strip() != "0"


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = _QWEN3_5_WEIGHTS_MAPPER | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        self.fp8_mtp_checkpoint_prefixes = {
            f"model.layers.{int(name.split('.')[2]) - self.mtp_start_layer_idx}"
            ".mlp.experts"
            for name in getattr(
                draft_vllm_config.quant_config, "checkpoint_prefixes", ()
            )
        }
        self.fp8_mtp_tp_size = draft_vllm_config.parallel_config.tensor_parallel_size
        # The drafter shares the target cache_config; load_weights needs the
        # main KV cache dtype to finalize the draft QSA E4M3 scales.
        self._kv_cache_dtype = draft_vllm_config.cache_config.cache_dtype
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=model_config.dtype,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size

        # The drafter is stage-local, so this module is always a complete
        # model: gpu_model_runner returns the IntermediateTensors on every
        # non-final pipeline rank before speculation is reached, and the
        # weights here are replicated rather than partitioned (embed_tokens,
        # fc_embedding, fc_hidden and every MTP layer are built on all ranks).
        # Branching on the TARGET model's pipeline position sent the final
        # rank into the "receive from the previous stage" path and asserted on
        # intermediate tensors that nobody sends -- with the fullgraph AOT
        # compile of 1.5.0 that is a hard compile error, so no k > 0 boots at
        # all under pipeline parallelism.
        assert hidden_states is not None
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        # Embedding branch: pre-norm -> fc_embedding -> [T, H].
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        inputs_embeds = self.fc_embedding(inputs_embeds)

        # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
        # the main model truly emits the pre-final-mixer multi stream
        # on the first step; subsequent steps reuse the prior draft
        # step's multi stream).
        num_tokens = hidden_states.shape[0]
        hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
        hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
            num_tokens, hc_count, hidden_size
        )
        hidden_states = self.fc_hidden(hidden_states)
        # Add the embedding residual to every branch, then fold back
        # to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.
        hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
        hidden_states = hidden_states.flatten(-2)

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):
    # Qwen4Exp repacks the small BF16/shared/MTP tensors separately from the
    # target experts and PLE tables. Loading the standalone drafter from that
    # compact shard set avoids scanning the full target checkpoint a second
    # time. The remaining patterns preserve compatibility with conventional
    # Hugging Face checkpoint layouts.
    allow_patterns_overrides = [
        "model-bf16-*.safetensors",
        "*.safetensors",
        "*.bin",
        "*.pt",
    ]

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4ExpMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)
        # SX batch 3 (draft): MTP-K1 tile table + MTP-8 draft FP16 routes.
        _sx_prepare_mtp_draft_sm70(self, vllm_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        from .mtp_fp8_checkpoint import prepare_mtp_fp8_checkpoint

        loaded_weights = loader.load_weights(
            prepare_mtp_fp8_checkpoint(
                remap_weight_names(),
                self.model.fp8_mtp_checkpoint_prefixes,
                tp_size=self.model.fp8_mtp_tp_size,
                num_experts=self.model.config.num_experts,
            )
        )
        _validate_mtp_expert_weights_loaded(self, loaded_weights)
        # An E4M3 main KV cache makes the drafter's QSA layer E4M3 too. Its
        # k/v scales must come calibrated from the checkpoint: unit scales
        # gave invalid proposals, so missing ones fail the load. No-op for
        # FP16/BF16 caches and on the PLE offload process.
        from .model import _finalize_qsa_e4m3_scale_load

        _finalize_qsa_e4m3_scale_load(
            self,
            loaded_weights,
            self.model._kv_cache_dtype,
            require_calibrated_speculative_draft=True,
        )
        return loaded_weights


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
