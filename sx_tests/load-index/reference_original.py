# SPDX-License-Identifier: Apache-2.0
"""Verbatim originals of the functions the "load-index" group changed.

These are the oracles of the equivalence tests: each body is a character-for-
character copy (re-indented to column 0) of the code as it was at commit
4c999703b, before its SX_OPT_* switch was added. Do not edit them to match a
change in the tree; a diverging behaviour is exactly what the tests look for.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

logger = type("_L", (), {"debug": staticmethod(lambda *a, **k: None)})()


# FusedMoE.load_weights (vllm/model_executor/layers/fused_moe/layer.py)
def load_weights_original(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[str]:
    if (expert_mapping := self.expert_mapping) is None:
        raise ValueError(
            "`self.expert_mapping` must be provided to "
            "load weights using `self.load_weights`."
        )
    for expert_name, loaded_weight in weights:
        qual_name = f"{self.layer_name}.{expert_name}"
        for param_name, weight_name, expert_id, shard_id in expert_mapping:
            if weight_name not in qual_name:
                continue
            weight_name = qual_name.replace(weight_name, param_name)
            param_name = weight_name.removeprefix(f"{self.layer_name}.")
            param = getattr(self, param_name)
            # Fused expert weights can be identified by their 3D tensors
            if loaded_weight.dim() == 3:
                # Repurpose expert_id as shard_idx for deconcatenating w1 and w3
                if shard_id in {"w1", "w3"}:
                    shard_idx = expert_id
                    experts_shard = loaded_weight.chunk(2, dim=1)[shard_idx]
                else:
                    experts_shard = loaded_weight
                start = 0
            else:
                # loaded_weight is a single expert weight, so we add a dummy expert
                # dimension to unify the loading logic with the fused case
                experts_shard = loaded_weight.unsqueeze(0)
                start = expert_id

            # Unified loading logic for fused and non-fused experts
            loaded_experts = experts_shard.unbind()
            for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                success = self.weight_loader(
                    param=param,
                    loaded_weight=loaded_expert,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded expert %d of shard %s into %s for layer %s",
                        expert_id,
                        shard_id,
                        param_name,
                        self.layer_name,
                    )
                    yield param_name


# _remap_qsa_cache_scale_name (vllm/models/qwen4_exp/nvidia/model.py)
def remap_qsa_cache_scale_name_original(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's invalid-until-loaded ``k_scale``/``v_scale`` slots.
    """

    scale_suffixes = {
        "k_proj.k_scale": "k_scale",
        "k_proj.output_scale": "k_scale",
        "attn.k_scale": "k_scale",
        "attn._k_scale": "k_scale",
        "k_scale": "k_scale",
        "_k_scale": "k_scale",
        "v_proj.v_scale": "v_scale",
        "v_proj.output_scale": "v_scale",
        "attn.v_scale": "v_scale",
        "attn._v_scale": "v_scale",
        "v_scale": "v_scale",
        "_v_scale": "v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


# AutoWeightsLoader._can_skip (vllm/model_executor/models/utils.py)
def can_skip_original(self, qualname: str) -> bool:
    return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
        substr in qualname for substr in self.skip_substrs
    )
