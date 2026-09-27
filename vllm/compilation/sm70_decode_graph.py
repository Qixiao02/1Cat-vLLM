# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-phase selection for the SM70 Qwen3.8 decode graph.

The Qwen3.8 V100 serving lane uses two compiled backbones that share the same
module parameters: a dynamic prefill graph and a small-shape decode graph.
This context is active only while tracing/capturing the latter.

SX batch 3a (lane-core, design_1 MTP-1): the SM70 Qwen3.8 native-MTP lane
also runs the target model in the dual-compile lane, so its FULL uniform
verify graphs (every request has k+1 query rows) are traced by the decode
compiler and take the M=1 / exact multi-row routes. The MTP drafter is a
separate single compile that is traced outside any target FULL capture; before
this lane existed it always had the legacy single-compile semantics
(``use_sm70_decode_graph_semantics() == True``: sum2 all-reduce, native Gemma
RMSNorm, ...). To keep the drafter exactly as it was, the target marks its
*main* (prefill/mixed) backbone forward with :func:`sm70_target_main_backbone`
once the lane is installed (:func:`set_sm70_mtp_lane_installed`, called by the
Qwen3.8 target at construction); everything else traced in that process keeps
decode semantics. Nothing changes unless the lane is installed, so the no-MTP
lane evaluates exactly as before.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

import vllm.envs as envs

_sm70_decode_graph_compilation = ContextVar(
    "sm70_decode_graph_compilation", default=False
)
# SX batch 3a: set around the target's main (non-decode) backbone forward in
# the native-MTP lane.
_sm70_target_main_backbone = ContextVar("sm70_target_main_backbone", default=False)
# SX batch 3a: this process built a Qwen3.8 target under the native-MTP lane
# contract with the dual-compile lane enabled.
_SM70_MTP_LANE_INSTALLED = False


@contextmanager
def sm70_decode_graph_compilation(enabled: bool = True) -> Generator[None, None, None]:
    token = _sm70_decode_graph_compilation.set(enabled)
    try:
        yield
    finally:
        _sm70_decode_graph_compilation.reset(token)


@contextmanager
def sm70_target_main_backbone(enabled: bool = True) -> Generator[None, None, None]:
    """Mark the target's main-compile forward (native-MTP lane only)."""
    token = _sm70_target_main_backbone.set(enabled)
    try:
        yield
    finally:
        _sm70_target_main_backbone.reset(token)


def set_sm70_mtp_lane_installed(installed: bool) -> None:
    """Record that the Qwen3.8 native-MTP dual-compile lane is installed."""
    global _SM70_MTP_LANE_INSTALLED
    _SM70_MTP_LANE_INSTALLED = bool(installed)


def sm70_mtp_lane_installed() -> bool:
    return _SM70_MTP_LANE_INSTALLED


@torch.compiler.assume_constant_result
def is_sm70_decode_graph_compiling() -> bool:
    """Return a trace-time constant for the selected SM70 compilation phase."""
    return _sm70_decode_graph_compilation.get()


@torch.compiler.assume_constant_result
def is_sm70_mtp_drafter_decode_semantics() -> bool:
    """Trace-time constant: native-MTP lane code outside the target main
    backbone (the MTP drafter) keeps the single-compile decode semantics."""
    return _SM70_MTP_LANE_INSTALLED and not _sm70_target_main_backbone.get()


def use_sm70_decode_graph_semantics() -> bool:
    """Preserve legacy behavior unless the dual-compile lane is active."""
    return (
        not envs.VLLM_SM70_QWEN38_DUAL_COMPILE
        or is_sm70_decode_graph_compiling()
        or is_sm70_mtp_drafter_decode_semantics()
    )


__all__ = [
    "is_sm70_decode_graph_compiling",
    "is_sm70_mtp_drafter_decode_semantics",
    "set_sm70_mtp_lane_installed",
    "sm70_decode_graph_compilation",
    "sm70_mtp_lane_installed",
    "sm70_target_main_backbone",
    "use_sm70_decode_graph_semantics",
]
