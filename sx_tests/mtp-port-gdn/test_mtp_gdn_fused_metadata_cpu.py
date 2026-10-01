# SPDX-License-Identifier: Apache-2.0
"""Fused native-MTP GDN metadata (upstream 98b81ea69 + cfe8490a8) on CPU.

Run (CPU, no vLLM install needed; see ``_port_boot.py``):

    python -m pytest -q sx_tests/mtp-port-gdn/test_mtp_gdn_fused_metadata_cpu.py

What runs is the fork's real code, cut out of gdn_attn.py / attn_utils.py /
backends/utils.py / mamba_hybrid.py:

* ``_dflash2_gdn_group_metadata_kernel`` and ``_load_gdn_i32_ptr`` execute
  their real bodies against a minimal fake ``tl`` (pointer = flat tensor plus
  offsets; masked load/store; pointer tables resolve registered data_ptrs).
  Triton's truncating division only differs from torch's floor division on
  masked-off lanes, which the kernel clamps to column 0 either way.
* ``prepare_dflash2_gdn_group_metadata`` runs unchanged except for its
  ``device.type != "cuda"`` guard, relaxed to admit CPU tensors; the shadow
  oracle (VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW) is on in every replay case.
* ``sx_mtp_pure_common_gdn_metadata`` is compared with upstream's
  ``compute_common_gdn_attn_metadata``; the per-group reference state rows
  come from ``mamba_get_block_table_tensor`` with the builder's FULL-graph
  padding (PAD rows, masks, query offsets, accepted counts and selectors).

Asserted:
* pure verify batches (k = 1..4, B = 1..8 padded to the graph size, align
  and none cache modes, 784- and 16-token state blocks, three cache groups,
  sequence lengths across block boundaries, changing values over successive
  steps with a reused descriptor) produce exactly the per-group metadata;
* mixed / decode / non-prefix / zero-draft / oversized batches, either kill
  switch, a builder without shared buffers and mode/argument mismatches keep
  the per-group builds (None);
* admission: lane contract on SM70, CUDA device, none/align cache mode,
  SX_OPT_MTP_GDN_FUSED_META and both upstream switches;
* MambaHybridModelState.prepare_attn: the fused step runs only for FULL
  graphs outside capture, passes the align-mode sequence lengths, and hands
  shared metadata to the builders only when the fused write succeeded.
"""

from __future__ import annotations

import os
import random
import re
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _port_boot as boot  # noqa: E402

PAD_SLOT_ID = int(
    re.search(r"(?m)^PAD_SLOT_ID = (-?\d+)$", boot.read(boot.BACKEND_UTILS))[1]
)


# ---------------------------------------------------------------------------
# Minimal Triton language stand-in for the metadata kernel.
# ---------------------------------------------------------------------------
REGISTRY: dict[int, torch.Tensor] = {}


class Ptr:
    def __init__(self, flat: torch.Tensor, offset: Any = 0):
        self.flat = flat
        self.offset = offset

    def __add__(self, other):
        return Ptr(self.flat, self.offset + other)

    __radd__ = __add__


class FakeTL:
    constexpr = object
    int32 = "int32"
    pid = 0

    @staticmethod
    def pointer_type(dtype):
        return ("ptr", dtype)

    def program_id(self, axis):
        assert axis == 0
        return self.pid

    @staticmethod
    def arange(start, end):
        return torch.arange(start, end)

    @staticmethod
    def load(ptr, mask=None, other=0):
        offset = ptr.offset
        if not torch.is_tensor(offset) or offset.dim() == 0:
            return ptr.flat[int(offset)]
        offset = offset.long()
        if mask is None:
            mask = torch.ones_like(offset, dtype=torch.bool)
        mask = torch.as_tensor(mask).expand(offset.shape)
        out = torch.full(offset.shape, other, dtype=ptr.flat.dtype)
        out[mask] = ptr.flat[offset[mask]]
        return out

    @staticmethod
    def store(ptr, value, mask=None):
        offset = ptr.offset.long()
        value = torch.as_tensor(value).expand(offset.shape)
        if mask is None:
            mask = torch.ones_like(offset, dtype=torch.bool)
        mask = torch.as_tensor(mask).expand(offset.shape)
        ptr.flat[offset[mask]] = value[mask].to(ptr.flat.dtype)

    @staticmethod
    def maximum(a, b):
        return torch.maximum(a, torch.as_tensor(b, dtype=a.dtype))

    @staticmethod
    def minimum(a, b):
        return torch.minimum(a, torch.as_tensor(b, dtype=a.dtype))

    @staticmethod
    def cast(value, pointer_type):
        return Ptr(REGISTRY[int(value)], 0)

    @staticmethod
    def multiple_of(value, n):
        return value


TL = FakeTL()


class FakeKernel:
    def __init__(self, fn):
        self.fn = fn
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.pop("num_warps", None)
            self.launches += 1
            args = tuple(
                Ptr(arg.reshape(-1)) if torch.is_tensor(arg) else arg for arg in args
            )
            for pid in range(grid[0]):
                TL.pid = pid
                self.fn(*args, **kwargs)

        return launch


def register(*tensors):
    for tensor in tensors:
        assert tensor.is_contiguous()
        REGISTRY[tensor.data_ptr()] = tensor.view(-1)


# ---------------------------------------------------------------------------
# The cut-out modules.
# ---------------------------------------------------------------------------
class Envs:
    def __init__(self):
        self.VLLM_SM70_MTP4_SHARED_GDN_METADATA = True
        self.VLLM_SM70_MTP4_FUSED_GDN_METADATA = True
        self.VLLM_SM70_DFLASH2_FUSED_GDN_METADATA = False
        self.VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW = True


class FakeMambaSpec:
    def __init__(self, block_size, num_speculative_blocks):
        self.block_size = block_size
        self.num_speculative_blocks = num_speculative_blocks


def _attn_utils_ns():
    namespace = {"torch": torch, "dataclass": dataclass}
    boot.cut(
        boot.ATTN_UTILS,
        ("CommonGDNSpecMetadata", "compute_common_gdn_attn_metadata"),
        namespace,
    )
    return namespace


ATTN_UTILS = _attn_utils_ns()
CommonGDNSpecMetadata = ATTN_UTILS["CommonGDNSpecMetadata"]
compute_common = ATTN_UTILS["compute_common_gdn_attn_metadata"]

UTILS_NS = boot.cut(
    boot.BACKEND_UTILS,
    ("mamba_get_block_table_tensor",),
    {"torch": torch, "KVCacheSpec": object, "MambaSpec": FakeMambaSpec},
)
mamba_get_block_table_tensor = UTILS_NS["mamba_get_block_table_tensor"]


def _relax_cuda_guard(code: str) -> str:
    old = 'if num_accepted_tokens.device.type != "cuda":'
    assert code.count(old) == 1
    return code.replace(old, 'if num_accepted_tokens.device.type not in ("cuda", "cpu"):')


@pytest.fixture
def gdn(monkeypatch):
    monkeypatch.delenv("SX_OPT_MTP_GDN_FUSED_META", raising=False)
    REGISTRY.clear()
    _GDNDdTreeFastCommonBuffers = None  # noqa: F841 - filled by cut
    envs = Envs()
    namespace = {
        "torch": torch,
        "os": os,
        "dataclass": dataclass,
        "tl": TL,
        "triton": SimpleNamespace(
            next_power_of_2=lambda n: 1 << max(int(n) - 1, 0).bit_length()
        ),
        "envs": envs,
        "PAD_SLOT_ID": PAD_SLOT_ID,
        "CommonGDNSpecMetadata": CommonGDNSpecMetadata,
        "GDNAttentionMetadataBuilder": object,
        "GDN_SPEC_METADATA_TENSORS": tuple,
    }
    boot.cut(
        boot.GDN_ATTN,
        (
            "GDN_SPEC_METADATA_TENSORS",
            "_GDNDdTreeFastCommonBuffers",
            "DFlash2GDNGroupDescriptor",
            "_GDN_DDTREE_FAST_COMMON_BUFFERS",
            "_get_ddtree_gdn_fast_common_buffers",
            "GDNAttentionMetadata",
            "SxSpecRows",
            "_SX_PIN_MEMORY",
            "_sx_pin_memory",
            "sx_h2d_nosync",
            "_sx_mtp_fused_gdn_meta_switch",
            "_sx_mtp_gdn_lane_contract",
            "sx_mtp_fused_gdn_metadata_admitted",
            "sx_mtp_pure_common_gdn_metadata",
            "_load_gdn_i32_ptr",
            "_dflash2_gdn_group_metadata_kernel",
            "sx_prepare_mtp_fused_gdn_metadata",
        ),
        namespace,
    )
    boot.cut(
        boot.GDN_ATTN,
        ("prepare_dflash2_gdn_group_metadata",),
        namespace,
        transform=_relax_cuda_guard,
    )
    kernel = FakeKernel(namespace["_dflash2_gdn_group_metadata_kernel"])
    namespace["_dflash2_gdn_group_metadata_kernel"] = kernel
    namespace["kernel"] = kernel
    return namespace


class FakeBuilder:
    def __init__(self, ns, *, width, mode, max_bs, block_size, buffers=True):
        self.num_spec_state_tokens = width - 1
        self.use_full_cuda_graph = True
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode=mode)
        )
        self.decode_cudagraph_max_bs = max_bs
        self._ddtree_fast_common_buffers = (
            ns["_get_ddtree_gdn_fast_common_buffers"](
                torch.device("cpu"), max_bs, width
            )
            if buffers
            else None
        )
        self.spec_state_indices_tensor = torch.full(
            (max_bs, width), 777, dtype=torch.int32
        )
        self.kv_cache_spec = SimpleNamespace(block_size=block_size)


# ---------------------------------------------------------------------------
# A verify step of the MRV2 runner (mamba_hybrid.prepare_attn conventions).
# ---------------------------------------------------------------------------
@dataclass
class Step:
    drafts: torch.Tensor  # CPU int32 [num_reqs_padded]
    qsl_cpu: torch.Tensor  # CPU int32 [num_reqs_padded + 1]
    qsl: torch.Tensor
    accepted: torch.Tensor  # int32 [num_reqs_padded]
    seq_lens: torch.Tensor  # int32 [num_reqs_padded]
    num_tokens: int
    num_spec: int


def make_step(rng, *, k, real, padded, block_size, query_lens=None, drafts=None):
    width = k + 1
    if query_lens is None:
        query_lens = [width] * real + [0] * (padded - real)
    if drafts is None:
        drafts = [k] * real + [-1] * (padded - real)
    qsl = np.zeros(padded + 1, dtype=np.int32)
    qsl[1:] = np.cumsum(query_lens)
    accepted = [rng.randint(1, width) for _ in range(real)] + [1] * (padded - real)
    seq_lens = []
    for _ in range(real):
        boundary = rng.randint(1, 6) * block_size
        seq_lens.append(max(width, boundary + rng.choice((-width, -1, 0, 1, width))))
    seq_lens += [0] * (padded - real)
    return Step(
        drafts=torch.tensor(drafts, dtype=torch.int32),
        qsl_cpu=torch.from_numpy(qsl),
        qsl=torch.from_numpy(qsl.copy()),
        accepted=torch.tensor(accepted, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_tokens=padded * width,
        num_spec=real,
    )


def make_tables(rng, groups, max_reqs, cols):
    tables = []
    for group in range(groups):
        values = list(range(1, max_reqs * cols + 1))
        rng.shuffle(values)
        tables.append(
            torch.tensor(values, dtype=torch.int32).reshape(max_reqs, cols)
            + 10000 * group
        )
    return tuple(tables)


def reference(step, table, *, mode, k, block_size):
    """The per-group FULL-graph result the builders produce for this step."""
    width = k + 1
    tokens, real = step.num_tokens, step.num_spec
    rows = step.drafts.numel()
    if mode == "align":
        source = mamba_get_block_table_tensor(
            table[:rows], step.seq_lens[:rows], FakeMambaSpec(block_size, k), mode
        )
    else:
        source = table[:rows]
    state = torch.full((tokens, width), PAD_SLOT_ID, dtype=torch.int32)
    state[:real] = source[:real, :width]
    masks = torch.arange(tokens) < real
    qsl = step.qsl[torch.clamp(torch.arange(tokens + 1), max=real)]
    accepted = torch.ones(tokens, dtype=torch.int32)
    accepted[:real] = step.accepted[:real]
    token_size = min(real * width, int(step.qsl_cpu[-1]))
    return dict(
        state=state,
        masks=masks,
        qsl=qsl,
        accepted=accepted,
        token_indx=torch.arange(token_size, dtype=torch.int32),
        num_spec_decode_tokens=int(step.qsl_cpu[-1]),
    )


def run_step(ns, builders, tables, step, *, mode, descriptor=None):
    # MRV2 hands the builders [num_reqs_padded, max_blocks] row slices.
    tables = tuple(table[: step.drafts.numel()] for table in tables)
    register(*tables, *(b.spec_state_indices_tensor for b in builders))
    return ns["sx_prepare_mtp_fused_gdn_metadata"](
        builders_by_group=list(enumerate(builders)),
        block_tables=tables,
        num_decode_draft_tokens_cpu=step.drafts,
        query_start_loc=step.qsl,
        query_start_loc_cpu=step.qsl_cpu,
        num_accepted_tokens=step.accepted,
        num_actual_tokens=step.num_tokens,
        descriptor=descriptor,
        seq_lens=step.seq_lens if mode == "align" else None,
    )


def assert_matches(prepared, builders, tables, step, *, mode, k, block_size):
    for builder, table in zip(builders, tables, strict=True):
        meta = prepared[id(builder)]
        ref = reference(step, table, mode=mode, k=k, block_size=block_size)
        assert (meta.num_prefills, meta.num_decodes) == (0, 0)
        assert meta.num_spec_decodes == step.num_spec
        assert meta.num_spec_decode_tokens == ref["num_spec_decode_tokens"]
        assert meta.num_actual_tokens == step.num_tokens
        assert torch.equal(meta.spec_state_indices_tensor, ref["state"])
        assert meta.spec_state_indices_tensor.data_ptr() == (
            builder.spec_state_indices_tensor.data_ptr()
        )
        assert torch.equal(meta.spec_sequence_masks, ref["masks"])
        assert torch.equal(meta.spec_query_start_loc, ref["qsl"])
        assert torch.equal(meta.num_accepted_tokens, ref["accepted"])
        assert torch.equal(meta.spec_state_slot_selectors, ref["accepted"])
        assert torch.equal(meta.spec_token_indx, ref["token_indx"])
        assert meta.non_spec_token_indx.numel() == 0
        for field in (
            "has_initial_state",
            "non_spec_query_start_loc",
            "non_spec_state_indices_tensor",
            "chunk_indices",
            "chunk_offsets",
            "ddtree_parent_ids",
            "nums_dict",
        ):
            assert getattr(meta, field) is None, field


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["align", "none"])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("block_size", [784, 16])
def test_fused_steps_match_per_group(gdn, mode, k, block_size):
    rng = random.Random(1000 * k + block_size + (mode == "align"))
    width = k + 1
    max_reqs = 8
    max_bs = max_reqs * width
    cols = 8 + width + 1
    builders = [
        FakeBuilder(gdn, width=width, mode=mode, max_bs=max_bs, block_size=block_size)
        for _ in range(3)
    ]
    descriptor = None
    for step_index, (real, padded) in enumerate(
        [(1, 1), (1, 2), (2, 2), (3, 4), (4, 4), (5, 6), (8, 8), (1, 1), (6, 8)]
    ):
        tables = make_tables(rng, 3, max_reqs, cols)
        step = make_step(rng, k=k, real=real, padded=padded, block_size=block_size)
        launches = gdn["kernel"].launches
        result = run_step(gdn, builders, tables, step, mode=mode, descriptor=descriptor)
        assert result is not None, (step_index, real, padded)
        common, prepared, new_descriptor = result
        assert gdn["kernel"].launches == launches + 1  # one launch, all groups
        if descriptor is not None and new_descriptor.key == descriptor.key:
            assert new_descriptor is descriptor
        descriptor = new_descriptor
        assert common.num_spec_decodes == real
        assert_matches(
            prepared, builders, tables, step, mode=mode, k=k, block_size=block_size
        )


def test_descriptor_reused_and_prepared_cached(gdn):
    rng = random.Random(7)
    builders = [
        FakeBuilder(gdn, width=5, mode="align", max_bs=40, block_size=784)
        for _ in range(3)
    ]
    tables = make_tables(rng, 3, 8, 14)
    first = run_step(
        gdn, builders, tables, make_step(rng, k=4, real=2, padded=2, block_size=784),
        mode="align",
    )
    second = run_step(
        gdn, builders, tables, make_step(rng, k=4, real=2, padded=2, block_size=784),
        mode="align", descriptor=first[2],
    )
    assert second[2] is first[2]
    assert all(second[1][key] is first[1][key] for key in first[1])


def _pure_case(gdn, rng, **overrides):
    k = overrides.pop("k", 4)
    builders = overrides.pop(
        "builders",
        [FakeBuilder(gdn, width=k + 1, mode="align", max_bs=40, block_size=784)
         for _ in range(3)],
    )
    tables = make_tables(rng, 3, 8, 14)
    step = make_step(rng, k=k, block_size=784, **overrides)
    return builders, tables, step


@pytest.mark.parametrize(
    "label",
    [
        "decode row",
        "prefill row",
        "non-prefix",
        "zero drafts",
        "oversized",
        "sx switch off",
        "upstream shared off",
        "upstream fused off",
        "no shared buffers",
        "align without seq_lens",
        "none with seq_lens",
        "cache mode all",
    ],
)
def test_fallbacks_keep_per_group_builds(gdn, monkeypatch, label):
    rng = random.Random(11)
    kwargs: dict[str, Any] = dict(real=2, padded=3)
    mode = "align"
    builders = None
    if label == "decode row":
        kwargs.update(query_lens=[5, 5, 1], drafts=[4, 4, -1])
    elif label == "prefill row":
        kwargs.update(query_lens=[5, 5, 300], drafts=[4, 4, -1])
    elif label == "non-prefix":
        kwargs.update(query_lens=[0, 5, 5], drafts=[-1, 4, 4])
    elif label == "zero drafts":
        kwargs.update(drafts=[0, 0, -1])
    elif label == "oversized":
        kwargs.update(real=9, padded=9)
    elif label == "sx switch off":
        monkeypatch.setenv("SX_OPT_MTP_GDN_FUSED_META", "0")
    elif label == "upstream shared off":
        gdn["envs"].VLLM_SM70_MTP4_SHARED_GDN_METADATA = False
    elif label == "upstream fused off":
        gdn["envs"].VLLM_SM70_MTP4_FUSED_GDN_METADATA = False
    elif label == "no shared buffers":
        builders = [
            FakeBuilder(gdn, width=5, mode="align", max_bs=40, block_size=784,
                        buffers=False)
            for _ in range(3)
        ]
    elif label == "cache mode all":
        mode = "all"
        builders = [
            FakeBuilder(gdn, width=5, mode="all", max_bs=40, block_size=784)
            for _ in range(3)
        ]
    if builders is not None:
        kwargs["builders"] = builders
    builders, tables, step = _pure_case(gdn, rng, **kwargs)
    launches = gdn["kernel"].launches
    if label == "align without seq_lens":
        step.seq_lens = None
        mode = "none"  # run_step passes seq_lens only for "align"
    if label == "none with seq_lens":
        builders = [
            FakeBuilder(gdn, width=5, mode="none", max_bs=40, block_size=784)
            for _ in range(3)
        ]
        mode = "align"
    result = run_step(gdn, builders, tables, step, mode=mode)
    assert result is None, label
    assert gdn["kernel"].launches == launches, label


@pytest.mark.parametrize("legacy", [False, True])
def test_host_classification_matches_compute_common(gdn, legacy):
    rng = random.Random(3)
    cases = []
    for k in (1, 2, 3, 4):
        for real, padded in ((1, 1), (1, 3), (4, 4), (5, 8)):
            cases.append(make_step(rng, k=k, real=real, padded=padded, block_size=784))
    for step in cases:
        k = int(step.drafts[0])
        expected = compute_common(
            num_decode_draft_tokens_cpu=step.drafts,
            query_start_loc=step.qsl,
            query_start_loc_cpu=step.qsl_cpu,
            num_spec_state_tokens=k,
            legacy_mixed_decode_routing=legacy,
        )
        actual = gdn["sx_mtp_pure_common_gdn_metadata"](
            num_decode_draft_tokens_cpu=step.drafts,
            query_start_loc=step.qsl,
            query_start_loc_cpu=step.qsl_cpu,
            num_spec_state_tokens=k,
        )
        assert expected is not None and actual is not None
        for field in (
            "num_prefills",
            "num_prefill_tokens",
            "num_decodes",
            "num_decode_tokens",
            "num_spec_decodes",
            "num_spec_decode_tokens",
        ):
            assert getattr(actual, field) == getattr(expected, field), field
        for field in (
            "spec_query_start_loc",
            "spec_sequence_masks_cpu",
            "spec_sequence_masks",
            "spec_token_indx",
            "non_spec_token_indx",
        ):
            assert torch.equal(getattr(actual, field), getattr(expected, field)), field
        assert actual.non_spec_query_start_loc is None
        assert actual.non_spec_query_start_loc_cpu is None
    # A shared arange buffer replaces the per-step arange.
    buffer = torch.arange(64, dtype=torch.int32)
    step = cases[-1]
    actual = gdn["sx_mtp_pure_common_gdn_metadata"](
        num_decode_draft_tokens_cpu=step.drafts,
        query_start_loc=step.qsl,
        query_start_loc_cpu=step.qsl_cpu,
        num_spec_state_tokens=4,
        spec_token_indx_buffer=buffer,
    )
    assert actual.spec_token_indx.data_ptr() == buffer.data_ptr()
    # Mixed batches: compute_common still classifies them, the host path not.
    mixed = make_step(rng, k=4, real=2, padded=3, block_size=784,
                      query_lens=[5, 5, 1], drafts=[4, 4, -1])
    assert compute_common(
        num_decode_draft_tokens_cpu=mixed.drafts,
        query_start_loc=mixed.qsl,
        query_start_loc_cpu=mixed.qsl_cpu,
        num_spec_state_tokens=4,
        legacy_mixed_decode_routing=legacy,
    ) is not None
    assert gdn["sx_mtp_pure_common_gdn_metadata"](
        num_decode_draft_tokens_cpu=mixed.drafts,
        query_start_loc=mixed.qsl,
        query_start_loc_cpu=mixed.qsl_cpu,
        num_spec_state_tokens=4,
    ) is None


def test_admission(gdn, monkeypatch):
    admitted = gdn["sx_mtp_fused_gdn_metadata_admitted"]
    state = {"lane": True}
    gdn["_sx_mtp_gdn_lane_contract"] = lambda cfg: state["lane"]
    cfg = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    assert admitted(cfg, "cuda") is True
    assert admitted(cfg, torch.device("cuda", 0)) is True
    assert admitted(cfg, "cpu") is False
    cfg.cache_config.mamba_cache_mode = "none"
    assert admitted(cfg, "cuda") is True
    cfg.cache_config.mamba_cache_mode = "all"
    assert admitted(cfg, "cuda") is False
    cfg.cache_config.mamba_cache_mode = "align"
    state["lane"] = False
    assert admitted(cfg, "cuda") is False
    state["lane"] = True
    gdn["envs"].VLLM_SM70_MTP4_FUSED_GDN_METADATA = False
    assert admitted(cfg, "cuda") is False
    gdn["envs"].VLLM_SM70_MTP4_FUSED_GDN_METADATA = True
    gdn["envs"].VLLM_SM70_MTP4_SHARED_GDN_METADATA = False
    assert admitted(cfg, "cuda") is False
    gdn["envs"].VLLM_SM70_MTP4_SHARED_GDN_METADATA = True
    monkeypatch.setenv("SX_OPT_MTP_GDN_FUSED_META", "0")
    assert admitted(cfg, "cuda") is False


def test_lane_contract_helper(gdn, monkeypatch):
    state = {"lane": True, "sm70": True, "raise": False}

    def contract(model_config, speculative_config, parallel_config):
        if state["raise"]:
            raise RuntimeError("partial config")
        state["args"] = (model_config, speculative_config, parallel_config)
        return state["lane"]

    config_mod = types.ModuleType("vllm.config.vllm")
    config_mod._is_sm70_qwen38_mtp_lane_contract = contract
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_device_capability=lambda cap: state["sm70"] and cap == 70
    )
    for name in ("vllm", "vllm.config"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", config_mod)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    helper = gdn["_sx_mtp_gdn_lane_contract"]
    cfg = SimpleNamespace(model_config="m", speculative_config="s", parallel_config="p")
    assert helper(cfg) is True and state["args"] == ("m", "s", "p")
    state["sm70"] = False
    assert helper(cfg) is False
    state["sm70"] = True
    state["lane"] = False
    assert helper(cfg) is False
    state["lane"] = True
    state["raise"] = True
    assert helper(cfg) is False


# ---------------------------------------------------------------------------
# MambaHybridModelState.prepare_attn wiring
# ---------------------------------------------------------------------------
FULL, PIECEWISE = "FULL", "PIECEWISE"


@pytest.fixture
def hybrid():
    calls: dict[str, list] = {"sx": [], "build": [], "common": [], "dflash2": []}
    state = {"result": None}

    @dataclass
    class ModelSpecificAttnMetadata:
        pass

    namespace = {
        "torch": torch,
        "np": np,
        "dataclass": dataclass,
        "Any": Any,
        "AttentionGroup": object,
        "InputBatch": object,
        "KVCacheConfig": object,
        "CommonGDNSpecMetadata": object,
        "GDNAttentionMetadata": object,
        "DFlash2SmallQPreparedMetadata": object,
        "ModelSpecificAttnMetadata": ModelSpecificAttnMetadata,
        "FlashAttnV100MetadataBuilder": type("F", (), {}),
        "Mamba2AttentionMetadataBuilder": type("M", (), {}),
        "GDNAttentionMetadataBuilder": type("G", (), {}),
        "PleShortConvAttentionMetadataBuilder": type("P", (), {}),
        "CUDAGraphMode": SimpleNamespace(FULL=FULL, PIECEWISE=PIECEWISE),
        "envs": SimpleNamespace(
            VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING=False,
            VLLM_SM70_DFLASH2_GDN_SYNC_ASSERT=False,
        ),
        "logger": SimpleNamespace(info=lambda *a, **k: None),
        "compute_common_gdn_attn_metadata": lambda **kw: calls["common"].append(kw),
        "prepare_dflash2_gdn_group_metadata": lambda **kw: calls["dflash2"].append(kw),
        "prepare_dflash2_smallq_group_metadata": lambda **kw: None,
        "build_attn_metadata": lambda **kw: calls["build"].append(kw) or {},
    }

    def sx_prepare(**kwargs):
        calls["sx"].append(kwargs)
        return state["result"]

    namespace["sx_prepare_mtp_fused_gdn_metadata"] = sx_prepare
    boot.cut(boot.MAMBA_HYBRID, ("MambaHybridAttnMetadata",), namespace)
    prepare_attn = boot.cut_method(
        boot.MAMBA_HYBRID, "MambaHybridModelState", "prepare_attn", namespace
    )
    return SimpleNamespace(prepare=prepare_attn, calls=calls, state=state)


def _model_state(*, fused=True, align=True):
    builders = [(0, object()), (2, object())]
    return SimpleNamespace(
        num_accepted_tokens_gpu=torch.tensor([3, 1, 2, 5], dtype=torch.int32),
        _use_dflash2_common_gdn_metadata=False,
        _use_dflash2_fused_gdn_metadata=False,
        _use_dflash2_grouped_smallq_metadata=False,
        _sx_mtp_fused_gdn_metadata=fused,
        _dflash2_gdn_group_descriptor="old-descriptor",
        _dflash2_fused_gdn_metadata_logged=False,
        _align_mode=align,
        _get_dflash2_gdn_builders=lambda attn_groups: builders,
        vllm_config=SimpleNamespace(),
        max_model_len=4096,
        builders=builders,
    )


def _input_batch():
    return SimpleNamespace(
        num_reqs=2,
        num_tokens=10,
        num_reqs_after_padding=3,
        num_tokens_after_padding=15,
        query_start_loc_np=np.array([0, 5, 10, 10], dtype=np.int32),
        num_scheduled_tokens=np.array([5, 5], dtype=np.int32),
        is_prefilling_np=np.array([False, False]),
        num_draft_tokens_per_req=np.array([4, 4]),
        idx_mapping=torch.tensor([3, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5, 10, 10], dtype=torch.int32),
        seq_lens=torch.tensor([800, 790, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([800, 790, 0], dtype=torch.int32),
        dcp_local_seq_lens=None,
        prefix_anchor_lens=None,
    )


def _call(hybrid, model_state, mode, for_capture=False):
    batch = _input_batch()
    hybrid.prepare(
        model_state,
        batch,
        mode,
        block_tables=("t0", "t1", "t2"),
        slot_mappings=None,
        attn_groups=[[], [], []],
        kv_cache_config=None,
        for_capture=for_capture,
    )
    return batch, hybrid.calls["build"][-1]["model_specific_attn_metadata"]


@pytest.mark.parametrize("align", [True, False])
def test_prepare_attn_runs_fused_step_on_full_graphs(hybrid, align):
    model_state = _model_state(align=align)
    common = SimpleNamespace(
        num_spec_decodes=2, spec_query_start_loc=torch.tensor([0, 5, 10])
    )
    hybrid.state["result"] = (common, {"g": "prepared"}, "new-descriptor")
    batch, meta = _call(hybrid, model_state, FULL)
    (kwargs,) = hybrid.calls["sx"]
    assert kwargs["builders_by_group"] is model_state.builders
    assert kwargs["block_tables"] == ("t0", "t1", "t2")
    assert kwargs["num_decode_draft_tokens_cpu"].tolist() == [4, 4, -1]
    assert kwargs["query_start_loc"] is batch.query_start_loc
    assert kwargs["query_start_loc_cpu"].tolist() == [0, 5, 10, 10]
    assert kwargs["num_accepted_tokens"].tolist() == [5, 3, 1]
    assert kwargs["num_actual_tokens"] == 15
    assert kwargs["descriptor"] == "old-descriptor"
    assert (kwargs["seq_lens"] is batch.seq_lens) == align
    assert kwargs["seq_lens"] is None or align
    assert meta.common_gdn_metadata is common
    assert meta.prepared_dflash2_gdn_metadata == {"g": "prepared"}
    assert model_state._dflash2_gdn_group_descriptor == "new-descriptor"
    assert hybrid.calls["common"] == [] and hybrid.calls["dflash2"] == []


def test_prepare_attn_failed_fused_step_keeps_per_group(hybrid):
    model_state = _model_state()
    hybrid.state["result"] = None
    _, meta = _call(hybrid, model_state, FULL)
    assert len(hybrid.calls["sx"]) == 1
    assert meta.common_gdn_metadata is None
    assert meta.prepared_dflash2_gdn_metadata is None
    assert model_state._dflash2_gdn_group_descriptor == "old-descriptor"


@pytest.mark.parametrize(
    "label, fused, mode, for_capture",
    [
        ("piecewise", True, PIECEWISE, False),
        ("capture", True, FULL, True),
        ("not admitted", False, FULL, False),
    ],
)
def test_prepare_attn_skips_fused_step(hybrid, label, fused, mode, for_capture):
    hybrid.state["result"] = ("common", {"g": "prepared"}, "new-descriptor")
    _, meta = _call(hybrid, _model_state(fused=fused), mode, for_capture)
    assert hybrid.calls["sx"] == [], label
    assert meta.common_gdn_metadata is None, label
    assert meta.prepared_dflash2_gdn_metadata is None, label


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
