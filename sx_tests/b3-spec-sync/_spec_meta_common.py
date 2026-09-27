# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the batch-3a spec-sync tests (SX_OPT_SPEC_META_NOSYNC).

Not a test module. The sibling test_*.py files import it first (pytest puts
this directory on sys.path). Run inside image 1.8.0-dev2 with the two patched
backend files bind-mounted over the installed package:

  -v $SRC/vllm/v1/attention/backends/gdn_attn.py:\
/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py:ro
  -v $SRC/vllm/v1/attention/backends/short_conv_attn.py:\
/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/short_conv_attn.py:ro

The builders are the real GDNAttentionMetadataBuilder and
PleShortConvAttentionMetadataBuilder, constructed with a minimal config
namespace (the only stubbed import is the GDN prefill-backend resolver, which
the builders merely record). Batches follow the MRV2 runner contract
(mamba_hybrid.prepare_attn): spec rows carry k drafts, plain decodes /
prefills / graph padding carry -1, FULL-graph batches are padded at the back
with zero-length requests, num_accepted_tokens is an int32 device tensor.
"""

from __future__ import annotations

import dataclasses
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent


def _prefer_installed_vllm() -> None:
    """Keep ``import vllm`` on the installed (overlaid) package."""
    repo_root = HERE.parents[1]
    if any((repo_root / "vllm").glob("_C*.so")):
        return
    for entry in list(sys.path):
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            continue
        if resolved == repo_root:
            sys.path.remove(entry)


_prefer_installed_vllm()

import torch  # noqa: E402

# Deployment geometry: Qwen3.8-Flash-Next MTP lane (align prefix caching,
# mamba block = scheduler block 784, max_num_seqs 24, k = 1..4).
MAMBA_BLOCK = 784
MAX_MODEL_LEN = 65536
MAX_NUM_SEQS = 24
K_VALUES = (1, 2, 3, 4)
B_VALUES = (1, 2, 4, 8, 12, 16, 24)
# MTP FULL-graph request sizes (_SM70_MTP_CUDAGRAPH_REQUEST_SIZES) + 24.
PAD_SIZES = (1, 2, 3, 4, 6, 8, 12, 16, 24)
PREFILL_LENS = (2, 17, 64, 333, 784, 1500, 2000, 7840)
MODES = ("align", "none", "all")

_GDN_LAYER_MOD = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def fake_vllm_config(
    k: int,
    mode: str,
    *,
    full: bool,
    method: str | None = "mtp",
    capture: int | None = None,
) -> NS:
    spec = None
    if method is not None and k > 0:
        spec = NS(
            num_speculative_tokens=k,
            parallel_drafting=False,
            method=method,
            num_speculative_state_tokens=lambda: k,
        )
    return NS(
        speculative_config=spec,
        num_speculative_tokens=k if spec is not None else 0,
        compilation_config=NS(
            cudagraph_mode=NS(has_full_cudagraphs=lambda: full),
            max_cudagraph_capture_size=capture,
        ),
        scheduler_config=NS(max_num_seqs=MAX_NUM_SEQS),
        cache_config=NS(mamba_cache_mode=mode),
        parallel_config=NS(
            decode_context_parallel_size=1,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        model_config=NS(max_model_len=MAX_MODEL_LEN),
    )


def _mamba_spec(k: int, mode: str, *, replicated: bool):
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        block_size=MAMBA_BLOCK,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mode,
        num_speculative_blocks=k,
        tp_replicated=replicated,
    )


def make_gdn_builder(monkeypatch, k: int, mode: str, device: str, *, full: bool,
                     nosync: bool | None, vllm_config: NS | None = None):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    fake = types.ModuleType(_GDN_LAYER_MOD)
    fake._resolve_gdn_prefill_backend = lambda cfg: (None, "flashqla_sm70")
    monkeypatch.setitem(sys.modules, _GDN_LAYER_MOD, fake)
    cfg = vllm_config or fake_vllm_config(k, mode, full=full)
    builder = GDNAttentionMetadataBuilder(
        _mamba_spec(k, mode, replicated=False),
        ["model.layers.0.linear_attn"],
        cfg,
        torch.device(device),
    )
    if nosync is not None:
        builder._sx_spec_meta_nosync = nosync
    return builder


def make_ple_builder(k: int, mode: str, device: str, *, full: bool,
                     nosync: bool | None, vllm_config: NS | None = None):
    from vllm.v1.attention.backends.short_conv_attn import (
        PleShortConvAttentionMetadataBuilder,
    )

    cfg = vllm_config or fake_vllm_config(k, mode, full=full)
    builder = PleShortConvAttentionMetadataBuilder(
        _mamba_spec(k, mode, replicated=True),
        ["model.layers.0.ple"],
        cfg,
        torch.device(device),
    )
    if nosync is not None:
        builder._sx_spec_meta_nosync = nosync
    return builder


def set_gdn_envs(monkeypatch, *, legacy_routing: bool, legacy_slot0: bool) -> None:
    from vllm import envs

    for name, flag in (
        ("VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING", legacy_routing),
        ("VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0", legacy_slot0),
    ):
        monkeypatch.setenv(name, "1" if flag else "0")
        if bool(getattr(envs, name)) != flag:  # envs cache enabled
            monkeypatch.setattr(envs, name, flag, raising=False)
        assert bool(getattr(envs, name)) == flag


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------
@dataclass
class SpecBatch:
    name: str
    k: int
    query_lens: list[int]
    drafts: list[int]
    seq_lens: list[int]
    accepted: list[int]
    num_actual_tokens: int
    full: bool  # built by a FULL-graph (uniform decode) builder
    has_prefill: bool = False


def _spec_drafts(k: int, rng: random.Random, ragged: bool) -> int:
    # ragged: per-request draft counts differ (adaptive k, requests near
    # max_tokens); 0 drafts is a V1-runner spec row with a 1-token query.
    return rng.randint(0, k) if ragged else k


def verify_batch(k: int, b: int, rng: random.Random, *, full: bool,
                 ragged: bool = False) -> SpecBatch:
    """Pure MTP verify step: b live spec rows (+ FULL-graph padding rows)."""
    pad_to = next(p for p in PAD_SIZES if p >= b) if full else b
    pad = pad_to - b
    live_drafts = [_spec_drafts(k, rng, ragged) for _ in range(b)]
    if ragged and not any(live_drafts):
        live_drafts[0] = k  # keep at least one draft token in the step
    qlens = [d + 1 for d in live_drafts] + [0] * pad
    drafts = live_drafts + [-1] * pad
    seqs = [rng.randint(1, 20000) + q for q in qlens[:b]] + [0] * pad
    acc = [rng.randint(1, k + 1) for _ in range(b)] + [1] * pad
    return SpecBatch(
        name=(
            f"verify_{'full' if full else 'eager'}{'_ragged' if ragged else ''}"
            f"_k{k}_b{b}_p{pad_to}"
        ),
        k=k,
        query_lens=qlens,
        drafts=drafts,
        seq_lens=seqs,
        accepted=acc,
        num_actual_tokens=pad_to * (k + 1) if full else sum(qlens),
        full=full,
    )


def nospec_batch(k: int, b: int, rng: random.Random, *, kind: str) -> SpecBatch:
    """A spec-capable builder's step without any spec row (-1 drafts only):
    plain decodes, or plain decodes + prefill chunks (reordered batch:
    decodes first, as split_decodes_and_prefills expects)."""
    qlens, drafts, seqs, acc = [], [], [], []
    num_dec = b // 2 if kind == "prefill" else b
    for i in range(b):
        if i >= num_dec:
            q = PREFILL_LENS[i % len(PREFILL_LENS)]
            computed = rng.choice((0, MAMBA_BLOCK))
        else:
            q = 1
            computed = rng.randint(1, 15000)
        qlens.append(q)
        drafts.append(-1)
        seqs.append(computed + q)
        acc.append(rng.randint(1, k + 1))
    return SpecBatch(
        name=f"nospec_{kind}_k{k}_b{b}",
        k=k,
        query_lens=qlens,
        drafts=drafts,
        seq_lens=seqs,
        accepted=acc,
        num_actual_tokens=sum(qlens),
        full=False,
        has_prefill=kind == "prefill",
    )


def mixed_batch(k: int, b: int, rng: random.Random, *, order: str,
                with_prefill: bool = True, zero_tail: int = 0,
                ragged: bool = False) -> SpecBatch:
    """Spec decodes + plain decodes (0 drafts) + prefill chunks."""
    assert b >= 2
    n_spec = max(1, b // 2)
    rest = b - n_spec
    if with_prefill:
        n_pre = max(1, rest // 2)
        n_dec = rest - n_pre
    else:
        n_pre = 0
        n_dec = rest
    kinds = ["s"] * n_spec + ["d"] * n_dec + ["p"] * n_pre
    if order == "shuffled":
        rng.shuffle(kinds)
        if kinds[:n_spec] == ["s"] * n_spec and b > n_spec:
            kinds.append(kinds.pop(0))  # force a non-prefix mask
    elif order == "spec_last":
        kinds = kinds[n_spec:] + kinds[:n_spec]
    else:
        assert order == "spec_first"
    qlens, drafts, seqs, acc = [], [], [], []
    pre_i = rng.randrange(len(PREFILL_LENS))
    for kind in kinds:
        ctx = rng.randint(1, 15000)
        if kind == "s":
            d = _spec_drafts(k, rng, ragged)
            if ragged and d == 0 and not any(x > 0 for x in drafts):
                d = k  # keep at least one draft token in the step
            qlens.append(d + 1)
            drafts.append(d)
            seqs.append(ctx + d + 1)
            acc.append(rng.randint(1, k + 1))
        elif kind == "d":
            qlens.append(1)
            drafts.append(-1)
            seqs.append(ctx + 1)
            # Rows that got 0 drafts keep the last verify's accepted count.
            acc.append(rng.randint(1, k + 1))
        else:
            q = PREFILL_LENS[pre_i % len(PREFILL_LENS)]
            pre_i += 1
            computed = rng.choice((0, 0, MAMBA_BLOCK, 3 * MAMBA_BLOCK))
            qlens.append(q)
            drafts.append(-1)
            seqs.append(computed + q)
            acc.append(1)
    qlens += [0] * zero_tail
    drafts += [-1] * zero_tail
    seqs += [0] * zero_tail
    acc += [1] * zero_tail
    return SpecBatch(
        name=(
            f"mixed_{order}{'' if with_prefill else '_nopre'}"
            f"{'_ragged' if ragged else ''}_z{zero_tail}_k{k}_b{b}"
        ),
        k=k,
        query_lens=qlens,
        drafts=drafts,
        seq_lens=seqs,
        accepted=acc,
        num_actual_tokens=sum(qlens),
        full=False,
        has_prefill=n_pre > 0,
    )


def batches_for(k: int, b: int, seed: int) -> list[SpecBatch]:
    rng = random.Random(seed)
    # The first two entries (pure verify FULL / eager) are relied upon by
    # callers that slice ``[:1]`` / ``[:2]``.
    out = [
        verify_batch(k, b, rng, full=True),
        verify_batch(k, b, rng, full=False),
        verify_batch(k, b, rng, full=False, ragged=True),
        nospec_batch(k, b, rng, kind="decode"),
        nospec_batch(k, b, rng, kind="prefill"),
    ]
    if b >= 2:
        out += [
            mixed_batch(k, b, rng, order="spec_first"),
            mixed_batch(k, b, rng, order="shuffled"),
            mixed_batch(k, b, rng, order="spec_last"),
            mixed_batch(k, b, rng, order="shuffled", zero_tail=2),
            mixed_batch(k, b, rng, order="shuffled", with_prefill=False),
            mixed_batch(k, b, rng, order="spec_first", ragged=True),
            mixed_batch(k, b, rng, order="spec_first", with_prefill=False),
        ]
    return out


def table_width(mode: str, k: int) -> int:
    if mode == "none":
        return k + 1
    if mode == "all":
        return cdiv(MAX_MODEL_LEN, MAMBA_BLOCK) + k
    return cdiv(MAX_MODEL_LEN, MAMBA_BLOCK) + k + 1


@dataclass
class BatchInputs:
    query_start_loc_cpu: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    drafts_cpu: torch.Tensor
    accepted: torch.Tensor
    selectors: torch.Tensor
    batch: SpecBatch
    # V1 GPUModelRunner align mode: ``current_mamba_state_block_ids`` view,
    # [rows, 1 + num_speculative_state_tokens] int32, PAD_SLOT_ID (-1) where a
    # request has no state block (graph padding, tail of a short table).
    current_state: torch.Tensor | None = None

    def common(self):
        from vllm.v1.attention.backend import CommonAttentionMetadata

        # Fresh object per build (no shared lazily cached fields). MRV2 sets
        # is_prefilling (CPU bool; prefill chunks only) and the CPU seq-len
        # upper bound on every step.
        is_prefilling = torch.tensor(
            [
                draft < 0 and qlen > 1
                for qlen, draft in zip(self.batch.query_lens, self.batch.drafts)
            ],
            dtype=torch.bool,
        )
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc,
            query_start_loc_cpu=self.query_start_loc_cpu,
            seq_lens=self.seq_lens,
            num_reqs=len(self.batch.query_lens),
            num_actual_tokens=self.batch.num_actual_tokens,
            max_query_len=max(max(self.batch.query_lens), 1),
            max_seq_len=max(max(self.batch.seq_lens), 1),
            block_table_tensor=self.block_table,
            slot_mapping=self.slot_mapping,
            is_prefilling=is_prefilling,
            seq_lens_cpu_upper_bound=torch.tensor(
                self.batch.seq_lens, dtype=torch.int32
            ),
        )


def make_inputs(batch: SpecBatch, mode: str, device: str, seed: int) -> BatchInputs:
    gen = torch.Generator().manual_seed(seed)
    rows = len(batch.query_lens)
    width = table_width(mode, batch.k)
    qsl_cpu = torch.zeros(rows + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.tensor(batch.query_lens, dtype=torch.int32).cumsum(0)
    ids = torch.randperm(rows * width * 2 + 8, generator=gen)[: rows * width]
    block_table = (ids.view(rows, width) + 1).to(torch.int32)
    k = batch.k
    selectors = torch.randint(1, k + 2, (rows,), generator=gen, dtype=torch.int32)
    # Runner buffer: max_num_reqs rows, sliced to the (padded) request count.
    max_reqs = max(MAX_NUM_SEQS, rows)
    state_ids = torch.randperm(max_reqs * (k + 1) * 4 + 8, generator=gen)
    current_state = torch.full((max_reqs, k + 1), -1, dtype=torch.int32)
    for row, (qlen, draft) in enumerate(zip(batch.query_lens, batch.drafts)):
        if qlen == 0:
            continue  # padding row: PAD_SLOT_ID everywhere
        valid = k + 1 if draft >= 0 or row % 3 else 1 + row % (k + 1)
        current_state[row, :valid] = (
            state_ids[row * (k + 1) : row * (k + 1) + valid] + 1
        ).to(torch.int32)
    return BatchInputs(
        query_start_loc_cpu=qsl_cpu,
        query_start_loc=qsl_cpu.to(device),
        seq_lens=torch.tensor(batch.seq_lens, dtype=torch.int32, device=device),
        block_table=block_table.to(device),
        slot_mapping=torch.zeros(
            batch.num_actual_tokens, dtype=torch.int64, device=device
        ),
        drafts_cpu=torch.tensor(batch.drafts, dtype=torch.int32),
        accepted=torch.tensor(batch.accepted, dtype=torch.int32, device=device),
        selectors=selectors.to(device),
        batch=batch,
        current_state=current_state.to(device)[:rows],
    )


def build_gdn(builder, inputs: BatchInputs, *, selectors: bool = False,
              current_state: bool = False):
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=inputs.common(),
        num_accepted_tokens=inputs.accepted,
        spec_state_slot_selectors=inputs.selectors if selectors else None,
        num_decode_draft_tokens_cpu=inputs.drafts_cpu,
        current_state_block_ids=inputs.current_state if current_state else None,
    )


def build_ple(builder, inputs: BatchInputs):
    return builder.build(
        0,
        inputs.common(),
        num_accepted_tokens=inputs.accepted,
        num_decode_draft_tokens_cpu=inputs.drafts_cpu,
    )


# ---------------------------------------------------------------------------
# Metadata snapshots / comparison
# ---------------------------------------------------------------------------
def _layout(t: torch.Tensor) -> tuple:
    strides = tuple(s for s, n in zip(t.stride(), t.shape) if n > 1)
    return (t.is_contiguous(), strides)


def _aligned16(t: torch.Tensor) -> bool:
    # Triton specializes kernels on ``data_ptr % 16 == 0`` and vectorized CUDA
    # loads may assume it: a metadata tensor that was 16-byte aligned on the
    # legacy path (fresh allocation / offset-0 view) must stay aligned.
    return t.numel() == 0 or t.data_ptr() % 16 == 0


def _snap(value):
    if isinstance(value, torch.Tensor):
        return (
            "T",
            value.dtype,
            value.device.type,
            tuple(value.shape),
            _layout(value),
            value.detach().clone(),
            _aligned16(value),
        )
    if isinstance(value, dict):
        return ("D", {key: _snap(val) for key, val in value.items()})
    if isinstance(value, (list, tuple)):
        return ("L", type(value).__name__, [_snap(val) for val in value])
    return ("V", value)


def snapshot(metadata) -> dict:
    """Clone every field now (FULL-graph metadata views persistent buffers)."""
    return {
        field.name: _snap(getattr(metadata, field.name))
        for field in dataclasses.fields(metadata)
    }


def _diff(path: str, old, new, out: list[str]) -> None:
    if old[0] != new[0]:
        out.append(f"{path}: kind {old[0]} != {new[0]}")
        return
    kind = old[0]
    if kind == "T":
        _, dt0, dev0, sh0, lay0, t0, al0 = old
        _, dt1, dev1, sh1, lay1, t1, al1 = new
        if (dt0, dev0, sh0) != (dt1, dev1, sh1):
            out.append(f"{path}: {dt0}/{dev0}/{sh0} != {dt1}/{dev1}/{sh1}")
            return
        if lay0 != lay1:
            out.append(f"{path}: layout {lay0} != {lay1}")
        if al0 and not al1:
            out.append(f"{path}: 16-byte aligned on the legacy path, not on the new one")
        if not torch.equal(t0.cpu(), t1.cpu()):
            out.append(f"{path}: values differ\n  old={t0.cpu().tolist()}\n"
                       f"  new={t1.cpu().tolist()}")
    elif kind == "D":
        if old[1].keys() != new[1].keys():
            out.append(f"{path}: keys {sorted(old[1])} != {sorted(new[1])}")
            return
        for key in old[1]:
            _diff(f"{path}[{key!r}]", old[1][key], new[1][key], out)
    elif kind == "L":
        if old[1] != new[1] or len(old[2]) != len(new[2]):
            out.append(f"{path}: {old[1]}[{len(old[2])}] != {new[1]}[{len(new[2])}]")
            return
        for i, (a, b) in enumerate(zip(old[2], new[2])):
            _diff(f"{path}[{i}]", a, b, out)
    else:
        if old[1] != new[1] or type(old[1]) is not type(new[1]):
            out.append(f"{path}: {old[1]!r} != {new[1]!r}")


def metadata_diff(old: dict, new: dict) -> list[str]:
    out: list[str] = []
    if old.keys() != new.keys():
        return [f"fields {sorted(old)} != {sorted(new)}"]
    for name in old:
        _diff(name, old[name], new[name], out)
    return out


def build_outcome(fn, *args, **kwargs):
    """Snapshot of a build, or the exception type it raised.

    Both paths must agree: equal metadata, or the same exception type (e.g.
    the pre-existing no-spec-row PLE limitation in mamba_cache_mode "all",
    which runs the untouched legacy branch on both paths)."""
    try:
        return ("ok", snapshot(fn(*args, **kwargs)))
    except Exception as exc:  # noqa: BLE001 - compared, not swallowed
        return ("error", type(exc).__name__)


def outcome_diff(old, new) -> list[str]:
    if old[0] == "ok" and new[0] == "ok":
        return metadata_diff(old[1], new[1])
    if old != new:
        return [f"outcome {old if old[0] == 'error' else 'ok'} != "
                f"{new if new[0] == 'error' else 'ok'}"]
    return []
