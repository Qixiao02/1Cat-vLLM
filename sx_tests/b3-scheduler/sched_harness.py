# SPDX-License-Identifier: Apache-2.0
"""Scheduler-level harness for the SX align chunk policy (CPU only).

Drives the REAL ``Scheduler`` / ``AsyncScheduler`` (``schedule()``,
``update_from_output()``, ``update_draft_token_ids()``) and the REAL
``KVCacheManager`` (hybrid coordinator, BlockPool, MambaManager align
allocation with ``num_speculative_blocks``) like EngineCore does, with a
Qwen3.8-like hybrid layout (QSA main + compressed + ring, 3 GDN groups, PLE
state), plus a worker model that mirrors the MRV2 align-mode state handling
with native-MTP speculative decoding:

* pre-process (``preprocess_mamba_align_fused_kernel`` + pre-copy): src column
  = previous state column (seeded as ``cdiv(num_computed, B) - 1``),
  ``token_bias = num_accepted - 1``, dst column =
  ``cdiv(num_computed + num_scheduled, B) - 1``; on a column change the state
  at ``src + token_bias`` is copied to dst and ``num_accepted`` becomes 1.
* forward: a prefill row reads/writes its state in place at column
  ``dst + num_accepted - 1``; a verify (decode) row reads that column and
  writes the state after each of its ``1 + m`` tokens to ``dst + j``
  (the target model processes the last sampled token plus m drafts).
* post-process (``_postprocess_mamba_align_kernel``): ``num_accepted =
  max(num_sampled, 1)``; when the accepted tokens reach a block boundary the
  state at that boundary is copied into the boundary's checkpoint column.

The oracle keeps, per (Mamba group, physical block), the identity of the
recurrent state it holds, ``(num_tokens, hash(token prefix))`` -- including
states of draft prefixes that verification rejects -- and raises
``OracleError`` when
1. a forward starts from a state other than exactly the state after the
   request's own computed prefix;
2. a pre-copy/post-copy/forward touches a null column or a block that the
   request no longer holds (``ref_cnt == 0``);
3. an interior column of a multi-block prefill chunk is a real block, or one
   of the ``1 + k`` gathered state columns ``[dst, dst + k]`` is null;
4. a Mamba block registered in the prefix cache holds anything other than
   the state after exactly the tokens its hash covers;
5. a finished request's output differs from the reference token sequence.

Draft tokens follow a reference sequence ``true_token(req, pos)``: the first
``a - 1`` drafts of a step are correct and the rest are wrong, with ``a``
drawn per step from the measured per-position acceptance (so rejection
sampling accepts exactly ``a - 1`` drafts).

Options: ``async_sched`` (AsyncScheduler; the output of step N reaches the
scheduler after step N+1 was scheduled, drafts stay worker-side like MRV2),
``attn_block_size`` (attention groups on a smaller block than the recurrent
state, e.g. 16-token pages), ``num_blocks`` (small pools preempt naturally),
``env`` (SX switches seen by ``Scheduler.__init__``; all others unset).

Runs against the installed vLLM (the deployed image) or any tree that makes
``vllm.v1.core.sched.scheduler`` importable.
"""

from __future__ import annotations

import os
import random
import zlib
from collections import defaultdict, deque
from contextlib import contextmanager
from types import SimpleNamespace

SX_ENV_NAMES = (
    "SX_OPT_ALIGN_MULTIBLOCK",
    "SX_OPT_ALIGN_MULTIBLOCK_SPEC",
    "SX_OPT_ALIGN_TAIL_CHECKPOINT",
    "SX_OPT_ALIGN_TAIL_MIN_TOKENS",
    "SX_OPT_ALIGN_MAX_CHUNK_BLOCKS",
    "SX_OPT_ALIGN_SHARED_CHECKPOINT",
    "SX_OPT_PREFILL_CAP_WITH_DECODES",
    "SX_OPT_PREFILL_CAP_BLOCKS",
    "SX_OPT_PREFILL_CAP_TOKENS",
)

QWEN38_TEXT = dict(
    hidden_size=2560,
    num_hidden_layers=48,
    num_experts=512,
    num_experts_per_tok=10,
    moe_intermediate_size=640,
    hc_count=4,
    hc_lowrank=320,
    num_attention_heads=24,
    num_key_value_heads=2,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)

WRONG_DRAFT = 90_000


class OracleError(AssertionError):
    pass


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def true_token(request_id: str, position: int) -> int:
    return 50_000 + zlib.crc32(f"{request_id}:{position}".encode()) % 4096


def make_prompt(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(10, 30_000) for _ in range(n)]


def measured_acceptance(rng: random.Random, k: int) -> int:
    """Accepted length a in [1, k+1]; per-position rates as measured on the
    1.8.0-dev2 MTP lane (0.95 / 0.90 / 0.86 / 0.82, then 0.8)."""
    rates = (0.95, 0.90, 0.86, 0.82, 0.80)
    a = 1
    for j in range(k):
        if rng.random() >= rates[min(j, len(rates) - 1)]:
            break
        a += 1
    return a


@contextmanager
def sx_env(env: dict[str, str] | None):
    """Set exactly ``env`` for the SX switches (others unset) while the
    Scheduler resolves its policy."""
    saved = {name: os.environ.get(name) for name in SX_ENV_NAMES}
    for name in SX_ENV_NAMES:
        os.environ.pop(name, None)
    for name, value in (env or {}).items():
        os.environ[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _FakeStructuredOutputManager:
    def should_advance(self, request, new_token_ids=None) -> bool:
        return False


class _FakeMMRegistry:
    def supports_multimodal_inputs(self, model_config) -> bool:
        return False


def make_spec_config(num_spec: int, method: str = "mtp", native: bool = True,
                     parallel_drafting: bool = False):
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_spec,
        num_speculative_state_tokens=lambda: num_spec,
        parallel_drafting=parallel_drafting,
        use_qwen4_exp_mtp=lambda: bool(native and method == "mtp"),
        use_eagle_kv_cache=lambda: method in ("mtp", "eagle", "eagle3"),
        use_dflash_ddtree=lambda: False,
        ddtree_disable_tree_verify=False,
        draft_model_config=SimpleNamespace(max_model_len=None),
    )


def make_vllm_config(*, block_size: int, num_blocks: int, budget: int,
                     max_num_seqs: int, max_model_len: int, num_spec: int,
                     method: str = "mtp", native: bool = True, tp: int = 4,
                     parallel_drafting: bool = False):
    import torch

    model_config = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        multimodal_config=SimpleNamespace(language_model_only=True),
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(**QWEN38_TEXT),
        is_encoder_decoder=False,
        is_hybrid=True,
        max_model_len=max_model_len,
        enable_return_routed_experts=False,
    )
    spec = (make_spec_config(num_spec, method, native, parallel_drafting)
            if num_spec else None)
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_scheduled_tokens=None,
            max_num_batched_tokens=budget,
            policy="fcfs",
            long_prefill_token_threshold=0,
            enable_chunked_prefill=True,
            scheduler_reserve_full_isl=False,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=num_blocks,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            block_size=block_size,
        ),
        lora_config=None,
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=tp,
        ),
        observability_config=SimpleNamespace(
            kv_cache_metrics=False,
            kv_cache_metrics_sample=0.0,
            enable_mfu_metrics=False,
        ),
        model_config=model_config,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=spec,
        num_speculative_tokens=num_spec,
        num_lookahead_tokens=num_spec,
        max_in_flight_tokens=None,
        use_v2_model_runner=True,
    )


def make_kv_cache_config(block_size: int, num_blocks: int, num_spec: int,
                         layout: str = "qwen38", attn_block_size: int | None = None):
    import torch

    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
    )

    t = torch
    attn_block = attn_block_size or block_size
    full = FullAttentionSpec(
        block_size=attn_block, num_kv_heads=1, head_size=1, dtype=t.float32
    )
    main_layers = ["qsa.main"] + (["mtp.qsa"] if num_spec else [])
    groups = [KVCacheGroupSpec(main_layers, full)]
    if layout == "qwen38":
        groups.append(
            KVCacheGroupSpec(
                ["qsa.compressed"],
                MLAAttentionSpec(
                    block_size=attn_block,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=t.float32,
                    compress_ratio=4,
                ),
            )
        )
        groups.append(
            KVCacheGroupSpec(
                ["qsa.ring"],
                CircularBufferSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    head_size_v=0,
                    dtype=t.float32,
                ),
            )
        )
    gdn = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(t.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
    )
    for i in range(3 if layout == "qwen38" else 1):
        groups.append(KVCacheGroupSpec([f"gdn{i}"], gdn))
    if layout == "qwen38":
        groups.append(
            KVCacheGroupSpec(
                ["ple"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((2,),),
                    dtypes=(t.float16,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=num_spec,
                    tp_replicated=True,
                ),
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


class _Worker:
    """Worker-side view of one request (MRV2 GPU state, exact counts)."""

    def __init__(self, tokens: list[int], num_computed: int) -> None:
        self.tokens = list(tokens)  # prompt + sampled outputs known to the worker
        self.computed = num_computed
        self.state_idx = -2  # seeded at the first pre-process
        self.num_accepted = 1
        self.drafts: list[int] = []


class SchedHarness:
    def __init__(
        self,
        *,
        block_size: int = 784,
        hash_block_size: int | None = None,
        budget: int = 8192,
        max_num_seqs: int = 24,
        num_blocks: int = 4096,
        max_model_len: int = 131072,
        num_spec: int = 0,
        method: str = "mtp",
        native_mtp: bool = True,
        env: dict[str, str] | None = None,
        async_sched: bool = False,
        sm70: bool = True,
        seed: int = 0,
        accept=None,
        layout: str = "qwen38",
        tp: int = 4,
        worker_bug: str | None = None,
        attn_block_size: int | None = None,
        parallel_drafting: bool = False,
    ) -> None:
        from vllm.config import vllm as config_vllm
        from vllm.utils.hashing import sha256
        from vllm.v1.core import kv_cache_utils
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        kv_cache_utils.init_none_hash(sha256)
        self._sha256 = sha256
        self._kvu = kv_cache_utils
        self.B = block_size
        self.hash_B = hash_block_size or block_size
        assert self.B % self.hash_B == 0
        self.k = num_spec
        self.async_sched = async_sched
        self.rng = random.Random(seed)
        self.accept = accept or measured_acceptance
        # Negative controls only: "skip_postcopy" drops the decode-boundary
        # checkpoint copy of the post-process.
        self.worker_bug = worker_bug
        self.vllm_config = make_vllm_config(
            block_size=block_size, num_blocks=num_blocks, budget=budget,
            max_num_seqs=max_num_seqs, max_model_len=max_model_len,
            num_spec=num_spec, method=method, native=native_mtp, tp=tp,
            parallel_drafting=parallel_drafting,
        )
        self.kv_cache_config = make_kv_cache_config(
            block_size, num_blocks, num_spec, layout, attn_block_size
        )
        cls = AsyncScheduler if async_sched else Scheduler
        orig_cap = config_vllm._any_participating_device_is_capability
        config_vllm._any_participating_device_is_capability = (
            lambda cfg, cap: bool(sm70) and tuple(cap) == (7, 0)
        )
        try:
            with sx_env(env):
                self.sched = cls(
                    vllm_config=self.vllm_config,
                    kv_cache_config=self.kv_cache_config,
                    structured_output_manager=_FakeStructuredOutputManager(),
                    block_size=block_size,
                    hash_block_size=self.hash_B,
                    mm_registry=_FakeMMRegistry(),
                    include_finished_set=False,
                    log_stats=False,
                )
        finally:
            config_vllm._any_participating_device_is_capability = orig_cap
        self.manager = self.sched.kv_cache_manager
        self.mamba_group_ids = [
            i
            for i, g in enumerate(self.kv_cache_config.kv_cache_groups)
            if type(g.kv_cache_spec).__name__ == "MambaSpec"
        ]
        self.requests: dict[str, object] = {}
        self.max_tokens: dict[str, int] = {}
        self.workers: dict[str, _Worker] = {}
        self.state: dict[tuple[int, int], tuple[int, int]] = {}
        self.ident_by_hash: dict[bytes, tuple[int, int]] = {}
        self._hash_seen: dict[str, int] = defaultdict(int)
        self.chunks: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.hits: dict[str, int] = {}
        self.all_hits: dict[str, list[int]] = defaultdict(list)
        self.finished: set[str] = set()
        self.pending: deque = deque()
        self.steps = 0
        self.step_log: list[dict] = []
        self.verify_rows = 0
        self.multiblock_chunks = 0
        self.spec_prefill_multiblock_chunks = 0

    # -- properties ------------------------------------------------------------
    @property
    def policy(self) -> tuple:
        s = self.sched
        return (
            s._sx_align_multiblock,
            s._sx_align_lane,
            s._sx_align_tail,
            s._sx_align_tail_min,
            s._sx_align_max_blocks,
            s._sx_align_shared,
            s._sx_prefill_cap,
        )

    # -- requests ------------------------------------------------------------------
    def add_request(self, request_id: str, prompt: list[int], max_tokens: int = 4):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=100)
        req = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt),
            sampling_params=params,
            pooling_params=None,
            block_hasher=self._kvu.get_request_block_hasher(
                self.hash_B, self._sha256
            ),
        )
        self.requests[request_id] = req
        self.max_tokens[request_id] = max_tokens
        self._register_hashes(req)
        self.sched.add_request(req)
        return req

    def busy(self) -> bool:
        return bool(self.sched.waiting or self.sched.skipped_waiting
                    or self.sched.running or self.pending)

    def run_until_idle(self, max_steps: int = 200_000) -> None:
        for _ in range(max_steps):
            if not self.busy():
                self._check_outputs()
                return
            self.step()
        raise AssertionError("harness did not drain")

    def prefill_chunk_ends(self, request_id: str) -> list[int]:
        return [end for _, end in self.chunks[request_id]]

    # -- one engine step ---------------------------------------------------------
    def step(self) -> dict:
        from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

        self.steps += 1
        out = self.sched.schedule()
        info = self._on_schedule(out)
        sampled = self._execute(out)
        req_ids = list(out.num_scheduled_tokens)
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[sampled.get(rid, []) for rid in req_ids],
        )
        if self.async_sched:
            # Async scheduling: the output of step N reaches the scheduler
            # after step N+1 was scheduled (the worker already ran N).
            self.pending.append((out, mro))
            while len(self.pending) > 1 or (
                self.pending and not out.num_scheduled_tokens
            ):
                prev_out, prev_mro = self.pending.popleft()
                self.sched.update_from_output(prev_out, prev_mro)
        else:
            self.sched.update_from_output(out, mro)
            if self.k:
                draft_ids = [
                    rid for rid in req_ids
                    if sampled.get(rid) and rid in self.workers
                    and not self.requests[rid].is_finished()
                ]
                if draft_ids:
                    self.sched.update_draft_token_ids(
                        DraftTokenIds(
                            req_ids=draft_ids,
                            draft_token_ids=[
                                list(self.workers[rid].drafts) for rid in draft_ids
                            ],
                        )
                    )
        for rid, req in self.requests.items():
            if rid not in self.finished:
                self._register_hashes(req)
                if req.is_finished():
                    self.finished.add(rid)
        self._check_cache()
        self.step_log.append(info)
        return info

    # -- schedule-time bookkeeping ----------------------------------------------------
    def _on_schedule(self, out) -> dict:
        for rid in list(out.finished_req_ids) + list(out.preempted_req_ids):
            self.workers.pop(rid, None)
        for nrd in out.scheduled_new_reqs:
            req = self.requests[nrd.req_id]
            self.workers[nrd.req_id] = _Worker(
                list(req._all_token_ids), nrd.num_computed_tokens
            )
            self.hits.setdefault(nrd.req_id, nrd.num_computed_tokens)
            self.all_hits[nrd.req_id].append(nrd.num_computed_tokens)
        info = {"prefill": {}, "decode": {}, "spec": {}}
        for rid, n in out.num_scheduled_tokens.items():
            w = self.workers[rid]
            m = len(out.scheduled_spec_decode_tokens.get(rid, ()))
            prefill_end = max(self.requests[rid].num_prompt_tokens,
                              len(w.tokens) - 1)
            if w.computed < prefill_end:
                self.chunks[rid].append((w.computed, w.computed + n))
                info["prefill"][rid] = (w.computed, w.computed + n)
                if (w.computed + n - 1) // self.B > w.computed // self.B:
                    self.multiblock_chunks += 1
                    if self.k:
                        self.spec_prefill_multiblock_chunks += 1
            else:
                info["decode"][rid] = n
                info["spec"][rid] = m
        return info

    # -- worker ----------------------------------------------------------------------
    @staticmethod
    def ident(tokens, n: int) -> tuple[int, int]:
        return (n, hash(tuple(tokens[:n])))

    def _blocks(self, rid: str, gid: int):
        return self.manager.coordinator.single_type_managers[gid].req_to_blocks[rid]

    def _live_block(self, rid: str, gid: int, col: int, what: str):
        blocks = self._blocks(rid, gid)
        if col < 0 or col >= len(blocks) or blocks[col].is_null:
            raise OracleError(
                f"{rid}: {what} column {col} is null/missing (group {gid}, "
                f"{len(blocks)} columns)"
            )
        blk = blocks[col]
        if blk.ref_cnt <= 0:
            raise OracleError(
                f"{rid}: {what} column {col} holds freed block {blk.block_id}"
            )
        return (gid, blk.block_id)

    def _execute(self, out) -> dict[str, list[int]]:
        B, k = self.B, self.k
        rows = []
        # Pre-process + pre-copy for the whole batch.
        for rid, n in out.num_scheduled_tokens.items():
            w = self.workers[rid]
            req = self.requests[rid]
            m = len(out.scheduled_spec_decode_tokens.get(rid, ()))
            prefill_end = max(req.num_prompt_tokens, len(w.tokens) - 1)
            is_prefill = w.computed < prefill_end
            if not is_prefill and n != 1 + m:
                raise OracleError(f"{rid}: decode row with {n} tokens, {m} drafts")
            if is_prefill and m:
                raise OracleError(f"{rid}: prefill row carries {m} drafts")
            if m and not self.async_sched:
                sched_drafts = list(out.scheduled_spec_decode_tokens[rid])
                if sched_drafts != w.drafts[:m]:
                    raise OracleError(f"{rid}: scheduled drafts {sched_drafts} "
                                      f"!= worker drafts {w.drafts[:m]}")
            before = w.computed
            src = w.state_idx if w.state_idx != -2 else cdiv(before, B) - 1
            token_bias = max(w.num_accepted - 1, 0)
            dst = cdiv(before + n, B) - 1
            w.state_idx = dst
            for gid in self.mamba_group_ids:
                blocks = self._blocks(rid, gid)
                # 1 + k gathered state columns must be real blocks.
                for col in range(dst, dst + 1 + k):
                    self._live_block(rid, gid, col, "gathered state")
                if is_prefill and before % B == 0:
                    for col in range(cdiv(before, B), dst):
                        if not blocks[col].is_null:
                            raise OracleError(
                                f"{rid}: interior column {col} of chunk "
                                f"[{before},{before + n}) is a real block"
                            )
            if src >= 0 and src != dst:
                w.num_accepted = 1
                for gid in self.mamba_group_ids:
                    s = self._live_block(rid, gid, src + token_bias, "pre-copy source")
                    d = self._live_block(rid, gid, dst, "pre-copy destination")
                    self.state[d] = self.state.get(s)
            rows.append((rid, w, n, m, is_prefill, before))

        # Forward.
        sampled: dict[str, list[int]] = {}
        for rid, w, n, m, is_prefill, before in rows:
            read_col = w.state_idx + w.num_accepted - 1
            want_in = self.ident(w.tokens, before) if before > 0 else None
            if is_prefill:
                if w.num_accepted != 1:
                    raise OracleError(f"{rid}: prefill row with num_accepted "
                                      f"{w.num_accepted}")
                seq = w.tokens
                writes = [(read_col, before + n)]
            else:
                self.verify_rows += 1
                seq = w.tokens[: before + 1] + w.drafts[:m]
                writes = [(w.state_idx + j, before + 1 + j) for j in range(1 + m)]
            for gid in self.mamba_group_ids:
                key = self._live_block(rid, gid, read_col, "forward input")
                if want_in is not None and self.state.get(key) != want_in:
                    raise OracleError(
                        f"{rid}: row [{before},{before + n}) starts from state "
                        f"{self.state.get(key)!r}, expected state after {before} "
                        f"tokens (group {gid}, column {read_col})"
                    )
            for col, count in writes:
                ident = self.ident(seq, count)
                for gid in self.mamba_group_ids:
                    key = self._live_block(rid, gid, col, "forward output")
                    self.state[key] = ident
            # Sampling / rejection.
            if is_prefill:
                end = before + n
                if end >= len(w.tokens):
                    new_tokens = [true_token(rid, len(w.tokens))]
                else:
                    new_tokens = []
                w.computed = end
            else:
                accepted = 0
                for j, d in enumerate(w.drafts[:m]):
                    if d != true_token(rid, before + 1 + j):
                        break
                    accepted += 1
                a = accepted + 1
                new_tokens = [true_token(rid, before + 1 + j) for j in range(a)]
                w.computed = before + a
            w.tokens.extend(new_tokens)
            sampled[rid] = new_tokens
            if new_tokens and k:
                # MTP drafts for the next verify: the first (a - 1) are right.
                a_next = self.accept(self.rng, k)
                base = len(w.tokens)  # position of the first draft
                w.drafts = [
                    true_token(rid, base + j) if j < a_next - 1 else WRONG_DRAFT + j
                    for j in range(k)
                ]
            # Post-process.
            w.num_accepted = max(len(new_tokens), 1)
            new = w.computed
            running = new - w.num_accepted + 1
            aligned = new // self.B * self.B
            if aligned >= running:
                token_bias = aligned - running
                dst = aligned // self.B - 1
                src = w.state_idx
                if src == dst:
                    w.num_accepted = 1
                if self.worker_bug == "skip_postcopy" and src != dst:
                    continue
                if not (src == dst and token_bias == 0):
                    for gid in self.mamba_group_ids:
                        s = self._live_block(rid, gid, src + token_bias,
                                             "post-copy source")
                        d = self._live_block(rid, gid, dst, "post-copy destination")
                        self.state[d] = self.state.get(s)
        return sampled

    # -- oracle checks -------------------------------------------------------------
    def _register_hashes(self, req) -> None:
        scale = self.B // self.hash_B
        hashes = req.block_hashes
        tokens = req._all_token_ids
        n_blocks = len(hashes) // scale
        for i in range(self._hash_seen[req.request_id], n_blocks):
            key = hashes[i] if scale == 1 else b"".join(
                hashes[i * scale:(i + 1) * scale]
            )
            self.ident_by_hash[bytes(key)] = self.ident(tokens, (i + 1) * self.B)
        self._hash_seen[req.request_id] = n_blocks

    def _check_cache(self) -> None:
        get_group_id = self._kvu.get_group_id
        get_block_hash = self._kvu.get_block_hash
        mamba = set(self.mamba_group_ids)
        for blk in self.manager.block_pool.blocks:
            if blk.is_null or blk.block_hash is None:
                continue
            gid = get_group_id(blk.block_hash)
            if gid not in mamba:
                continue
            want = self.ident_by_hash.get(bytes(get_block_hash(blk.block_hash)))
            got = self.state.get((gid, blk.block_id))
            if want is None or got != want:
                raise OracleError(
                    f"cached Mamba block {blk.block_id} (group {gid}) holds "
                    f"{got!r} but its hash covers {want!r}: a state that was "
                    "never checkpointed is registered in the prefix cache"
                )

    def _check_outputs(self) -> None:
        for rid, req in self.requests.items():
            if not req.is_finished():
                raise OracleError(f"{rid} did not finish")
            got = list(req.output_token_ids)
            want = [
                true_token(rid, req.num_prompt_tokens + i)
                for i in range(self.max_tokens[rid])
            ]
            if got != want:
                raise OracleError(f"{rid}: output differs from the reference")
