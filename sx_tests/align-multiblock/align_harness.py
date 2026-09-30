# SPDX-License-Identifier: Apache-2.0
"""Scheduler harness with a recurrent-state oracle (CPU only).

Drives the real ``Scheduler`` / ``AsyncScheduler`` (``schedule()``,
``update_from_output()``, ``update_draft_token_ids()``) and the real
``KVCacheManager`` (hybrid coordinator, block pool, ``MambaManager`` align
allocation) the way ``EngineCore`` does, loaded through ``align_boot``. The
GPU worker is replaced by a model of what Model Runner V2 does with the
recurrent state in ``align`` mode:

* block tables: the worker keeps its own table per request. It is rewritten
  for a new request and otherwise only appended to with the block ids the
  scheduler sends (``model_runner.update_requests``), so columns the
  scheduler nulled stay stale there, as on the GPU.
* pre-process and pre-copy (``preprocess_mamba_align_fused_kernel``,
  ``_precopy_mamba_align_kernel``): source column = previous state column,
  seeded as ``cdiv(num_computed, B) - 1``; ``token_bias = num_accepted - 1``;
  destination = ``cdiv(num_computed + num_scheduled, B) - 1``. The state at
  ``src + token_bias`` is copied to ``dst`` when the column changes or on a
  single-token query after a speculative step, and ``num_accepted`` becomes 1.
* forward: a prefill or plain decode row reads and writes its state in place
  at ``dst + num_accepted - 1``; a verify row reads that column and writes the
  state after each of its ``1 + m`` tokens to ``dst + j``.
* post-process (``_postprocess_mamba_align_kernel``): ``num_accepted =
  max(num_sampled, 1)``; when the accepted tokens reach a block boundary the
  state at that boundary is copied into the boundary's column.

The oracle keeps, per (Mamba group, physical block), the identity of the
state it holds: the number of tokens and a hash of exactly those tokens. It
raises ``OracleError`` when

1. a forward starts from anything but the state after the request's own
   computed prefix;
2. the worker touches a block the request does not hold at that moment (null
   column, freed block, block handed to another request);
3. a column crossed by a multi-block prefill chunk is a real block, or one of
   the ``1 + k`` state columns of a row is not;
4. a Mamba block in the prefix cache holds anything but the state after
   exactly the tokens its hash covers, or is overwritten after that;
5. a finished request's output differs from the reference sequence;
6. (``check_pool=True``) a block's reference count differs from the number
   of block tables holding it, or the free queue disagrees with the counts.

Pipeline: ``queue_depth`` is one more than the number of scheduled steps
whose output has not come back when the next step is scheduled. 1 is the
synchronous scheduler; 2 is the default async engine
(``max_concurrent_batches``). The worker model runs a step only when its
output is due, that is after the following ``queue_depth - 1`` steps were
scheduled. That is the order which exposes a block freed or handed out while
an in-flight step still needs it. ``eager_worker=True`` runs it right after
scheduling instead; ``jitter=True`` lets outputs come back early at random.

Not a test file; see the ``test_*.py`` files next to it.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import random
import zlib
from collections import Counter, defaultdict, deque
from types import SimpleNamespace

import align_boot

SX_ENV_NAMES = (
    "SX_OPT_ALIGN_MULTIBLOCK",
    "SX_OPT_ALIGN_MULTIBLOCK_SPEC",
    "SX_OPT_ALIGN_MAX_CHUNK_BLOCKS",
)

# Text config of the SM70 Qwen3.8 decode compile contract.
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
_PREFIX_SEED = 0x5EED


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
    """Accepted length in [1, k + 1] with falling per-position rates."""
    rates = (0.95, 0.90, 0.86, 0.82, 0.80)
    accepted = 1
    for j in range(k):
        if rng.random() >= rates[min(j, len(rates) - 1)]:
            break
        accepted += 1
    return accepted


@contextlib.contextmanager
def sx_env(env: dict[str, str] | None):
    """Set exactly `env` for the SX switches (the others unset)."""
    saved = {name: os.environ.pop(name, None) for name in SX_ENV_NAMES}
    os.environ.update(env or {})
    try:
        yield
    finally:
        for name in SX_ENV_NAMES:
            os.environ.pop(name, None)
            if saved[name] is not None:
                os.environ[name] = saved[name]


@contextlib.contextmanager
def capture_log(universe: align_boot.Universe, module: str = align_boot.SCHEDULER):
    """Collect (level, message) of what one of the universe's modules logs."""
    messages: list[tuple[int, str]] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(  # type: ignore[method-assign]
        (record.levelno, record.getMessage())
    )
    logger = universe.logger(module)
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


class PrefixIds:
    """Identity of the recurrent state after every prefix of a sequence."""

    def __init__(self) -> None:
        self._ids = [_PREFIX_SEED]

    def sync(self, tokens: list[int]) -> PrefixIds:
        ids = self._ids
        for token in tokens[len(ids) - 1 :]:
            ids.append(hash((ids[-1], token)))
        return self

    def at(self, n: int) -> tuple[int, int]:
        return (n, self._ids[n])


def extend_ident(ident: tuple[int, int], tokens: list[int]) -> list[tuple[int, int]]:
    n, value = ident
    out = []
    for token in tokens:
        n, value = n + 1, hash((value, token))
        out.append((n, value))
    return out


class _FakeStructuredOutputManager:
    def should_advance(self, request, new_token_ids=None) -> bool:
        return False


class _FakeMMRegistry:
    def supports_multimodal_inputs(self, model_config) -> bool:
        return False


def make_spec_config(method: str, num_spec: int, *, native_mtp: bool = True,
                     parallel_drafting: bool | None = None):
    """The parts of ``SpeculativeConfig`` the scheduler reads."""
    dflash = method == "dflash"
    if parallel_drafting is None:
        # SpeculativeConfig turns it on for the whole DFlash family.
        parallel_drafting = method in ("dflash", "dflash_ddtree")
    eagle = method in ("mtp", "eagle", "eagle3", "dflash", "dflash_ddtree")
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_spec,
        num_speculative_state_tokens=lambda: num_spec,
        parallel_drafting=parallel_drafting,
        use_qwen4_exp_mtp=lambda: bool(native_mtp and method == "mtp"),
        use_eagle=lambda: eagle,
        use_eagle_kv_cache=lambda: eagle and not dflash,
        use_dflash=lambda: dflash,
        use_dflash_ddtree=lambda: method == "dflash_ddtree",
        use_dflash_family=lambda: method in ("dflash", "dflash_ddtree"),
        ddtree_disable_tree_verify=False,
        draft_model_config=SimpleNamespace(max_model_len=None),
    )


def make_vllm_config(*, torch, model: str, block_size: int, num_blocks: int,
                     budget: int, threshold: int, max_num_seqs: int,
                     max_model_len: int, spec, retention: int | None, tp: int,
                     runner_v2: bool, queue_depth: int, connector: bool,
                     cache_mode: str = "align"):
    """The parts of ``VllmConfig`` the scheduler reads.

    `model` picks the target: "qwen38" satisfies the SM70 Qwen3.8 decode
    compile contract, "qwen35" is a Qwen3.5-family hybrid (the 27B lane), any
    other value is used as the architecture name.
    """
    if model == "qwen38":
        architectures = ["Qwen4ExpForConditionalGeneration"]
        text = SimpleNamespace(**QWEN38_TEXT)
    elif model == "qwen35":
        architectures = ["Qwen3_5ForConditionalGeneration"]
        text = SimpleNamespace(
            hidden_size=5120, num_attention_heads=24, num_key_value_heads=4,
            head_dim=256,
        )
    else:
        architectures = [model]
        text = SimpleNamespace(**QWEN38_TEXT)
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    if spec is None:
        num_lookahead = 0
    elif spec.use_dflash_family():
        num_lookahead = spec.num_speculative_state_tokens() + 1
    else:
        num_lookahead = num_spec
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_scheduled_tokens=None,
            max_num_batched_tokens=budget,
            policy="fcfs",
            long_prefill_token_threshold=threshold,
            enable_chunked_prefill=True,
            scheduler_reserve_full_isl=False,
            async_scheduling=queue_depth > 1,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=num_blocks,
            enable_prefix_caching=True,
            mamba_cache_mode=cache_mode,
            block_size=block_size,
            prefix_cache_retention_interval=retention,
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
        model_config=SimpleNamespace(
            architectures=architectures,
            multimodal_config=SimpleNamespace(language_model_only=True),
            dtype=torch.float16,
            hf_text_config=text,
            is_encoder_decoder=False,
            is_hybrid=True,
            max_model_len=max_model_len,
            enable_return_routed_experts=False,
        ),
        kv_transfer_config=(
            SimpleNamespace(kv_load_failure_policy="recompute") if connector else None
        ),
        ec_transfer_config=None,
        speculative_config=spec,
        num_speculative_tokens=num_spec,
        num_lookahead_tokens=num_lookahead,
        max_in_flight_tokens=max(queue_depth, 1) * budget,
        use_v2_model_runner=runner_v2,
    )


def make_kv_cache_config(universe: align_boot.Universe, *, layout: str,
                         block_size: int, num_blocks: int, num_spec: int,
                         attn_block_size: int | None = None,
                         mtp_layer: bool = False):
    """KV-cache groups shaped like the two production layouts.

    "flashnext": QSA main and compressed caches, the QSA ring (not prefix
    cacheable), three GDN state groups and the PLE short-conv state group,
    all on the state block. "27b": full attention and two GDN state groups on
    the state block, plus the DFlash2 draft's sliding-window group on half of
    it (FP16 draft KV next to FP8 target KV).
    """
    import torch

    kvi = universe.kv_cache_interface
    attn_block = attn_block_size or block_size
    full = kvi.FullAttentionSpec(
        block_size=attn_block, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    gdn = kvi.MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec,
    )
    groups = [
        kvi.KVCacheGroupSpec(["attn.main"] + (["mtp.attn"] if mtp_layer else []), full)
    ]
    if layout == "flashnext":
        groups.append(
            kvi.KVCacheGroupSpec(
                ["qsa.compressed"],
                kvi.MLAAttentionSpec(
                    block_size=attn_block, num_kv_heads=1, head_size=1,
                    dtype=torch.float32, compress_ratio=4,
                ),
            )
        )
        groups.append(
            kvi.KVCacheGroupSpec(
                ["qsa.ring"],
                kvi.CircularBufferSpec(
                    block_size=4, num_kv_heads=1, head_size=1, head_size_v=0,
                    dtype=torch.float32,
                ),
            )
        )
        groups.extend(kvi.KVCacheGroupSpec([f"gdn{i}"], gdn) for i in range(3))
        groups.append(
            kvi.KVCacheGroupSpec(
                ["ple"],
                kvi.MambaSpec(
                    block_size=block_size,
                    shapes=((2,),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=num_spec,
                    tp_replicated=True,
                ),
            )
        )
    else:
        assert layout == "27b", layout
        assert block_size % 2 == 0
        groups.extend(kvi.KVCacheGroupSpec([f"gdn{i}"], gdn) for i in range(2))
        groups.append(
            kvi.KVCacheGroupSpec(
                ["draft.swa"],
                kvi.SlidingWindowSpec(
                    block_size=block_size // 2, num_kv_heads=1, head_size=2,
                    dtype=torch.float32, sliding_window=2048,
                ),
            )
        )
    return kvi.KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


class _Worker:
    """Worker-side view of one request (exact counts, own block tables)."""

    def __init__(self, tokens: list[int], num_prompt_tokens: int,
                 num_computed: int, block_ids) -> None:
        self.tokens = list(tokens)  # prompt + sampled outputs the worker knows
        self.ids = PrefixIds().sync(self.tokens)
        self.num_prompt_tokens = num_prompt_tokens
        self.computed = num_computed
        self.state_idx = -2  # seeded at the first pre-process
        self.num_accepted = 1
        self.drafts: list[int] = []
        self.block_ids = [list(ids) for ids in block_ids]


class SchedHarness:
    def __init__(
        self,
        *,
        universe: align_boot.Universe | None = None,
        layout: str = "flashnext",
        block_size: int = 784,
        hash_block_size: int = 16,
        attn_block_size: int | None = None,
        budget: int = 8192,
        threshold: int = 0,
        max_num_seqs: int = 24,
        num_blocks: int = 4096,
        max_model_len: int = 300_000,
        num_spec: int = 0,
        method: str | None = None,
        native_mtp: bool = True,
        parallel_drafting: bool | None = None,
        retention: int | None = 0,
        env: dict[str, str] | None = None,
        queue_depth: int = 1,
        async_sched: bool | None = None,
        eager_worker: bool = False,
        jitter: bool = False,
        sm70: bool = True,
        model: str | None = None,
        tp: int = 4,
        runner_v2: bool = True,
        connector: bool = False,
        cache_mode: str = "align",
        check_pool: bool = False,
        seed: int = 0,
        accept=None,
        worker_bug: str | None = None,
    ) -> None:
        import torch

        self.universe = universe or align_boot.load()
        self.B = block_size
        self.hash_B = hash_block_size
        assert self.B % self.hash_B == 0
        self.k = num_spec
        if num_spec and method is None:
            method = "dflash" if layout == "27b" else "mtp"
        self.method = method if num_spec else None
        self.queue_depth = queue_depth
        self.async_sched = queue_depth > 1 if async_sched is None else async_sched
        assert self.async_sched or queue_depth == 1
        self.eager_worker = eager_worker
        self.jitter = jitter
        self.rng = random.Random(seed)
        self.accept = accept or measured_acceptance
        # Negative controls only: "skip_postcopy" drops the copy that
        # checkpoints a boundary crossed by accepted verify tokens.
        self.worker_bug = worker_bug
        self.check_pool = check_pool

        spec = (
            make_spec_config(self.method, num_spec, native_mtp=native_mtp,
                             parallel_drafting=parallel_drafting)
            if num_spec
            else None
        )
        if model is None:
            model = "qwen35" if layout == "27b" else "qwen38"
        self.vllm_config = make_vllm_config(
            torch=torch, model=model, block_size=block_size, num_blocks=num_blocks,
            budget=budget, threshold=threshold, max_num_seqs=max_num_seqs,
            max_model_len=max_model_len, spec=spec, retention=retention, tp=tp,
            runner_v2=runner_v2, queue_depth=queue_depth, connector=connector,
            cache_mode=cache_mode,
        )
        self.kv_cache_config = make_kv_cache_config(
            self.universe, layout=layout, block_size=block_size,
            num_blocks=num_blocks, num_spec=num_spec,
            attn_block_size=attn_block_size, mtp_layer=self.method == "mtp",
        )
        cls = (
            self.universe.async_scheduler.AsyncScheduler
            if self.async_sched
            else self.universe.scheduler.Scheduler
        )
        self.universe.config_vllm.SM70 = sm70
        with self.universe.activate(), sx_env(env), capture_log(
            self.universe
        ) as self.init_log:
            self.sched = cls(
                vllm_config=self.vllm_config,
                kv_cache_config=self.kv_cache_config,
                structured_output_manager=_FakeStructuredOutputManager(),
                block_size=attn_block_size or block_size,
                hash_block_size=self.hash_B,
                mm_registry=_FakeMMRegistry(),
                include_finished_set=False,
                log_stats=False,
            )
        self.manager = self.sched.kv_cache_manager
        self.pool = self.manager.block_pool
        self.mamba_group_ids = [
            gid
            for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if type(group.kv_cache_spec).__name__ == "MambaSpec"
        ]
        self.requests: dict[str, object] = {}
        self.max_tokens: dict[str, int] = {}
        self.workers: dict[str, _Worker] = {}
        self._request_ids: dict[str, PrefixIds] = {}
        # (gid, block_id) -> identity of the state the block holds.
        self.state: dict[tuple[int, int], tuple[int, int]] = {}
        self.ident_by_hash: dict[bytes, tuple[int, int]] = {}
        self._hash_seen: dict[str, int] = defaultdict(int)
        # Mamba blocks in the prefix cache: (gid, block_id) -> (hash key, step
        # whose execution makes the content valid).
        self._cached: dict[tuple[int, int], tuple[bytes, int]] = {}
        self._unverified: set[tuple[int, int]] = set()
        self._epoch = 0  # step being scheduled, or whose output is applied
        self._executing = 0  # step the worker model ran last
        # Cached blocks that were reallocated; parity runs require none.
        self.evictions = 0
        self._hook_pool()
        self.chunks: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.hits: dict[str, int] = {}
        self.all_hits: dict[str, list[int]] = defaultdict(list)
        self.finished: set[str] = set()
        # Scheduled steps whose output has not been applied yet.
        self.pending: deque = deque()
        self.steps = 0
        self.step_log: list[dict] = []
        self.verify_rows = 0
        self.stale_rows = 0
        self.multiblock_chunks = 0
        # Most real (non-null) state columns a request held in one Mamba group.
        self.peak_state_columns: dict[str, int] = defaultdict(int)

    def _hook_pool(self) -> None:
        """Follow prefix-cache registrations and evictions of Mamba blocks."""
        kvu = self.universe.kv_cache_utils
        mamba = set(self.mamba_group_ids)
        pool = self.pool
        cache_full_blocks = pool.cache_full_blocks
        evict = pool._maybe_evict_cached_block

        def tracking_cache(**kwargs) -> None:
            cache_full_blocks(**kwargs)
            gid = kwargs["kv_cache_group_id"]
            if gid not in mamba:
                return
            first, last = kwargs["num_cached_blocks"], kwargs["num_full_blocks"]
            for blk in kwargs["blocks"][first:last]:
                key = blk.block_hash
                if blk.is_null or key is None:
                    continue
                slot = (gid, blk.block_id)
                if self._cached.get(slot, (None,))[0] != key:
                    self._cached[slot] = (key, self._epoch)
                    self._unverified.add(slot)

        def tracking_evict(block) -> bool:
            key = block.block_hash
            evicted = evict(block)
            if evicted:
                self.evictions += 1
                slot = (kvu.get_group_id(key), block.block_id)
                self._cached.pop(slot, None)
                self._unverified.discard(slot)
            return evicted

        pool.cache_full_blocks = tracking_cache
        pool._maybe_evict_cached_block = tracking_evict

    # -- properties ----------------------------------------------------------
    @property
    def multiblock(self) -> bool:
        return bool(getattr(self.sched, "_sx_align_multiblock", False))

    def messages(self, level: int = logging.INFO) -> list[str]:
        """What the scheduler logged while it was constructed."""
        return [message for lvl, message in self.init_log if lvl >= level]

    # -- requests ------------------------------------------------------------
    def add_request(self, request_id: str, prompt: list[int], max_tokens: int = 4):
        kvu = self.universe.kv_cache_utils
        request = self.universe.request.Request(
            request_id=request_id,
            prompt_token_ids=list(prompt),
            sampling_params=align_boot.SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            block_hasher=kvu.get_request_block_hasher(self.hash_B, align_boot.sha256),
        )
        self.requests[request_id] = request
        self.max_tokens[request_id] = max_tokens
        self._request_ids[request_id] = PrefixIds()
        self._register_hashes(request)
        self.sched.add_request(request)
        return request

    def abort(self, request_id: str) -> None:
        status = self.universe.request.RequestStatus.FINISHED_ABORTED
        self.sched.finish_requests(request_id, status)
        self.finished.add(request_id)

    def busy(self) -> bool:
        sched = self.sched
        return bool(
            sched.waiting or sched.skipped_waiting or sched.running or self.pending
        )

    def run_until_idle(self, max_steps: int = 400_000) -> None:
        for _ in range(max_steps):
            if not self.busy():
                self._check_cache()
                self._check_outputs()
                return
            self.step()
        raise AssertionError("harness did not drain")

    def prefill_chunk_ends(self, request_id: str) -> list[int]:
        return [end for _, end in self.chunks[request_id]]

    def prefill_steps(self, request_id: str | None = None) -> int:
        if request_id is not None:
            return len(self.chunks[request_id])
        return sum(len(chunks) for chunks in self.chunks.values())

    def state_columns(self, request_id: str) -> list[int]:
        """Real state columns the request holds, per Mamba group."""
        managers = self.manager.coordinator.single_type_managers
        return [
            sum(
                not blk.is_null
                for blk in managers[gid].req_to_blocks.get(request_id, ())
            )
            for gid in self.mamba_group_ids
        ]

    def cached_state_boundaries(self) -> set[tuple[int, tuple[int, int]]]:
        """(group, state identity) of every Mamba block in the prefix cache."""
        kvu = self.universe.kv_cache_utils
        mamba = set(self.mamba_group_ids)
        out = set()
        for blk in self.pool.blocks:
            if blk.is_null or blk.block_hash is None:
                continue
            gid = kvu.get_group_id(blk.block_hash)
            if gid in mamba:
                key = bytes(kvu.get_block_hash(blk.block_hash))
                out.add((gid, self.ident_by_hash[key]))
        return out

    # -- one engine step -----------------------------------------------------
    def step(self) -> dict:
        self.steps += 1
        self._epoch = self.steps
        out = self.sched.schedule()
        info = self._on_schedule(out)
        entry = SimpleNamespace(out=out, index=self.steps, info=info, sampled=None)
        if self.eager_worker:
            entry.sampled = self._execute(entry)
        self.pending.append(entry)
        keep = self.queue_depth - 1
        if self.jitter and keep:
            keep = self.rng.randrange(keep + 1)
        if not out.num_scheduled_tokens:
            # Nothing new can run before an output comes back.
            keep = 0
        while len(self.pending) > keep:
            self._complete(self.pending.popleft())
        if self.check_pool:
            self.assert_pool_consistent()
        self.step_log.append(info)
        return info

    def drain(self) -> None:
        """Apply the output of every step that is still in flight."""
        while self.pending:
            self._complete(self.pending.popleft())
        if self.check_pool:
            self.assert_pool_consistent()

    def _complete(self, entry) -> None:
        outputs = self.universe.outputs
        if entry.sampled is None:
            entry.sampled = self._execute(entry)
        sampled = entry.sampled
        req_ids = list(entry.out.num_scheduled_tokens)
        # Blocks cached while this output is applied describe tokens the
        # worker has computed already.
        self._epoch = entry.index
        self.sched.update_from_output(
            entry.out,
            outputs.ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=[list(sampled.get(rid, ())) for rid in req_ids],
            ),
        )
        if self.k and not self.async_sched:
            # Async scheduling keeps the drafts on the worker (placeholders in
            # the scheduler); the synchronous scheduler is handed them.
            draft_ids = [
                rid
                for rid in req_ids
                if sampled.get(rid)
                and rid in self.workers
                and not self.requests[rid].is_finished()
            ]
            if draft_ids:
                self.sched.update_draft_token_ids(
                    outputs.DraftTokenIds(
                        req_ids=draft_ids,
                        draft_token_ids=[
                            list(self.workers[rid].drafts) for rid in draft_ids
                        ],
                    )
                )
        for rid, request in self.requests.items():
            if rid not in self.finished:
                self._register_hashes(request)
                if request.is_finished():
                    self.finished.add(rid)
        self._verify_cached()

    # -- schedule-time bookkeeping -------------------------------------------
    def _on_schedule(self, out) -> dict:
        info = {"prefill": {}, "decode": {}, "spec": {}, "incarnation": {}}
        for new_req in out.scheduled_new_reqs:
            rid = new_req.req_id
            self.hits.setdefault(rid, new_req.num_computed_tokens)
            self.all_hits[rid].append(new_req.num_computed_tokens)
        managers = self.manager.coordinator.single_type_managers
        for rid, n in out.num_scheduled_tokens.items():
            request = self.requests[rid]
            # `_update_after_schedule` has already advanced the request.
            start = request.num_computed_tokens - n
            prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
            info["incarnation"][rid] = request.num_preemptions
            if start < prefill_end:
                self.chunks[rid].append((start, start + n))
                info["prefill"][rid] = (start, start + n)
                if (start + n - 1) // self.B > start // self.B:
                    self.multiblock_chunks += 1
            else:
                info["decode"][rid] = n
                info["spec"][rid] = len(out.scheduled_spec_decode_tokens.get(rid, ()))
            for gid in self.mamba_group_ids:
                real = sum(not blk.is_null for blk in managers[gid].req_to_blocks[rid])
                if real > self.peak_state_columns[rid]:
                    self.peak_state_columns[rid] = real
        # What the worker is sent, captured now like the pickled output.
        info["new"] = {
            new_req.req_id: (
                list(new_req.prefill_token_ids),
                self.requests[new_req.req_id].num_prompt_tokens,
                new_req.num_computed_tokens,
                [list(ids) for ids in new_req.block_ids],
            )
            for new_req in out.scheduled_new_reqs
        }
        return info

    def assert_pool_consistent(self) -> None:
        """Reference counts match the block tables; free means unreferenced.

        A double free shows as a block with fewer references than holders, a
        leaked reference as one with more.
        """
        holders: Counter = Counter()
        for manager in self.manager.coordinator.single_type_managers:
            for blocks in manager.req_to_blocks.values():
                holders.update(blk.block_id for blk in blocks if not blk.is_null)
        free_blocks = self.pool.free_block_queue.get_all_free_blocks()
        free = [blk.block_id for blk in free_blocks]
        if len(free) != len(set(free)) or len(free) != self.pool.get_num_free_blocks():
            raise OracleError("free queue holds a block twice or miscounts")
        unreferenced = set()
        for blk in self.pool.blocks:
            if blk.is_null:
                continue
            if blk.ref_cnt != holders[blk.block_id]:
                raise OracleError(
                    f"block {blk.block_id} has ref_cnt {blk.ref_cnt} but "
                    f"{holders[blk.block_id]} holder(s)"
                )
            if blk.ref_cnt == 0:
                unreferenced.add(blk.block_id)
        if unreferenced != set(free):
            raise OracleError("free queue and unreferenced blocks differ")

    def snapshot(self) -> tuple:
        """Block tables, free-queue order and prefix cache, for comparisons."""
        tables = tuple(
            tuple(
                (rid, tuple(blk.block_id for blk in blocks))
                for rid, blocks in sorted(manager.req_to_blocks.items())
                if blocks
            )
            for manager in self.manager.coordinator.single_type_managers
        )
        free = tuple(
            blk.block_id for blk in self.pool.free_block_queue.get_all_free_blocks()
        )
        cached = tuple(
            (blk.block_id, blk.block_hash, blk.ref_cnt)
            for blk in self.pool.blocks
            if blk.block_hash is not None
        )
        return tables, free, cached

    # -- worker --------------------------------------------------------------
    def _held(self, rid: str, gid: int) -> set[int]:
        manager = self.manager.coordinator.single_type_managers[gid]
        return {
            blk.block_id
            for blk in manager.req_to_blocks.get(rid, ())
            if not blk.is_null and blk.ref_cnt > 0
        }

    def _slot(self, w: _Worker, rid: str, gid: int, col: int, what: str, held):
        """State slot of column `col` of the worker's table.

        `held` is what the request holds right now per group, or None for a
        row whose request is gone (see `_execute`).
        """
        ids = w.block_ids[gid]
        block_id = ids[col] if 0 <= col < len(ids) else 0
        if held is not None and block_id not in held[gid]:
            raise OracleError(
                f"{rid}: {what} column {col} is block {block_id}, which the "
                f"request does not hold (group {gid}, {len(ids)} columns)"
            )
        return (gid, block_id)

    def _store(self, slot: tuple[int, int], ident) -> None:
        cached = self._cached.get(slot)
        if cached is not None and cached[1] < self._executing:
            want = self._ident_of(cached[0])
            if ident != want:
                raise OracleError(
                    f"cached Mamba block {slot[1]} (group {slot[0]}) holding "
                    f"{want!r} is overwritten with {ident!r}"
                )
        self.state[slot] = ident

    def _execute(self, entry) -> dict[str, list[int]]:
        out, info = entry.out, entry.info
        B, k = self.B, self.k
        self._executing = entry.index
        for rid in list(out.finished_req_ids) + list(out.preempted_req_ids or ()):
            self.workers.pop(rid, None)
        for rid, (tokens, num_prompt, computed, block_ids) in info["new"].items():
            self.workers[rid] = _Worker(tokens, num_prompt, computed, block_ids)
        cached = out.scheduled_cached_reqs
        for rid, new_block_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_block_ids is not None:
                for ids, new_ids in zip(self.workers[rid].block_ids, new_block_ids):
                    ids.extend(new_ids)

        rows = []
        # Pre-process and pre-copy for the whole batch.
        for rid, n in out.num_scheduled_tokens.items():
            w = self.workers[rid]
            request = self.requests[rid]
            # A row whose request was preempted, finished or aborted after it
            # was scheduled still runs on the GPU. Its blocks may belong to
            # another request by now, whose first use of them comes later.
            stale = (
                request.is_finished()
                or request.num_preemptions != info["incarnation"][rid]
            )
            held = (
                None
                if stale
                else {gid: self._held(rid, gid) for gid in self.mamba_group_ids}
            )
            self.stale_rows += stale
            m = len(out.scheduled_spec_decode_tokens.get(rid, ()))
            prefill_end = max(w.num_prompt_tokens, len(w.tokens) - 1)
            is_prefill = w.computed < prefill_end
            if not is_prefill and n != 1 + m:
                raise OracleError(f"{rid}: decode row with {n} tokens, {m} drafts")
            if is_prefill and m:
                raise OracleError(f"{rid}: prefill row carries {m} drafts")
            if m and not self.async_sched:
                scheduled = list(out.scheduled_spec_decode_tokens[rid])
                if scheduled != w.drafts[:m]:
                    raise OracleError(
                        f"{rid}: scheduled drafts {scheduled} != worker drafts "
                        f"{w.drafts[:m]}"
                    )
            before = w.computed
            src = w.state_idx if w.state_idx != -2 else cdiv(before, B) - 1
            token_bias = max(w.num_accepted - 1, 0)
            dst = cdiv(before + n, B) - 1
            w.state_idx = dst
            for gid in self.mamba_group_ids:
                # The 1 + k gathered state columns must be real blocks.
                for col in range(dst, dst + 1 + k):
                    self._slot(w, rid, gid, col, "gathered state", held)
                if held is not None and is_prefill and before % B == 0:
                    blocks = self.manager.coordinator.single_type_managers[
                        gid
                    ].req_to_blocks[rid]
                    for col in range(cdiv(before, B), dst):
                        if not blocks[col].is_null:
                            raise OracleError(
                                f"{rid}: interior column {col} of chunk "
                                f"[{before},{before + n}) is a real block"
                            )
            single_token_tail = n == 1 and token_bias > 0
            if src >= 0 and (src != dst or single_token_tail):
                w.num_accepted = 1
                for gid in self.mamba_group_ids:
                    s = self._slot(w, rid, gid, src + token_bias, "pre-copy source",
                                   held)
                    d = self._slot(w, rid, gid, dst, "pre-copy destination", held)
                    self._store(d, self.state.get(s))
            rows.append((rid, w, n, m, is_prefill, before, held))

        # Forward, sampling and post-process.
        sampled: dict[str, list[int]] = {}
        for rid, w, n, m, is_prefill, before, held in rows:
            read_col = w.state_idx + w.num_accepted - 1
            want_in = w.ids.at(before) if before > 0 else None
            if is_prefill:
                if w.num_accepted != 1:
                    raise OracleError(
                        f"{rid}: prefill row with num_accepted {w.num_accepted}"
                    )
                writes = [(read_col, w.ids.at(before + n))]
            else:
                self.verify_rows += m > 0
                first = w.ids.at(before + 1)
                idents = [first] + extend_ident(first, w.drafts[:m])
                writes = [(w.state_idx + j, ident) for j, ident in enumerate(idents)]
            for gid in self.mamba_group_ids:
                key = self._slot(w, rid, gid, read_col, "forward input", held)
                if (
                    held is not None
                    and want_in is not None
                    and self.state.get(key) != want_in
                ):
                    raise OracleError(
                        f"{rid}: row [{before},{before + n}) starts from state "
                        f"{self.state.get(key)!r}, expected the state after "
                        f"{before} tokens (group {gid}, column {read_col})"
                    )
            for col, ident in writes:
                for gid in self.mamba_group_ids:
                    self._store(
                        self._slot(w, rid, gid, col, "forward output", held), ident
                    )
            # Sampling and rejection.
            if is_prefill:
                end = before + n
                new_tokens = (
                    [true_token(rid, len(w.tokens))] if end >= len(w.tokens) else []
                )
                w.computed = end
            else:
                accepted = 0
                for j, draft in enumerate(w.drafts[:m]):
                    if draft != true_token(rid, before + 1 + j):
                        break
                    accepted += 1
                new_tokens = [
                    true_token(rid, before + 1 + j) for j in range(accepted + 1)
                ]
                w.computed = before + accepted + 1
            w.tokens.extend(new_tokens)
            w.ids.sync(w.tokens)
            sampled[rid] = new_tokens
            if new_tokens and k:
                # Drafts for the next verify: the first a - 1 are right.
                a_next = self.accept(self.rng, k)
                base = len(w.tokens)  # position of the first draft
                w.drafts = [
                    true_token(rid, base + j) if j < a_next - 1 else WRONG_DRAFT + j
                    for j in range(k)
                ]
            # Post-process.
            w.num_accepted = max(len(new_tokens), 1)
            running = w.computed - w.num_accepted + 1
            aligned = w.computed // B * B
            if aligned >= running:
                token_bias = aligned - running
                dst = aligned // B - 1
                src = w.state_idx
                if src == dst:
                    w.num_accepted = 1
                if self.worker_bug == "skip_postcopy" and src != dst:
                    continue
                if not (src == dst and token_bias == 0):
                    for gid in self.mamba_group_ids:
                        s = self._slot(w, rid, gid, src + token_bias,
                                       "post-copy source", held)
                        d = self._slot(w, rid, gid, dst, "post-copy destination",
                                       held)
                        self._store(d, self.state.get(s))
        self._verify_cached()
        return sampled

    # -- oracle checks -------------------------------------------------------
    def _register_hashes(self, request) -> None:
        scale = self.B // self.hash_B
        hashes = request.block_hashes
        num_blocks = len(hashes) // scale
        rid = request.request_id
        if num_blocks <= self._hash_seen[rid]:
            return
        ids = self._request_ids[rid].sync(request._all_token_ids)
        for i in range(self._hash_seen[rid], num_blocks):
            key = b"".join(hashes[i * scale : (i + 1) * scale])
            self.ident_by_hash[bytes(key)] = ids.at((i + 1) * self.B)
        self._hash_seen[rid] = num_blocks

    def _ident_of(self, key: bytes):
        kvu = self.universe.kv_cache_utils
        return self.ident_by_hash.get(bytes(kvu.get_block_hash(key)))

    def _assert_holds_its_state(self, gid: int, block_id: int, key: bytes) -> None:
        want = self._ident_of(key)
        got = self.state.get((gid, block_id))
        if want is None or got != want:
            raise OracleError(
                f"cached Mamba block {block_id} (group {gid}) holds {got!r} but "
                f"its hash covers {want!r}: a state that was never materialized "
                "is registered in the prefix cache"
            )

    def _verify_cached(self) -> None:
        """Check the registrations whose step the worker has run by now."""
        due = [s for s in self._unverified if self._cached[s][1] <= self._executing]
        for slot in due:
            self._unverified.discard(slot)
            self._assert_holds_its_state(*slot, self._cached[slot][0])

    def _check_cache(self) -> None:
        """Every Mamba block in the prefix cache holds the state of its hash."""
        kvu = self.universe.kv_cache_utils
        mamba = set(self.mamba_group_ids)
        seen = set()
        for blk in self.pool.blocks:
            key = blk.block_hash
            if blk.is_null or key is None:
                continue
            gid = kvu.get_group_id(key)
            if gid in mamba:
                seen.add((gid, blk.block_id))
                self._assert_holds_its_state(gid, blk.block_id, key)
        if seen != set(self._cached) or self._unverified:
            raise OracleError("the harness lost track of the prefix cache")

    def _check_outputs(self) -> None:
        aborted = self.universe.request.RequestStatus.FINISHED_ABORTED
        for rid, request in self.requests.items():
            if not request.is_finished():
                raise OracleError(f"{rid} did not finish")
            if request.status == aborted:
                continue
            got = list(request.output_token_ids)
            want = [
                true_token(rid, request.num_prompt_tokens + i)
                for i in range(self.max_tokens[rid])
            ]
            if got != want:
                raise OracleError(f"{rid}: output differs from the reference")


def split_shim(universe: align_boot.Universe, block_size: int = 784, *,
               eagle: bool = False, multiblock: bool = True, retention: int = 0,
               max_blocks: int = 0, alignment: int | None = None):
    """The scheduler attributes ``_mamba_block_aligned_split`` reads.

    The real method is called unbound on this object, as upstream's
    ``test_mamba_align_chunk_split.py`` does. Replay boundaries come from the
    real ``KVCacheCoordinator.get_replay_boundaries``.
    """
    coordinator_cls = universe.kv_cache_coordinator.KVCacheCoordinator
    coordinator = SimpleNamespace(eagle_group_ids={0} if eagle else set())
    coordinator.get_replay_boundaries = functools.partial(
        coordinator_cls.get_replay_boundaries, coordinator
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        mamba_state_block_size=block_size,
        use_eagle=eagle,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
        _sx_align_multiblock=multiblock,
        _sx_align_hit_alignment=alignment or block_size,
        _sx_align_retention=retention,
        _sx_align_max_blocks=max_blocks,
    )


def chunk_ends(split, shim, prompt_len: int, *, start: int = 0, budget: int = 8192,
               threshold: int = 0, shared_prefix_boundary: int = 0) -> list[int]:
    """Chunk ends of a lone request, as the scheduler's running loop asks."""
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        shared_prefix_boundary=shared_prefix_boundary,
    )
    ends = []
    while request.num_computed_tokens < prompt_len:
        num_new_tokens = prompt_len - request.num_computed_tokens
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        num_new_tokens = split(shim, request, min(num_new_tokens, budget))
        assert num_new_tokens > 0
        request.num_computed_tokens += num_new_tokens
        ends.append(request.num_computed_tokens)
    return ends


def run_sequential(h: SchedHarness, seed: int, num_requests: int, max_len: int,
                   max_output: int = 40) -> SchedHarness:
    """Requests one after the other: fresh prompts, identical resends, next
    turns of finished conversations and siblings sharing a prefix.

    The workload depends only on `seed` and on the outputs of earlier
    requests, which are a function of request id and position, so two
    harnesses given the same arguments see the same requests.
    """
    rng = random.Random(seed)
    block = h.B
    prompts: list[list[int]] = []
    conversations: list[list[int]] = []

    def fresh(n: int) -> list[int]:
        return [rng.randrange(10, 30_000) for _ in range(n)]

    for i in range(num_requests):
        kinds = ["fresh", "resend", "turn", "sibling"] if prompts else ["fresh"]
        kind = rng.choice(kinds)
        if kind == "fresh":
            n = rng.choice(
                [
                    rng.randrange(1, 3 * block),
                    rng.randrange(1, max_len),
                    rng.randrange(1, max_len) // block * block
                    + rng.choice([0, 1, block - 1]),
                ]
            )
            prompt = fresh(max(1, n))
        elif kind == "resend":
            prompt = list(rng.choice(prompts))
        elif kind == "turn":
            base = rng.choice(conversations)
            prompt = list(base) + fresh(rng.randrange(1, 3 * block))
        else:
            base = rng.choice(prompts)
            prompt = base[: rng.randrange(1, len(base) + 1)] + fresh(
                rng.randrange(1, 4 * block)
            )
        prompt = prompt[:max_len]
        rid = f"r{i}"
        h.add_request(rid, prompt, max_tokens=rng.choice([1, 3, 5, max_output]))
        h.run_until_idle()
        prompts.append(prompt)
        conversations.append(list(h.requests[rid]._all_token_ids))
    return h


def concurrent_workload(seed: int, block: int, *, max_prompt_blocks: float = 9.0,
                        max_output: int = 300):
    """Overlapping requests: shared prefixes, resends and siblings.

    Yields ("add", request id, prompt, max tokens) and ("step",) actions, so
    that several harnesses can be driven through the same workload.
    """
    rng = random.Random(seed)
    bases = [make_prompt(1000 + j, int(12 * block)) for j in range(3)]
    sent: list[list[int]] = []
    count = 0

    def add(prompt: list[int]):
        nonlocal count
        sent.append(prompt)
        count += 1
        return ("add", f"q{count - 1}", prompt, rng.randrange(1, max_output))

    for _ in range(6):
        for _ in range(rng.randrange(1, 5)):
            share = rng.choice([0, block // 2, block, 2 * block, int(6.4 * block)])
            tail = make_prompt(
                rng.randrange(1 << 30), rng.randrange(1, int(max_prompt_blocks * block))
            )
            yield add(rng.choice(bases)[:share] + tail)
            roll = rng.random()
            if roll < 0.25:
                yield add(list(rng.choice(sent)))
            elif roll < 0.45:
                share = rng.choice([block, 2 * block, 2 * block + block // 20])
                yield add(
                    rng.choice(bases)[:share]
                    + make_prompt(rng.randrange(1 << 30), rng.randrange(1, block // 2))
                )
        for _ in range(rng.randrange(1, 25)):
            yield ("step",)


def run_concurrent(h: SchedHarness, seed: int, **kwargs) -> SchedHarness:
    """Drive `h` through `concurrent_workload` and drain it."""
    count = 0
    for action in concurrent_workload(seed, h.B, **kwargs):
        if action[0] == "add":
            h.add_request(*action[1:])
            count += 1
        elif h.busy():
            h.step()
    h.run_until_idle()
    assert len(h.finished) == count
    return h
