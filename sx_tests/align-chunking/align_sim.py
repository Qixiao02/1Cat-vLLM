# SPDX-License-Identifier: Apache-2.0
"""Scheduler-level simulation of Mamba ``align`` prefix caching (CPU only).

Drives the REAL ``KVCacheManager`` (hybrid coordinator, BlockPool,
MambaManager/FullAttentionManager/CircularBufferManager) and the REAL
``Scheduler._mamba_block_aligned_split`` / shared-prefix helpers through a
small scheduling loop that mirrors ``Scheduler.schedule()`` (running requests
first, then waiting requests with a prefix lookup, one token budget), plus a
recurrent-state oracle that mirrors the MRV2 worker:

* ``preprocess_mamba_align_fused_kernel``: src column = previous state column
  (seeded as ``cdiv(num_computed, B) - 1`` for a new request), dst column =
  ``(num_computed + num_scheduled - 1) // B``; pre-copy src -> dst when they
  differ.
* the forward reads the initial recurrent state from the dst column when
  ``num_computed > 0`` and writes the state after the chunk to the dst column.
* post-process is a no-op without speculative decoding.

The oracle stores, per (mamba group, physical block), the identity of the
recurrent state it holds: ``(num_tokens, hash(prefix tokens))``. It raises
``AlignOracleError`` when

1. a forward starts from a state that is not exactly the state after the
   request's own ``num_computed`` prefix (i.e. a prefix hit or a pre-copy
   served a wrong/stale state), or
2. a block registered in the prefix cache for a Mamba group holds anything
   other than the state after exactly the tokens its hash covers (i.e. a
   block was registered whose state was never checkpointed).

The backend object supplies construction of specs/managers/requests so the
same harness runs against vLLM in the deployed image (``RealBackend``).
"""

from __future__ import annotations

import zlib
from collections import defaultdict, deque
from types import SimpleNamespace


class AlignOracleError(AssertionError):
    pass


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class RealBackend:
    """Backend using the installed vLLM package (the deployed image)."""

    def __init__(self) -> None:
        import torch

        from vllm.utils.hashing import sha256
        from vllm.v1.core import kv_cache_utils
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched import scheduler as sched_mod
        from vllm.v1.kv_cache_interface import (
            CircularBufferSpec,
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            MambaSpec,
            MLAAttentionSpec,
        )

        self.torch = torch
        self.sha256 = sha256
        self.kvu = kv_cache_utils
        self.KVCacheManager = KVCacheManager
        self.sched_mod = sched_mod
        self.CircularBufferSpec = CircularBufferSpec
        self.FullAttentionSpec = FullAttentionSpec
        self.KVCacheConfig = KVCacheConfig
        self.KVCacheGroupSpec = KVCacheGroupSpec
        self.MambaSpec = MambaSpec
        self.MLAAttentionSpec = MLAAttentionSpec
        kv_cache_utils.init_none_hash(sha256)
        S = sched_mod.Scheduler
        self.scheduler_functions = {
            name: getattr(S, name)
            for name in (
                "_mamba_block_aligned_split",
                "_sx_record_shared_prefix_stop",
                "_sx_attention_prefix_hit_tokens",
            )
        }
        self.get_block_hash = kv_cache_utils.get_block_hash
        self.get_group_id = kv_cache_utils.get_group_id

    def is_mamba_spec(self, spec) -> bool:
        return isinstance(spec, self.MambaSpec)

    def make_kv_cache_config(self, block_size: int, num_blocks: int, layout: str):
        t = self.torch
        full = self.FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1, dtype=t.float32
        )
        groups = [self.KVCacheGroupSpec(["qsa.main"], full)]
        if layout == "qwen38":
            groups.append(
                self.KVCacheGroupSpec(
                    ["qsa.compressed"],
                    self.MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=t.float32,
                        compress_ratio=4,
                    ),
                )
            )
            groups.append(
                self.KVCacheGroupSpec(
                    ["qsa.ring"],
                    self.CircularBufferSpec(
                        block_size=4,
                        num_kv_heads=1,
                        head_size=1,
                        head_size_v=0,
                        dtype=t.float32,
                    ),
                )
            )
        gdn = self.MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(t.float32,),
            mamba_cache_mode="align",
        )
        num_gdn = 3 if layout == "qwen38" else 1
        for i in range(num_gdn):
            groups.append(self.KVCacheGroupSpec([f"gdn{i}"], gdn))
        if layout == "qwen38":
            groups.append(
                self.KVCacheGroupSpec(
                    ["ple"],
                    self.MambaSpec(
                        block_size=block_size,
                        shapes=((2,),),
                        dtypes=(t.float16,),
                        mamba_cache_mode="align",
                        tp_replicated=True,
                    ),
                )
            )
        return self.KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        )

    def make_manager(self, kv_cache_config, max_model_len: int, hash_block_size: int):
        return self.KVCacheManager(
            kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=True,
            hash_block_size=hash_block_size,
        )

    def make_request(self, request_id: str, prompt: list[int], hash_block_size: int,
                     max_tokens: int):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=100)
        return Request(
            request_id=request_id,
            prompt_token_ids=list(prompt),
            sampling_params=params,
            pooling_params=None,
            block_hasher=self.kvu.get_request_block_hasher(
                hash_block_size, self.sha256
            ),
        )


def make_scheduler_shim(backend, manager, block_size: int, *, multiblock: bool,
                        tail: bool = True, max_blocks: int = 0, shared: bool = False,
                        split_fn=None):
    """Object carrying what the Scheduler split/shared helpers read."""
    fns = dict(backend.scheduler_functions)
    if split_fn is not None:
        fns["_mamba_block_aligned_split"] = split_fn
    cls = type("SchedulerShim", (), fns)
    shim = cls()
    shim.mamba_state_block_size = block_size
    shim.cache_config = SimpleNamespace(block_size=block_size,
                                        enable_prefix_caching=True)
    shim.use_eagle = False
    shim.kv_cache_manager = manager
    shim._sx_align_multiblock = multiblock
    shim._sx_align_tail = tail
    shim._sx_align_max_blocks = max_blocks
    shim._sx_align_shared = shared
    return shim


def output_token(request_id: str, position: int) -> int:
    return 50_000 + zlib.crc32(f"{request_id}:{position}".encode()) % 4096


class AlignSim:
    def __init__(self, backend, *, block_size: int = 784,
                 hash_block_size: int | None = None, budget: int = 8192,
                 max_num_seqs: int = 24, num_blocks: int = 4096,
                 max_model_len: int = 131072, multiblock: bool = True,
                 tail: bool = True, max_blocks: int = 0, shared: bool = False,
                 lag_in_flight: bool = False, split_fn=None,
                 layout: str = "qwen38") -> None:
        self.backend = backend
        self.B = block_size
        self.hash_B = hash_block_size or block_size
        assert self.B % self.hash_B == 0
        cfg = backend.make_kv_cache_config(block_size, num_blocks, layout)
        self.kv_cache_config = cfg
        self.manager = backend.make_manager(cfg, max_model_len, self.hash_B)
        self.sched = make_scheduler_shim(
            backend, self.manager, block_size, multiblock=multiblock, tail=tail,
            max_blocks=max_blocks, shared=shared, split_fn=split_fn,
        )
        self.budget = budget
        self.max_num_seqs = max_num_seqs
        self.lag = lag_in_flight
        self.mamba_group_ids = [
            i for i, g in enumerate(cfg.kv_cache_groups)
            if backend.is_mamba_spec(g.kv_cache_spec)
        ]
        self.waiting: deque = deque()
        self.running: list = []
        self.state: dict[tuple[int, int], tuple[int, int]] = {}
        self.state_col: dict[str, int] = {}
        self.chunks: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.hits: dict[str, int] = {}
        self.num_outputs: dict[str, int] = defaultdict(int)
        self.max_tokens: dict[str, int] = {}
        self.requests: dict[str, object] = {}
        self.finished: list[str] = []
        self.ident_by_hash: dict[bytes, tuple[int, int]] = {}
        self._hash_seen: dict[str, int] = defaultdict(int)
        self._pending_in_flight: list[tuple[object, int]] = []
        self.steps = 0
        self.multiblock_steps = 0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def ident(tokens, n: int) -> tuple[int, int]:
        return (n, hash(tuple(tokens[:n])))

    def _mamba_manager(self, gid: int):
        return self.manager.coordinator.single_type_managers[gid]

    def _register_hashes(self, req) -> None:
        scale = self.B // self.hash_B
        hashes = req.block_hashes
        tokens = req._all_token_ids
        n_blocks = len(hashes) // scale
        for i in range(self._hash_seen[req.request_id], n_blocks):
            if scale == 1:
                key = hashes[i]
            else:
                key = b"".join(hashes[i * scale:(i + 1) * scale])
            self.ident_by_hash[bytes(key)] = self.ident(tokens, (i + 1) * self.B)
        self._hash_seen[req.request_id] = n_blocks

    def add_request(self, request_id: str, prompt: list[int], max_tokens: int = 4):
        req = self.backend.make_request(request_id, prompt, self.hash_B, max_tokens)
        self.requests[request_id] = req
        self.max_tokens[request_id] = max_tokens
        self._register_hashes(req)
        self.waiting.append(req)
        return req

    # -- one engine step -----------------------------------------------------
    def step(self) -> list[tuple[str, int, int]]:
        self.steps += 1
        m = self.manager
        m.new_step_starts()
        budget = self.budget
        scheduled = []

        for req in list(self.running):
            if budget <= 0:
                break
            num_new = req.num_tokens - req.num_computed_tokens
            if num_new <= 0:
                continue
            num_new = min(num_new, budget)
            num_new = self.sched._mamba_block_aligned_split(req, num_new)
            if num_new == 0:
                continue
            start = req.num_computed_tokens
            blocks = m.allocate_slots(req, num_new)
            assert blocks is not None, "simulation ran out of KV blocks"
            self._schedule(req, start, num_new, scheduled)
            budget -= num_new

        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            computed_blocks, hit = m.get_computed_blocks(req)
            if self.sched._sx_align_shared:
                self.sched._sx_record_shared_prefix_stop(req, hit)
            num_new = min(req.num_tokens - hit, budget)
            num_new = self.sched._mamba_block_aligned_split(req, num_new, hit, 0)
            if num_new == 0:
                break
            blocks = m.allocate_slots(
                req, num_new, num_new_computed_tokens=hit,
                new_computed_blocks=computed_blocks,
            )
            if blocks is None:
                break
            self.waiting.popleft()
            self.running.append(req)
            self.hits[req.request_id] = hit
            req.num_computed_tokens = hit
            self._schedule(req, hit, num_new, scheduled)
            budget -= num_new

        # "GPU": forwards execute in scheduling order on one stream.
        for req, start, num_new in scheduled:
            self._forward(req, start, num_new)
        self._check_cache()

        # Outputs. With lag, the previous step's in-flight tokens retire only
        # now (async scheduling keeps one step in flight while scheduling).
        if self.lag:
            for req, n in self._pending_in_flight:
                req.num_in_flight_tokens -= n
            self._pending_in_flight = [(r, n) for r, _, n in scheduled]
        else:
            for req, _, n in scheduled:
                req.num_in_flight_tokens -= n
        for req, start, num_new in scheduled:
            if req.num_computed_tokens != req.num_tokens:
                continue  # chunked prefill: no sample yet
            rid = req.request_id
            req.append_output_token_ids(output_token(rid, req.num_tokens))
            self.num_outputs[rid] += 1
            self._register_hashes(req)
            if self.num_outputs[rid] >= self.max_tokens[rid]:
                self._finish(req)
        return [(r.request_id, s, n) for r, s, n in scheduled]

    def _schedule(self, req, start: int, num_new: int, scheduled: list) -> None:
        self._check_interior_null(req, start, num_new)
        if (start + num_new - 1) // self.B > start // self.B:
            self.multiblock_steps += 1
        scheduled.append((req, start, num_new))
        self.chunks[req.request_id].append((start, start + num_new))
        req.num_computed_tokens = start + num_new
        req.num_in_flight_tokens += num_new

    def _finish(self, req) -> None:
        self.manager.free(req)
        self.running.remove(req)
        self.state_col.pop(req.request_id, None)
        self.finished.append(req.request_id)

    def preempt(self, request_id: str) -> None:
        """Recompute-preempt a running request, like Scheduler._preempt_request.

        Its blocks go back to the pool (cached checkpoints stay registered),
        the worker forgets its state column, and it re-enters the waiting
        queue at the front with num_computed_tokens = 0, so re-admission does
        a fresh prefix lookup and replays prompt + generated tokens.
        """
        req = self.requests[request_id]
        assert req in self.running, f"{request_id} is not running"
        self.manager.free(req)
        self.running.remove(req)
        self.state_col.pop(request_id, None)
        # In-flight work of the preempted request is discarded with it.
        self._pending_in_flight = [
            (r, n) for r, n in self._pending_in_flight if r is not req
        ]
        req.num_in_flight_tokens = 0
        req.num_computed_tokens = 0
        req.num_preemptions = getattr(req, "num_preemptions", 0) + 1
        self.waiting.appendleft(req)

    def run_until_idle(self, max_steps: int = 100000) -> None:
        for _ in range(max_steps):
            if not self.waiting and not self.running:
                return
            self.step()
        raise AssertionError("simulation did not drain")

    # -- oracle ----------------------------------------------------------------
    def _check_interior_null(self, req, start: int, num_new: int) -> None:
        dst = (start + num_new - 1) // self.B
        for gid in self.mamba_group_ids:
            blocks = self._mamba_manager(gid).req_to_blocks[req.request_id]
            for col in range(cdiv(start, self.B), dst):
                if not blocks[col].is_null:
                    raise AlignOracleError(
                        f"{req.request_id}: interior column {col} of chunk "
                        f"[{start},{start + num_new}) is a real block"
                    )

    def _forward(self, req, start: int, num_new: int) -> None:
        B = self.B
        rid = req.request_id
        tokens = req._all_token_ids
        col = self.state_col.get(rid)
        if col is None:
            col = cdiv(start, B) - 1  # seeded like preprocess_mamba_align
        dst = (start + num_new - 1) // B
        want_in = self.ident(tokens, start) if start > 0 else None
        want_out = self.ident(tokens, start + num_new)
        for gid in self.mamba_group_ids:
            blocks = self._mamba_manager(gid).req_to_blocks[rid]
            if dst >= len(blocks) or blocks[dst].is_null:
                raise AlignOracleError(
                    f"{rid}: no real state block at dst column {dst}"
                )
            dkey = (gid, blocks[dst].block_id)
            if col >= 0 and col != dst:
                if blocks[col].is_null:
                    raise AlignOracleError(
                        f"{rid}: pre-copy source column {col} is null"
                    )
                self.state[dkey] = self.state.get((gid, blocks[col].block_id))
            if want_in is not None and self.state.get(dkey) != want_in:
                raise AlignOracleError(
                    f"{rid}: chunk [{start},{start + num_new}) starts from state "
                    f"{self.state.get(dkey)!r}, expected state after {start} "
                    f"tokens (group {gid}, col {col}->{dst})"
                )
            self.state[dkey] = want_out
        self.state_col[rid] = dst

    def _check_cache(self) -> None:
        get_group_id = self.backend.get_group_id
        get_block_hash = self.backend.get_block_hash
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
                raise AlignOracleError(
                    f"cached Mamba block {blk.block_id} (group {gid}) holds "
                    f"{got!r} but its hash covers {want!r}: a state that was "
                    "never checkpointed is registered in the prefix cache"
                )

    # -- reporting -----------------------------------------------------------
    def chunk_ends(self, request_id: str) -> list[int]:
        return [end for _, end in self.chunks[request_id]]

    def prefill_chunk_ends(self, request_id: str) -> list[int]:
        prompt_len = self.requests[request_id].num_prompt_tokens
        return [end for start, end in self.chunks[request_id] if start < prompt_len]
