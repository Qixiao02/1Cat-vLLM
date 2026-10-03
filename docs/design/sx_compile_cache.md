# SX_OPT_COMPILE_CACHE: reusing torch.compile work across restarts (Flash-Next lanes)

Status: implemented, **default off**, not yet validated on a V100. Validate with
`sx_tests/compile-cache/cache_parity.sh` (see the README next to it) before
turning it on anywhere.

## Why

A cold start of the Swift-1.5 Flash-Next NVFP4 engine on 4 V100 takes 513-600 s.
torch.compile is about 185 s of it (the prefill/mixed backbone and the decode
backbone are two compiles, 87-98 s each, Dynamo alone 38-42 s), and the decode
compile runs inside CUDA-graph capture (~95 s of the graph phase). The Flash-V100
compile graph forces `VLLM_DISABLE_COMPILE_CACHE=1` and in-memory AOT, so none
of it is reused. The reason was greedy token drift after reloading cached
artifacts (June 2026).

## The switch

| `SX_OPT_COMPILE_CACHE` | meaning |
|---|---|
| unset, `0`, `off` | today, byte for byte: the config forces the opt-out, `VLLM_USE_AOT_COMPILE=1` in memory. No cache key, directory name or code path changes. |
| `1`, `on`, `subgraph` | no forced opt-out; `VLLM_USE_AOT_COMPILE=0`: every piecewise submodule's compiled graph is reloaded from `VLLM_CACHE_ROOT/torch_compile_cache/<key>/rank_*/<prefix>/`, Dynamo traces again on every start (about 40 s per compiler). The strategy upstream qualified for the 27B/E4M3/DFlash2 contract (#753). |
| `aot` | no forced opt-out; `VLLM_USE_AOT_COMPILE=1`: the AOT artifact (FX graph and compiled pieces) is reloaded, no Dynamo. Fastest restart; this is the path whose parity gate failed upstream. |

An explicit `VLLM_DISABLE_COMPILE_CACHE` or `VLLM_USE_AOT_COMPILE` always wins
(and a conflict is logged). Any other value is "off".

With the switch on, each rank also logs `sx-compile-cache build identity ...`
once and `SX compile-cache counters: ...` after warmup.

## Which upstream changes matter for the Flash-Next lanes

Upstream base of this fork: `02c87ab89`; compared with `origin/main` at
`dd3b52c69`. Many commits appear under several SHAs (merge, cherry-pick,
squash); each row is one change.

| upstream | title | needed? | what was done |
|---|---|---|---|
| `3780a4460` (#621) | drop the forced compile-cache opt-out (config + `envs.disable_compile_cache`) | yes, the point | hand-ported as the switch (`3052f7802`): the same two places, gated; `VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING` stays (meaningful while the switch is off) |
| `b504a224b` / `ad7e0fb23` (#675, merged `d30469863`) | restore Triton state on the first AOT reload; backend source hash ignores `<frozen ...>` | yes for `aot` mode | cherry-picked (`82514f813`), docs hunks dropped; the frozen-name hash is gated on the switch so cache-off directory names do not move |
| `5f668ebb9` (#622) | resolve the PLE table pointer inside the gather op | yes, for every mode that reloads a compiled graph: the production lane gathers through `qwen4_exp_ple_pinned_gather(…, table_ptr: int, …)` and Inductor writes the literal address into the artifact | hand-adapted and gated (`fff2aa2e6`): new op `qwen4_exp_ple_pinned_gather_by_name`, module registers under its prefix only with the switch on; off = the old op and graph |
| `9326ea76c` (#753) | reuse compiled graph caches (AOT off) for E4M3 DFlash2 | the strategy yes, the DFlash2/E4M3/TP4 admission no (no DFlash2 lane here) | `subgraph` mode is that strategy for the Flash-Next lanes |
| `00b6672c7` | ship the torch 2.10.0 backport (`tools/torch_patches`) | **no** for the default configuration, see below | cherry-picked and dormant; opt-in Docker build arg `SX_TORCH_BACKPORTS=1` |
| `d64a257a2` `7983a516d` `90604f53a` `16dc373d9` (#710/#768), `a03926298` (#672) | resolve FP8 / TurboMind QPN4 workspace addresses on reload | no: those are the `Fp8LinearMethod`, compressed-tensors FP8 and compressed-tensors NVFP4 QPN4 routes; the ModelOpt NVFP4 linears of this checkpoint use op kind `nvfp4` with no address argument, and the excluded layers are FP16 | not ported. The harness fails on `resides on host memory` in a warm log, so a missed route cannot pass silently |
| `5f02520b5` | revert of `b504a224b` inside one merge scope | no | the net upstream effect is #675 |
| `36979d488` `5d216a8bb` (#757) | keep explicit mode/cudagraph_mode in the Flash-V100 policies | no, independent of reload correctness | not ported |
| `9d3fedbda` `a9e37ade2` | effective cache state in `vllm/sm70_profiles` | no (the module does not exist here) | replaced by the two log lines above |
| `8f79edd55`, `ac1e67685` (#780) | CUDA-graph event nodes for shared score workspace; XQA staged-rescale scratch lifetime | no: graph lifetime of the 79T prefill / E4M3 KV paths, not the compile cache | not ported |
| `fa3c4535b`, `2273c24d7` | per-piecewise autotune isolation; flashinfer-sm70 state placeholders | no: not on upstream main (rejected experiment #682 / WIP) | not ported |

Order: #675, then the switch (it calls the frozen-name option of #675's
helper), then the PLE change (it uses the switch and the baked-constant
registry of the switch commit).

### The torch 2.10.0 backport is not the torch patch you need

`00b6672c7` patches the installed torch's `torch/_dynamo/aot_compile_types.py`
so that `BundledAOTAutogradSerializableCallable` pickles the Triton kernel side
table next to the artifact (pytorch #173556, in 2.11+; 2.11 drops Volta from the
cu128 wheels, so 2.10.0 stays pinned). It is applied by
`tools/torch_patches/apply.sh` after the install, sha256-checked and idempotent.
Upstream ships the script and the patch but nothing runs it: not the
Dockerfile, not the wheel, not `setup.py`.

On torch 2.10 vLLM's AOT path is the *non-mega* one:
`VLLM_USE_MEGA_AOT_ARTIFACT` defaults to on only from 2.12, so
`standalone_compile` runs without `aot=True` and `bundled_autograd_cache` is
forced off (`compiler_interface._get_vllm_functorch_config`). The bundled-AOT
serializer is never called. What is pickled instead is the FX graph owned by
`VllmSerializableFunction`, whose Triton nodes #675 remaps; upstream's own notes
say the backport alone does not cover that serializer. The backport only matters
if someone sets `VLLM_USE_MEGA_AOT_ARTIFACT=1` on 2.10. For that case the image
gets an opt-in build step (`docker build --build-arg SX_TORCH_BACKPORTS=1`), and
`sx_compile_cache.torch_backport_level()` is part of the cache key because
artifacts written with and without the patch have different formats. No change
to `setup.py` or the wheel is needed or possible (a wheel cannot patch torch).

## The cache key

Two caches, both under `VLLM_CACHE_ROOT` (`/cache/vllm` with
`XDG_CACHE_HOME=/cache`):

* AOT artifact (`aot` mode): `torch_aot_compile/<sha256>/rank_<r>_<dp>/model`,
  key = `envs.compile_factors()` hash, `VllmConfig.compute_hash()`, and per
  compiled class `vllm.__version__`, forward qualname and first line number.
  The source files Dynamo inlined are compared by content when the artifact is
  loaded (`_verify_source_unchanged`). The Inductor/torch version enters only on
  the mega path.
* compiled subgraphs (every mode that saves): `torch_compile_cache/<10 hex>/
  rank_<r>_<dp>/<prefix>/vllm_compile_cache.py` plus artifacts, key =
  `[env hash, config hash, hash of the traced source files, compiler hash]`
  (torch version, Inductor and functorch config, system). The directory name is
  the *only* check: a hit loads artifact N for graph N without comparing the
  graph.

What was already in the key in this fork: every registered `VLLM_` variable by
its getter value, every `VLLM_`/`SX_OPT_` variable present in the environment
(unregistered ones since upstream #536, `SX_OPT_*` since the MTP-lane work), the
whole `VllmConfig` (model, cache, parallel, scheduler incl. `max_num_batched_tokens`
and `max_num_seqs`, speculative, compilation incl. capture sizes and compile
ranges, kernel config). The dual-compile decode backbone gets its own key from its
copy of the config (smaller `max_num_batched_tokens` and compile range), the MTP
drafter from its draft model config.

What the switch adds, only when it is on (off: the key is unchanged and
`SX_OPT_COMPILE_CACHE=0` is not even hashed):

| factor | covers |
|---|---|
| canonical mode | `1`/`on`/`subgraph` are one key, `aot` another |
| build identity (`sx_compile_cache.build_identity`, once per process, ~0.3 s) | torch, CUDA, triton versions, Python minor, device name (NVML, no CUDA init in the API server), content of every `.py` file of `vllm` (and `flash_attn_v100`, `flash_qla`, `flashinfer_sm70` when installed), name+size+head/tail content of every shared library, torch backport level, `TORCHINDUCTOR_*`, `TORCHDYNAMO_*`, `TORCH_COMPILE*`, `TRITON_*` (not the directories), `FLASH_QLA_*`, `Q_SCALE_CONSTANT`, `V_SCALE_CONSTANT`, `USE_DEFAULT_FLA_NORM`, `CUBLAS_WORKSPACE_CONFIG` |
| baked constants (`register_baked_constant`) | values model code derives from runtime state and that become literals of the graph: the device/host row split of the PLE table (automatic host budget depends on free memory) |
| checkpoint identity (in `VllmConfig.compute_hash`) | file names, sizes, mtimes of the checkpoint files, json by content: NVFP4 linears bake a per-layer global scale (a Python float read from the weights) into the graph |

Why a build identity: the Dynamo-traced source files cover Python that was
inlined; they do not cover Triton kernels that Inductor embeds, `register_fake`
implementations, custom-op signatures, native libraries or the torch build, and
`vllm.__version__` is the same for two images built from the same version string.
"Any change anywhere in the package invalidates" is deliberate: a recompile costs
minutes, a wrongly reused graph costs correctness. A cache directory that
outlives several images simply accumulates unused entries.

Not covered, by design: the weights' bytes (only names, sizes and mtimes; a
content-identical copy with a new mtime misses, a different checkpoint with the
same size *and* mtime would hit), the GPU driver version, and values of
non-prefixed environment variables other than the list above.

## Risks

* **Quality.** June 2026: reloading cached AOT artifacts of the Flash-V100 graph
  produced deterministic greedy drift (index 196 of a request), and plain
  non-AOT torch.compile drifted against the in-memory AOT baseline too. Upstream
  (September) traced the cache-key part of that to unregistered `VLLM_` switches
  (fixed, #536, in this fork) but its own 27B/DFlash2 validation (2026-10-02)
  still failed complete-output parity for AOT reload (2106 vs 2181 tokens, first
  difference at 1370) while the compiled-subgraph reuse with AOT off matched all
  three starts. The lead there is not the reload but the Triton autotune: a cold
  compile and a reload ended with different `.best_config` for 14-23 kernels
  (reduction tiles 64 vs 256), and 17 of 136 tuning keys carried conflicting
  saved configs; per-rank and per-subgraph isolation did not help. Our lane
  additionally compiles with `combo_kernels` and `benchmark_combo_kernel`, whose
  choices are timing based. The harness therefore has the `off2` control: if two
  cache-off starts already differ, a cache difference proves nothing.
* **Save can fail loudly.** With the cache on, a compiled piece that cannot be
  serialized raises (`The compiled artifact is not serializable`) instead of
  falling back; the cold arm would not come up. AOT save failures are only a
  warning (`unable to save AOT compiled function`); the harness fails on it.
* **First reload.** Artifacts are only trustworthy after the *first* reload in a
  fresh process (the empty side table); the warm arm is that reload, `warm2` the
  second.
* **Cache-on startup is not free.** `subgraph` still pays Dynamo (about 40 s per
  compiler) and the decode compile inside graph capture; upstream measured 92 s
  against 76 s for AOT reload on the 27B, 221 s cold. Both still capture CUDA
  graphs (83-143 s here).

## What could not be verified without a GPU

Everything numerical: parity of cold, warm and cache-off outputs; that `subgraph`
and `aot` actually reload on the Flash-Next graphs (including the dual-compile
decode backbone and the MTP drafter); that no other op bakes a process-local
value; the Triton autotune determinism; startup times. The CPU tests check the
gating logic, that off is byte-identical to `8593319b2` (getters, cache key, lane
block, PLE op calls), the key's sensitivity, the Triton side-table remap on real
FX graphs, and the harness against a fake engine.
