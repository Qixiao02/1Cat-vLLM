# sx_tests/compile-cache

Tests and the V100 validation harness for `SX_OPT_COMPILE_CACHE` (design and
analysis: [`docs/design/sx_compile_cache.md`](../../docs/design/sx_compile_cache.md)).

| file | what | needs |
|---|---|---|
| `test_switch_cpu.py` | the switch, the getters in `envs.py`, the lane block of `VllmConfig`, the cache key (with the switch off: identical to `8593319b2`) | Python + pytest + torch (CPU) |
| `test_ple_gather_cpu.py` | PLE gather by layer name instead of a baked pointer | same |
| `test_aot_side_table_cpu.py` | Triton side-table save/restore of the AOT serializer (#675) | same |
| `test_parity_cpu.py` | the harness itself, end to end against a fake engine | same |
| `cache_parity.py`, `cache_parity.sh` | the V100 harness | docker, 4 free V100, the model |

Run the CPU tests with one pytest process per directory (the boot helpers
register bare `vllm` packages in `sys.modules`):

    python -m pytest sx_tests/compile-cache -q

## What `cache_parity` does

It starts the engine several times (GPU 4-7, port 8141, production flags of the
Flash-Next lane, or the native-MTP k=4 lane) against cache directories it
controls, and compares only what the cache is allowed to change: the startup
time.

| arm | `SX_OPT_COMPILE_CACHE` | cache dir | tells you |
|---|---|---|---|
| `off` | `0` | `cache-off`, fresh | the reference (today's behaviour) and its time to healthy |
| `cold` | mode | `cache-on`, fresh | the compile with saving: same model as `off`? |
| `warm` | mode | `cache-on`, reused | the reload: reuse really happened, outputs did not move |
| `off2` | `0` | `cache-off`, reused | control: is the reference reproducible across restarts at all? |
| `off_noaot` | `0` + `VLLM_USE_AOT_COMPILE=0` | `cache-off` | control: AOT vs non-AOT numerics, no reload involved |
| `warm2` | mode | `cache-on`, reused | reload of an already reloaded artifact |
| `inval` | mode + unused `VLLM_SX_CACHE_PROBE=1` | `cache-on`, reused | must recompile: a changed switch changes the cache key |

Default arms: `off,cold,warm`. Modes: `subgraph` (`SX_OPT_COMPILE_CACHE=1`:
compiled subgraphs reloaded, `VLLM_USE_AOT_COMPILE=0`) and `aot`
(`SX_OPT_COMPILE_CACHE=aot`: whole AOT artifact reloaded; the warm arm runs with
`VLLM_FORCE_AOT_LOAD=1` so a load failure is an error, not a silent recompile).

Per arm it records

* time to healthy (from `docker compose up -d` to `/health`), `Model loading`,
  `torch.compile` (slowest rank), Dynamo, compile-range and cache-load times,
  graph capture, init engine, KV cache size, all parsed from the engine log;
* the compile/reload counters each rank logs (`SX compile-cache counters: ...`);
* greedy outputs (token ids via `return_token_ids`, text, finish reason) of 10
  fixed prompts: code, math, Chinese, JSON, translation, logic, and four long
  ones (needle codes, a summary, a code question; 8K, 16K and 32K tokens,
  28K in the MTP lane whose context is 32K), thinking off, temperature 0, seed
  0, 256-512 tokens, one request at a time (so batch-dependent numerics and
  prefix-cache hits are out of the picture; every prompt starts with its own
  session id);
* with MTP: the `vllm:spec_decode_*` counters (drafts, draft tokens, accepted
  tokens, accepted per position) before and after the prompt pass.

It exits **1** and prints `CACHE PARITY FAILED` if

* any compared pair (`off`/`cold`, `off`/`warm`, `cold`/`warm`, `warm`/`warm2`,
  `cold`/`inval`) differs by a single token or, with MTP, by a single draft
  counter;
* the warm arm did not reuse what the cold arm saved (`subgraph`: loaded no
  compiled artifact or compiled and saved a new one; `aot`: fewer AOT loads than
  saves, or any AOT compile);
* the warm log shows a load failure, a failed save, an illegal memory access, a
  stale pointer (`resides on host memory`), a missing Triton kernel index or a
  changed source file;
* any arm or request failed, or `inval` reused the cache.

A difference between `off` and `off2` (or `off_noaot`) is printed as a note: the
reference itself is then not reproducible, and a cache pair difference is not
evidence against the cache. `report.md` and `report.json` hold the table; the
raw `arm_<name>.json` (outputs, parsed log, counters), `engine_<name>.log`,
`compose.<name>.yaml` and `prompts.json` stay in the work directory.

## Run it

    # both lanes, subgraph mode, arms off,cold,warm  (about 35-40 minutes per lane)
    sx_tests/compile-cache/cache_parity.sh

    # the AOT reload, with the control and the invalidation arm
    MODES="subgraph aot" LANES=nomtp ARMS=off,off2,cold,warm,inval \
        sx_tests/compile-cache/cache_parity.sh

    # compare arm files of two runs by hand
    python3 sx_tests/compile-cache/cache_parity.py compare --cache-mode aot \
        --arm off=run1/arm_off.json --arm cold=run2/arm_cold.json --arm warm=run2/arm_warm.json

The wrapper stops the instance that normally runs on GPU 4-7 first and starts
it again at the end (`STOP_FIRST`, `RESTART_AFTER`). Set `CACHE_SEED` to a copy
of the production `/cache` **without its `vllm/` subdirectory** so every fresh
cache directory starts with warm Triton and TileLang caches, like production
(without it the first arm of each directory also pays the Triton/TileLang JIT,
and `off` and `cold` are slower than they would be in service). Bind-mounted
directories are made world-writable for the container user.

`IMAGE` must carry this change (`SX_OPT_COMPILE_CACHE` is ignored by older
images and all arms would run the same code).

### Other launchers

The default launcher turns a compose file shaped like
`compose.swift15-flashnext-tp4-gpu0123.yaml` into a `vllm serve` arm the way
`sx_bench/as_run/arm_compose.py` does and fails with the unmatched pattern when
the shape is different. `--engine-script PATH` replaces it: the script is called
as `PATH start|alive|logs|stop` with `CP_ARM`, `CP_LANE`, `CP_PORT`,
`CP_CACHE_DIR` (host directory that must be `/cache` of the engine),
`CP_ENV_JSON` (a JSON file with the engine environment, including
`SX_OPT_COMPILE_CACHE`) and `CP_WORKDIR` in its environment. `start` launches the
engine, `alive` exits 0 while it runs, `logs` prints its log, `stop` removes it.
`--flag`, `--drop-flag` and `--env` adjust the `vllm serve` flags and the
environment of every arm.
