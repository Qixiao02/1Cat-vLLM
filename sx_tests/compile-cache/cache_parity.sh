#!/bin/bash
# cache_parity.sh : torch.compile cache parity on the V100 server (see README.md).
#
# For each lane (no-MTP production lane, native MTP k=4 lane) and cache mode it starts
# the engine three times on GPU 4-7 / port 8141 -- cache off, cache on cold, cache on
# warm (reload) -- compares greedy outputs token for token (and the draft acceptance
# counters with MTP) and exits 1 if anything differs or the warm start did not reuse
# what the cold start saved. The instance that normally runs on these GPUs is stopped
# first and started again at the end; production on GPU 0-3 is not touched.
#
# Everything is overridable from the environment:
#   COMPOSE      base compose (shape of compose.swift15-flashnext-tp4-gpu0123.yaml)
#   IMAGE        engine image (a rebuilt image carries SX_OPT_COMPILE_CACHE)
#   GPUS PORT    where the arms run
#   WORK         output root; per lane/mode: $WORK/<lane>-<mode>/ and caches-<lane>-<mode>/
#   LANES        "nomtp mtp"
#   MODES        "subgraph" (SX_OPT_COMPILE_CACHE=1) and/or "aot"
#   ARMS         off,cold,warm  (+ off2 off_noaot warm2 inval, see cache_parity.py)
#   CACHE_SEED   directory copied into every fresh cache dir, e.g. a copy of the production
#                /cache without its vllm/ subdirectory (keeps Triton/TileLang caches warm)
#   STOP_FIRST / RESTART_AFTER   commands run once before / after everything
#   EXTRA        extra arguments for cache_parity.py run (e.g. "--env VLLM_FOO=1")
#
#   sx_tests/compile-cache/cache_parity.sh                       # both lanes, subgraph mode
#   MODES="subgraph aot" LANES=nomtp ARMS=off,cold,warm,inval sx_tests/compile-cache/cache_parity.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
COMPOSE=${COMPOSE:-/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml}
IMAGE=${IMAGE:-shixiang/1cat-vllm-v100:heavily-modified-v1-mtp2-sm70main}
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-8141}
WORK=${WORK:-/mnt/2t/build/cache_parity}
LANES=${LANES:-"nomtp mtp"}
MODES=${MODES:-"subgraph"}
ARMS=${ARMS:-off,cold,warm}
F1C=/opt/shixiang-inference/docker-dflash2/compose.forkwheel-flashnext-tp4-gpu4567.yaml
STOP_FIRST=${STOP_FIRST-"docker compose -f $F1C stop"}   # STOP_FIRST= (empty) runs nothing
RESTART_AFTER=${RESTART_AFTER-"docker compose -f $F1C start"}
EXTRA=${EXTRA:-}

mkdir -p "$WORK"
[ -n "$STOP_FIRST" ] && { eval "$STOP_FIRST"; sleep 15; }
rc=0
for lane in $LANES; do
  for mode in $MODES; do
    echo "=== lane=$lane mode=$mode $(date +%T)"
    seed=()
    [ -n "${CACHE_SEED:-}" ] && seed=(--cache-seed "$CACHE_SEED")
    # shellcheck disable=SC2086
    python3 "$HERE/cache_parity.py" run \
      --lane "$lane" --cache-mode "$mode" --arms "$ARMS" \
      --compose "$COMPOSE" --image "$IMAGE" --gpus "$GPUS" --port "$PORT" \
      --tag "$lane-$mode" --workdir "$WORK/$lane-$mode" --cache-root "$WORK/caches-$lane-$mode" \
      ${seed[@]+"${seed[@]}"} $EXTRA \
      2>&1 | tee "$WORK/$lane-$mode.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && rc=1
  done
done
[ -n "$RESTART_AFTER" ] && eval "$RESTART_AFTER"
[ $rc -ne 0 ] && echo "CACHE_PARITY_FAILED (see $WORK/*/report.md)"
exit $rc
