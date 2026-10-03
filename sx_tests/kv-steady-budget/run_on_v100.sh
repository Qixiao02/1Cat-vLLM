#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run_on_v100.sh: does --gpu-memory-utilization mean what it says?
#
# For each utilisation of a list this starts the engine (Swift-1.5 Qwen3.8-
# Flash-Next NVFP4, TP4, V100) from an image, records
#   * "GPU KV cache size" and "Available KV cache memory" from the engine log,
#   * the engine's own "KV steady budget" / "KV steady audit" lines,
#   * nvidia-smi memory.used of every GPU every 2 s: while starting, idle, and
#     during a stress run (four concurrent 8K prompts + two 16K at once, then
#     one 32K prompt),
#   * out-of-memory errors, tracebacks and failed requests,
# and fails a utilisation when the highest memory.used of any GPU exceeds the
# peak limit (default 32200 MiB), when less than --headroom-mib of CUDA-usable
# memory would be free at that peak, when a request failed, when the engine log
# has an error, or when the engine's audit says its reservation was SHORT.
#
# Needs: docker compose, nvidia-smi, python3 (standard library only), curl, and
# four idle GPUs. It does not touch any other container.
#
# Usage:
#   run_on_v100.sh --image IMAGE --models-dir DIR [options]
#
#   --image IMAGE         image to test (required); it must contain this change
#   --models-dir DIR      host directory that holds Swift-1.5-Qwen3.8-Flash-Next-NVFP4
#   --cache-dir DIR       host cache directory (compile/triton caches; persistent
#                         between trials, default /opt/shixiang-inference/cache-kvsteady-tp4)
#   --lane MTP|no-MTP     MTP: native MTP k=4, max-num-seqs 16, max-model-len 32768
#                         no-MTP: production shape, max-num-seqs 24, max-model-len 131072
#   --utils "0.87 0.90 0.93 0.95"
#                         utilisations to run (default per lane)
#   --switch 1|0|both     SX_OPT_KV_STEADY_BUDGET for the trials. "both" (default)
#                         runs every util with 1 and, first, the first util with 0
#                         as the "today" baseline
#   --gpus 4,5,6,7        GPUs to use (must be idle)
#   --port 8141           host port bound on 127.0.0.1
#   --out DIR             results directory (default ./kv-steady-<time>)
#   --peak-limit-mib N    FAIL above this memory.used (default 32200)
#   --headroom-mib N      FAIL when less CUDA-usable memory is free at the peak (default 500)
#   --start-timeout S     seconds to wait for /health (default 2400)
#   --sample-interval S   seconds between nvidia-smi samples (default 2)
#   --max-num-seqs N / --max-model-len N   override the lane shape
#   --env K=V             extra container environment (repeatable)
#   --patch-repo REPO [--patch-base REF]
#                         the change is Python only: run the image with the
#                         vllm/ files that differ between REF (default
#                         sx/mtp-integrate) and REPO's HEAD overlaid read-only,
#                         instead of building an image that contains them
#   --patch-dir DIR       same with an explicit directory: every file under
#                         DIR/vllm/ is overlaid on the image's package
#   --site-packages DIR   where the image keeps vllm
#                         (default /opt/venv/lib/python3.12/site-packages)
#   --no-stress           only start and idle (no requests)
#   --force               do not refuse to start on GPUs that are in use
#   -h, --help
#
# Example (MTP lane, the four utilisations, baseline first):
#   sx_tests/kv-steady-budget/run_on_v100.sh \
#       --image shixiang/1cat-vllm-v100:IMAGE --models-dir /path/to/models \
#       --lane MTP --utils "0.87 0.90 0.93 0.95"
#
# Exit status: 0 when every trial passes, 1 when any fails, 2 on a usage or
# environment error.

set -u -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

IMAGE=""
MODELS_DIR=""
CACHE_DIR="/opt/shixiang-inference/cache-kvsteady-tp4"
LANE="MTP"
UTILS=""
SWITCH="both"
GPUS="4,5,6,7"
PORT=8141
OUT=""
PEAK_LIMIT_MIB=32200
HEADROOM_MIB=500
START_TIMEOUT=2400
SEQS=""
MAXLEN=""
STRESS=1
FORCE=0
EXTRA_ENV=()
PATCH_DIR=""
PATCH_REPO=""
PATCH_BASE="sx/mtp-integrate"
SITE_PACKAGES=""
SERVED_MODEL="Swift-1.5-Qwen3.8-Flash-Next"
SAMPLE_INTERVAL=2

usage() { sed -n '3,/^set -u/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "run_on_v100.sh: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=${2:?}; shift 2 ;;
    --models-dir) MODELS_DIR=${2:?}; shift 2 ;;
    --cache-dir) CACHE_DIR=${2:?}; shift 2 ;;
    --lane) LANE=${2:?}; shift 2 ;;
    --utils) UTILS=${2:?}; shift 2 ;;
    --switch) SWITCH=${2:?}; shift 2 ;;
    --gpus) GPUS=${2:?}; shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --peak-limit-mib) PEAK_LIMIT_MIB=${2:?}; shift 2 ;;
    --headroom-mib) HEADROOM_MIB=${2:?}; shift 2 ;;
    --start-timeout) START_TIMEOUT=${2:?}; shift 2 ;;
    --sample-interval) SAMPLE_INTERVAL=${2:?}; shift 2 ;;
    --max-num-seqs) SEQS=${2:?}; shift 2 ;;
    --max-model-len) MAXLEN=${2:?}; shift 2 ;;
    --env) EXTRA_ENV+=("${2:?}"); shift 2 ;;
    --patch-dir) PATCH_DIR=${2:?}; shift 2 ;;
    --patch-repo) PATCH_REPO=${2:?}; shift 2 ;;
    --patch-base) PATCH_BASE=${2:?}; shift 2 ;;
    --site-packages) SITE_PACKAGES=${2:?}; shift 2 ;;
    --no-stress) STRESS=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

[ -n "$IMAGE" ] || die "--image is required"
[ -n "$MODELS_DIR" ] || die "--models-dir is required"
case "$SWITCH" in 0|1|both) ;; *) die "--switch must be 1, 0 or both" ;; esac
LANE_INFO=$(python3 "$HERE/compose_gen.py" --lane-info "$LANE" 2>/dev/null) \
  || die "--lane must be MTP or no-MTP"
LANE_NORM=${LANE_INFO%%|*}
[ -n "$UTILS" ] || UTILS=${LANE_INFO#*|}
for tool in docker nvidia-smi python3 curl; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[ -d "$MODELS_DIR" ] || die "models dir $MODELS_DIR does not exist"
[ -n "$OUT" ] || OUT="./kv-steady-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT" "$CACHE_DIR" || die "cannot create $OUT or $CACHE_DIR"
OUT=$(cd "$OUT" && pwd)

if [ -n "$PATCH_REPO" ]; then
  [ -z "$PATCH_DIR" ] || die "--patch-repo and --patch-dir are exclusive"
  command -v git >/dev/null 2>&1 || die "--patch-repo needs git"
  PATCH_DIR="$OUT/patch"
  rm -rf "$PATCH_DIR"; mkdir -p "$PATCH_DIR"
  changed=$(git -C "$PATCH_REPO" diff --name-only --diff-filter=AM "$PATCH_BASE"...HEAD -- vllm) \
    || die "git diff $PATCH_BASE...HEAD failed in $PATCH_REPO"
  [ -n "$changed" ] || die "no vllm/ files differ between $PATCH_BASE and HEAD in $PATCH_REPO"
  while IFS= read -r f; do
    mkdir -p "$PATCH_DIR/$(dirname "$f")"
    git -C "$PATCH_REPO" show "HEAD:$f" > "$PATCH_DIR/$f"
  done <<< "$changed"
  echo "overlaying $(echo "$changed" | wc -l) changed file(s) from $PATCH_REPO ($PATCH_BASE...HEAD) over the image"
fi
[ -z "$PATCH_DIR" ] || [ -d "$PATCH_DIR/vllm" ] || die "$PATCH_DIR has no vllm/ directory"
LANE_TAG=$(echo "$LANE_NORM" | tr 'A-Z' 'a-z' | tr -d '-')
SAMPLER_PID=""
CURRENT_COMPOSE=""

stop_sampler() {
  if [ -n "$SAMPLER_PID" ]; then
    kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
    SAMPLER_PID=""
  fi
}
compose_down() {
  if [ -n "$CURRENT_COMPOSE" ]; then
    docker compose -f "$CURRENT_COMPOSE" down 2>&1 | tail -1
    CURRENT_COMPOSE=""
  fi
}
cleanup() { stop_sampler; compose_down; }
trap cleanup EXIT
trap 'echo "interrupted"; exit 130' INT TERM

event() { echo "$(date +%s.%N) $1" >> "$2"; }

gpu_used_max() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
}

start_sampler() {  # start_sampler <samples file>
  (
    while :; do
      ts=$(date +%s.%N)
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
        | sed "s/^/$ts, /"
      sleep "$SAMPLE_INTERVAL"
    done >> "$1"
  ) &
  SAMPLER_PID=$!
}

run_trial() {  # run_trial <switch> <util>
  local sw=$1 util=$2
  local tag="${LANE_TAG}-s${sw}-u${util//./}"
  local dir="$OUT/$tag"
  local compose="$dir/compose.yaml"
  local cname="sx-kvsteady-$tag"
  mkdir -p "$dir"
  : > "$dir/samples.csv"; : > "$dir/events.log"

  echo
  echo "=== $tag: lane $LANE_NORM, util $util, SX_OPT_KV_STEADY_BUDGET=$sw  $(date +%T)"
  local used
  used=$(gpu_used_max)
  if [ "$FORCE" != 1 ] && [ "${used:-0}" -gt 1500 ]; then
    echo "GPUs $GPUS are in use (${used} MiB); refusing to start (use --force to override)" >&2
    return 2
  fi

  local gen=(python3 "$HERE/compose_gen.py" --image "$IMAGE" --tag "$tag" --lane "$LANE_NORM" \
    --util "$util" --switch "$sw" --gpus "$GPUS" --models-dir "$MODELS_DIR" --cache-dir "$CACHE_DIR" \
    --port "$PORT" --out "$compose")
  [ -n "$SEQS" ] && gen+=(--max-num-seqs "$SEQS")
  [ -n "$MAXLEN" ] && gen+=(--max-model-len "$MAXLEN")
  [ -n "$PATCH_DIR" ] && gen+=(--patch-dir "$PATCH_DIR")
  [ -n "$SITE_PACKAGES" ] && gen+=(--site-packages "$SITE_PACKAGES")
  local kv
  for kv in "${EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && gen+=(--env "$kv"); done
  "${gen[@]}" || { echo "compose generation failed" >&2; return 2; }
  docker compose -f "$compose" config -q || { echo "BAD_COMPOSE $compose" >&2; return 2; }

  CURRENT_COMPOSE="$compose"
  start_sampler "$dir/samples.csv"
  event compose_up "$dir/events.log"
  docker compose -f "$compose" up -d 2>&1 | tail -1

  local t0 ready=0 state
  t0=$(date +%s)
  while :; do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" -o /dev/null; then ready=1; break; fi
    state=$(docker inspect -f '{{.State.Status}}' "$cname" 2>/dev/null || echo gone)
    if [ "$state" != running ]; then echo "container is $state before it became healthy"; break; fi
    if [ $(( $(date +%s) - t0 )) -gt "$START_TIMEOUT" ]; then echo "no /health after ${START_TIMEOUT}s"; break; fi
    sleep 10
  done
  if [ "$ready" = 1 ]; then
    event ready "$dir/events.log"
    echo "healthy after $(( $(date +%s) - t0 )) s"
    sleep $(( SAMPLE_INTERVAL * 5 ))      # idle samples
    if [ "$STRESS" = 1 ]; then
      python3 "$HERE/stress.py" --port "$PORT" --model "$SERVED_MODEL" \
        --out "$dir/stress.json" --events "$dir/events.log" || echo "stress: some requests failed"
      sleep $(( SAMPLE_INTERVAL * 3 ))
    fi
  fi
  docker logs "$cname" > "$dir/engine.log" 2>&1 || true
  stop_sampler
  compose_down

  local stress_arg=()
  [ -f "$dir/stress.json" ] && stress_arg=(--stress "$dir/stress.json")
  grep -E "GPU KV cache size|Available KV cache memory|Maximum concurrency|Graph capturing finished|KV steady" "$dir/engine.log" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //' | awk '!s[$0]++' | cut -c1-260 | head -40
  python3 "$HERE/analyze.py" run --samples "$dir/samples.csv" --events "$dir/events.log" \
    --log "$dir/engine.log" "${stress_arg[@]}" --tag "$tag" --util "$util" --switch "$sw" \
    --lane "$LANE_NORM" --peak-limit-mib "$PEAK_LIMIT_MIB" --headroom-mib "$HEADROOM_MIB" \
    --out "$OUT/result.$tag.json"
}

first_util=${UTILS%% *}
any_fail=0
if [ "$SWITCH" = both ] || [ "$SWITCH" = 0 ]; then
  if [ "$SWITCH" = 0 ]; then
    for u in $UTILS; do run_trial 0 "$u" || any_fail=1; done
  else
    run_trial 0 "$first_util" || any_fail=1       # "today": the baseline row
  fi
fi
if [ "$SWITCH" = both ] || [ "$SWITCH" = 1 ]; then
  for u in $UTILS; do run_trial 1 "$u" || any_fail=1; done
fi

echo
echo "=== summary ($OUT)"
python3 "$HERE/analyze.py" table "$OUT"/result.*.json | tee "$OUT/summary.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || any_fail=1
echo "peak limit ${PEAK_LIMIT_MIB} MiB, headroom ${HEADROOM_MIB} MiB; raw samples, engine logs and compose files are under $OUT"
exit "$any_fail"
