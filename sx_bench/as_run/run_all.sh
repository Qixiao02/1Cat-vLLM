#!/bin/bash
# run_all.sh : fork 1001 vs official main@d30469863 on GPU 4-7, one arm after the other (see arm_compose.py).
# F1 is the 1001 instance already running there (production settings); every other arm is started, measured and
# removed. After O1 the chain stops if O1 does not come close to upstream's own C4 figure (266 tok/s aggregate at
# 8K input / 256 greedy tokens, docs/design/sm70_qwen38_concurrency40.md). At the end the 1001 instance is started
# again. Production 8031 (GPU 0-3) is not touched.
set -u
W=/mnt/2t/build/cmp1001
B=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
F1C=/opt/shixiang-inference/docker-dflash2/compose.forkwheel-flashnext-tp4-gpu4567.yaml
cd $W

bench() {  # bench <arm> <name> <pfx_bench args...>
  local a=$1 name=$2; shift 2
  python3 $B/pfx_bench.py --port 8141 --model $M --gpus 4,5,6,7 --out $W/${a}_${name}.json "$@" > $W/${a}_${name}.log 2>&1
  grep -E "pass [0-9]|ERROR|Traceback|WAVE" $W/${a}_${name}.log | sed "s/^/[$a $name] /" | cut -c1-250
}

tests() {  # tests <arm> <std|mtp>
  local a=$1
  echo "=== tests $a $(date +%T)"
  python3 /mnt/2t/build/mtp_kv/kvq_check.py 8141 $M $W/${a}_answers.json 2>&1 | tail -1 | sed "s/^/[$a] /"
  if [ "$2" = std ]; then
    for c in 1 2 4 8 16 24; do
      [ $c = 24 ] && [ "${a:0:1}" = O ] && continue      # official arms run max-num-seqs 16
      bench $a sweep_c$c --conc $c --lengths 8000 --gen 256 --passes 2 --temperature 0 --seed 2026100101
      if [ $a = O1 ] && [ $c = 4 ]; then
        agg=$(python3 -c "import json;c=json.load(open('$W/O1_sweep_c4.json'))['cells'];print(round(sum(x['decode_agg_tok_s'] for x in c)/len(c),1))")
        echo "[O1] C4 aggregate decode $agg tok/s (upstream's own figure 266; gate 226)"
        python3 -c "import sys;sys.exit(0 if $agg >= 226 else 1)" || { echo "GATE_FAIL O1 does not reproduce upstream"; return 1; }
      fi
    done
    bench $a long --conc 4 --lengths 8000,16000,32000,64000 --gen 400 --passes 2 --seed 2026100103
    bench $a l128k --conc 1 --lengths 128000 --gen 256 --passes 1 --seed 2026100104
  else
    python3 $W/mtp_bench.py 8141 $M $W/${a}_mtp.json 2>&1 | grep mtp_bench | sed "s/^/[$a] /"
    bench $a mtp_c1_8k --conc 1 --lengths 8000 --gen 256 --passes 1 --temperature 0 --seed 2026100101
    bench $a mtp_c4_8k --conc 4 --lengths 8000 --gen 256 --passes 1 --temperature 0 --seed 2026100101
  fi
  echo "[$a] gpu mem (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | tr '\n' ' ')"
}

keylines() {
  docker logs $1 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished|speculative_config=|Auto-setting VLLM_SM70_QWEN38_BATCH|Auto-enabling" \
    | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^INFO [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | cut -c1-200 | head -8
  echo "auto-set switches: $(docker logs $1 2>&1 | grep -c 'Auto-setting')"
}

arm() {  # arm <arm> <std|mtp>
  local a=$1 n=sx-cmp-$(echo $1 | tr A-Z a-z)
  python3 $W/arm_compose.py $a > /dev/null && docker compose -f $W/compose.$a.yaml config -q || { echo "BAD_COMPOSE $a"; return 1; }
  echo "=== start $a $(date +%T)"
  docker compose -f $W/compose.$a.yaml up -d 2>&1 | tail -1
  local t0=$(date +%s)
  until curl -sf -m 5 localhost:8141/health -o /dev/null; do
    st=$(docker inspect -f '{{.State.Status}}' $n 2>/dev/null)
    if [ "$st" != running ] || [ $(( $(date +%s) - t0 )) -gt 3600 ]; then
      echo "ARM_FAIL $a ($st)"; docker logs --tail 30 $n 2>&1 | cut -c1-250
      docker logs $n > $W/engine_$a.log 2>&1; docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1; return 1
    fi
    sleep 15
  done
  echo "[$a] healthy after $(( $(date +%s) - t0 ))s"
  keylines $n | sed "s/^/[$a] /"
  tests $a $2; local rc=$?
  echo "[$a] errors in engine log: $(docker logs $n 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
  docker logs $n > $W/engine_$a.log 2>&1
  docker compose -f $W/compose.$a.yaml down 2>&1 | tail -1
  return $rc
}

echo "=== F1 = the running 1001 instance (production settings) $(date +%T)"
keylines shixiang-inference-forkwheel-flashnext-tp4 | sed "s/^/[F1] /"
tests F1 std
docker logs shixiang-inference-forkwheel-flashnext-tp4 > $W/engine_F1.log 2>&1
docker compose -f $F1C stop 2>&1 | tail -1
if arm O1 std; then
  arm O2 std
  arm F2 std
  arm OM mtp
  arm FM mtp
else
  echo "CHAIN_STOPPED_AT_O1"
fi
echo "=== start the 1001 instance again $(date +%T)"
docker compose -f $F1C start 2>&1 | tail -1
echo "RUN_ALL_DONE $(date +%T)"
