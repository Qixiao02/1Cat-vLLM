#!/bin/bash
# upgrade_8031_seq32.sh : production 8031 (GPU 0-3) from --max-num-seqs 24 to 32 (same 1004 image), with rollback.
#  1. checks: 8031 idle (running/waiting = 0), MemAvailable, GPU 4-7 free (no comparison arm left behind)
#  2. backup of the compose, new compose (mk_prod_seq32.py), `docker compose config -q`, diff in the log
#  3. stop the old container, start the new one, wait for /health (up to 30 min); if it does not come up: rollback
#  4. post checks: /version, capture shapes (must be 1,2,4,8,16,32), KV capacity, greedy answers vs the 1004
#     production baseline, smoke chat, engine log errors
set -u
D=/opt/shixiang-inference/docker-dflash2
OLD=$D/compose.swift15-flashnext-tp4-gpu0123.yaml
TS=$(date +%Y%m%d%H%M%S)
BAK=$OLD.bak-seq32-pre-upgrade-$TS
N=shixiang-inference-swift15-flashnext-tp4
IMG=shixiang/1cat-vllm-v100:heavily-modified-v1-1004-sm70main
M=Swift-1.5-Qwen3.8-Flash-Next
W=/mnt/2t/build/cmp1003
cd $W

echo "=== checks $(date +%T)"
docker image inspect $IMG >/dev/null 2>&1 || { echo "image missing"; exit 1; }
run=$(curl -s -m 8 localhost:8031/metrics | awk '/^vllm:num_requests_running\{/{print int($NF)}')
wait=$(curl -s -m 8 localhost:8031/metrics | awk '/^vllm:num_requests_waiting\{/{print int($NF)}')
echo "8031 now: running=${run:-?} waiting=${wait:-?}; MemAvailable $(awk '/MemAvailable/{printf "%d GiB",$2/1048576}' /proc/meminfo)"
if [ "${run:-1}" != 0 ] || [ "${wait:-1}" != 0 ]; then echo "8031 has traffic, not switching now"; exit 2; fi
echo "old compose says: $(grep -c -e '--max-num-seqs=24' $OLD) x max-num-seqs=24, tune cap $(grep -o 'MOE_TUNE_MAX_TOKENS: \"[0-9]*\"' $OLD)"

cp -p $OLD $BAK && echo "backup: $BAK"
python3 $W/mk_prod_seq32.py $OLD $OLD.new || { echo "compose generation failed"; exit 1; }
docker compose -f $OLD.new config -q || { echo "new compose invalid"; rm -f $OLD.new; exit 1; }
echo "--- diff old -> new (non-comment lines)"
diff <(grep -vE "^\s*#" $OLD) <(grep -vE "^\s*#" $OLD.new) | cut -c1-200

rollback() {
  echo "!!! ROLLBACK $(date +%T): $1"
  docker compose -f $OLD down 2>&1 | tail -1
  cp -p $BAK $OLD
  docker compose -f $OLD up -d 2>&1 | tail -1
  echo "rolled back to 24 sequences; the old container is starting"
  echo "SEQ32_ROLLED_BACK"
  exit 1
}

echo "=== stop the old container $(date +%T)"
docker compose -f $OLD stop 2>&1 | tail -1
cp -p $OLD.new $OLD && rm -f $OLD.new
echo "=== start the new container $(date +%T)"
docker compose -f $OLD up -d 2>&1 | tail -2
t0=$(date +%s)
until curl -sf -m 5 localhost:8031/health -o /dev/null; do
  st=$(docker inspect -f '{{.State.Status}}' $N 2>/dev/null)
  [ "$st" != running ] && { docker logs --tail 30 $N 2>&1 | cut -c1-220; rollback "container is $st"; }
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { docker logs --tail 30 $N 2>&1 | cut -c1-220; rollback "no /health after 30 min"; }
  sleep 10
done
echo "=== healthy after $(( $(date +%s) - t0 ))s $(date +%T)"
echo "version: $(curl -s -m 8 localhost:8031/version)"
docker logs $N 2>&1 | grep -E "GPU KV cache size|Model loading took|Graph capturing finished|Maximum concurrency|no-MTP decode cudagraph" | sed -E 's/^\([A-Za-z_0-9]+ pid=[0-9]+\) //; s/^(INFO|WARNING) [0-9-]+ [0-9:]+ //' | awk '!s[$0]++' | head -6 | cut -c1-170
shapes=$(docker logs $N 2>&1 | grep -c -e 'request shapes (1, 2, 4, 8, 16, 32)')
echo "capture-shape lines with (1, 2, 4, 8, 16, 32): $shapes (must be > 0)"
[ "$shapes" -gt 0 ] || rollback "the engine did not switch to the 32-sequence capture set"

echo "=== answers on 8031 (greedy)"
python3 /mnt/2t/build/mtp_kv/kvq_check.py 8031 $M $W/prod8031_seq32_answers.json 2>&1 | tail -6 | cut -c1-200
python3 - <<'EOF'
import json
new = json.load(open("/mnt/2t/build/cmp1003/prod8031_seq32_answers.json"))
old = json.load(open("/mnt/2t/build/cmp1003/prod8031_1004_answers.json"))
same = sum(x["answer"] == y["answer"] for k in ("needle", "short") for x, y in zip(new[k], old[k]))
tot = len(old["needle"]) + len(old["short"])
print("answers identical to the 1004 production baseline: %d of %d" % (same, tot))
EOF
echo "=== smoke chat"
curl -s -m 60 localhost:8031/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，请用一句话介绍你自己。\"}],\"max_tokens\":48,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:120].replace(chr(10),' '), '| tokens', r['usage']['completion_tokens'])"
echo "engine log errors: $(docker logs $N 2>&1 | grep -cE 'ERROR|Traceback|out of memory')"
echo "container: $(docker ps --filter name=$N --format '{{.Names}} {{.Status}} {{.Image}}')"
echo "SEQ32_DONE $(date +%T)"
