#!/bin/bash
# pfx_ab.sh on|off : prefix caching ON vs OFF, cold prefill + decode at concurrency 4 (8K/16K/32K/64K prompts),
# for the production Flash-Next image (1cat-vllm-heavily-modified-v1-0930) on GPU 0-3.
#   on  : benchmarks the live production instance on 8031 as it is (prefix caching on). No restart.
#   off : stops production 8031, starts the same image and config with --no-enable-prefix-caching on
#         127.0.0.1:8131, benchmarks it, removes the test container and ALWAYS restores production 8031.
# While the benchmark itself runs, the BuildKit step containers of the official image build are frozen
# (cgroup.freeze) so the compile does not compete for CPU; they are thawed right after, also on any exit.
set -u
PHASE=${1:?on|off}
D=/opt/shixiang-inference/docker-dflash2
P=$D/compose.swift15-flashnext-tp4-gpu0123.yaml
W=/mnt/2t/build/pfx_ab
M=Swift-1.5-Qwen3.8-Flash-Next
TC=sx-pfxoff
cd $W

freeze_build() {
  for f in /sys/fs/cgroup/system.slice/system.slice:docker:*/cgroup.freeze; do
    [ -e "$f" ] && echo 1 > "$f"
  done
  sleep 5
  echo "build frozen: $(ls -d /sys/fs/cgroup/system.slice/system.slice:docker:* 2>/dev/null | wc -l) cgroup(s), cpu idle now $(vmstat 1 2 | tail -1 | awk '{print $15}')%"
}
thaw_build() {
  for f in /sys/fs/cgroup/system.slice/system.slice:docker:*/cgroup.freeze; do
    [ -e "$f" ] && echo 0 > "$f"
  done
}

bench() {  # bench <port> <tag>
  freeze_build
  python3 $W/pfx_bench.py --port $1 --model $M --out $W/result_$2.json 2>&1 | tee $W/bench_$2.log
  thaw_build
  echo "build thawed $(date +%T)"
}

if [ "$PHASE" = on ]; then
  trap 'thaw_build' EXIT
  echo "=== ON arm on production 8031 $(date +%T)"
  docker logs shixiang-inference-swift15-flashnext-tp4 2>&1 | grep -E "Prefix caching is enabled|SX align multi-block|GPU KV cache size" | sed -E "s/^\([A-Za-z_0-9]+ pid=[0-9]+\) //" | sort -u | cut -c1-230
  bench 8031 on
  echo "PFX_ON_DONE $(date +%T)"
  exit 0
fi

python3 - "$P" <<'EOF'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r"(?m)^name: .*$", "name: sx-pfxoff", s, count=1)
s = re.sub(r"(?m)^    container_name: .*$", "    container_name: sx-pfxoff", s, count=1)
s = re.sub(r"(?m)^    restart: .*$", '    restart: "no"', s, count=1)
assert 'ports: ["0.0.0.0:8031:8001"]' in s and "      # - --no-enable-prefix-caching" in s
s = s.replace('ports: ["0.0.0.0:8031:8001"]', 'ports: ["127.0.0.1:8131:8001"]')
s = s.replace("      # - --no-enable-prefix-caching", "      - --no-enable-prefix-caching")
open("/mnt/2t/build/pfx_ab/compose.pfxoff.yaml", "w").write(s)
EOF
diff $P compose.pfxoff.yaml
docker compose -f compose.pfxoff.yaml config -q || { echo "BAD_COMPOSE"; exit 1; }

restore() {
  thaw_build
  echo "=== restore production 8031 $(date +%T)"
  docker logs $TC > $W/engine_pfxoff.log 2>&1 || true
  docker compose -f $W/compose.pfxoff.yaml down 2>&1 | tail -2
  docker compose -f $P up -d 2>&1 | tail -2
  T0=$(date +%s)
  until curl -sf -m 5 localhost:8031/health -o /dev/null; do
    [ $(( $(date +%s) - T0 )) -gt 2700 ] && { echo "PRODUCTION_HEALTH_TIMEOUT"; return 1; }
    sleep 15
  done
  echo "production healthy after $(( $(date +%s) - T0 ))s"
  curl -s -m 120 localhost:8031/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Swift-1.5-Qwen3.8-Flash-Next","messages":[{"role":"user","content":"用一句话介绍北京。"}],"max_tokens":48,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('smoke:', d['choices'][0]['message']['content'][:80])"
}
trap 'restore; echo PFX_OFF_DONE $(date +%T)' EXIT

echo "=== stop production 8031 $(date +%T)"
docker compose -f $P stop 2>&1 | tail -2
echo "=== up test (prefix caching off) $(date +%T)"
docker compose -f compose.pfxoff.yaml up -d 2>&1 | tail -2
T0=$(date +%s)
until curl -sf -m 5 localhost:8131/health -o /dev/null; do
  st=$(docker inspect -f '{{.State.Status}}' $TC 2>/dev/null)
  [ "$st" != "running" ] && { echo "TEST_CONTAINER_$st"; docker logs --tail 40 $TC 2>&1 | cut -c1-220; exit 1; }
  [ $(( $(date +%s) - T0 )) -gt 2700 ] && { echo "TEST_HEALTH_TIMEOUT"; docker logs --tail 40 $TC 2>&1 | cut -c1-220; exit 1; }
  sleep 15
done
echo "test healthy after $(( $(date +%s) - T0 ))s"
docker logs $TC 2>&1 | grep -E "Prefix caching|mamba_cache_mode|SX align multi-block|GPU KV cache size|enable_prefix_caching" | sed -E "s/^\([A-Za-z_0-9]+ pid=[0-9]+\) //" | sort -u | cut -c1-260
if docker logs $TC 2>&1 | grep -q "Prefix caching is enabled"; then echo "PREFIX_CACHING_STILL_ON - aborting"; exit 1; fi
bench 8131 off
echo "--- errors in test engine log: $(docker logs $TC 2>&1 | grep -cE 'ERROR|Traceback')"
