"""mtp_compose.py <src compose> <tag> <util> <seqs> <maxlen> <batched> [K=V,...] [patch dir] : write
compose.<tag>.yaml for one MTP (k=4) trial of the fork on GPU 4-7, test port 127.0.0.1:8141, derived from the
production 8031 compose. Every file under <patch dir>/vllm/ is bind-mounted read-only over the same file of the
installed package, so a source change can be tried without building an image."""
import os
import re
import sys

src, tag, util, seqs, maxlen, batched = sys.argv[1:7]
extra = sys.argv[7] if len(sys.argv) > 7 else ""
patch_dir = sys.argv[8] if len(sys.argv) > 8 else ""
s = open(src).read()


def sub(pattern, repl):
    global s
    s, n = re.subn(pattern, repl, s, count=1, flags=re.M)
    assert n == 1, pattern


sub(r"^name: .*$", "name: sx-mtp-%s" % tag)
sub(r"^    container_name: .*$", "    container_name: sx-mtp-%s" % tag)
sub(r"^    restart: .*$", '    restart: "no"')
sub(r'^    ports: \["0\.0\.0\.0:8031:8001"\]$', '    ports: ["127.0.0.1:8141:8001"]')
sub(r"source: /opt/shixiang-inference/cache-flashnext-tp4, target: /cache",
    "source: /opt/shixiang-inference/cache-mtp-test-tp4, target: /cache")
sub(r"^      GPU_MEMORY_UTILIZATION: .*$", '      GPU_MEMORY_UTILIZATION: "%s"' % util)
sub(r"^      MAX_NUM_SEQS: .*$", '      MAX_NUM_SEQS: "%s"' % seqs)
sub(r"^      MAX_MODEL_LEN: .*$", '      MAX_MODEL_LEN: "%s"' % maxlen)
sub(r"^      MAX_NUM_BATCHED_TOKENS: .*$", '      MAX_NUM_BATCHED_TOKENS: "%s"' % batched)
sub(r"^      MTP_SPECULATIVE_TOKENS: .*$", '      MTP_SPECULATIVE_TOKENS: "4"')
sub(r"^      VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: .*$", '      VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS: "1200"')
sub(r'^              device_ids: \["0","1","2","3"\]$', '              device_ids: ["4","5","6","7"]')
for kv in [x for x in extra.split(",") if x]:
    key, value = kv.split("=", 1)
    s = re.sub(r"(?m)^      %s: .*\n" % re.escape(key), "", s)
    m = re.search(r"(?m)^    environment:\n", s)
    s = s[:m.end()] + '      %s: "%s"\n' % (key, value) + s[m.end():]
if patch_dir:
    mounts = []
    for root, _, files in os.walk(os.path.join(patch_dir, "vllm")):
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, patch_dir)
            mounts.append("      - {type: bind, source: %s, target: /opt/venv/lib/python3.12/site-packages/%s, "
                          "read_only: true}\n" % (path, rel))
    assert mounts, "no files under %s/vllm" % patch_dir
    m = re.search(r"(?m)^    volumes:\n", s)
    s = s[:m.end()] + "".join(sorted(mounts)) + s[m.end():]
open("/mnt/2t/build/mtp_kv/compose.%s.yaml" % tag, "w").write(s)
