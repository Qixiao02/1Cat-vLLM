#!/bin/bash
# dryrun_seq32.sh : generate the 32-sequence production compose next to the real one and show every difference.
set -eu
D=/opt/shixiang-inference/docker-dflash2
F=$D/compose.swift15-flashnext-tp4-gpu0123.yaml
W=/mnt/2t/build/cmp1003
python3 $W/mk_prod_seq32.py $F /tmp/compose.seq32.test.yaml
docker compose -f /tmp/compose.seq32.test.yaml config -q && echo COMPOSE_OK
echo "--- full diff old -> new ---"
diff $F /tmp/compose.seq32.test.yaml | cut -c1-200
echo "--- non-comment diff ---"
diff <(grep -vE "^\s*#" $F) <(grep -vE "^\s*#" /tmp/compose.seq32.test.yaml) | cut -c1-200
echo "--- new header ---"
head -6 /tmp/compose.seq32.test.yaml
echo "--- flags / env that must have changed ---"
grep -n -e "max-num-seqs" -e "MOE_TUNE_MAX_TOKENS" /tmp/compose.seq32.test.yaml
echo "DRYRUN_OK"
