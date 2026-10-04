#!/bin/bash
# dryrun_cache.sh : dry-run mk_prod_cache.py against the live production compose. Writes only to /tmp.
P=/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml
cd /mnt/2t/build/cmp1003 || exit 1
python3 mk_prod_cache.py "$P" /tmp/test_cache.yaml || exit 1
echo "--- non-comment diff (< old  > new)"
diff <(grep -v '^[[:space:]]*#' "$P") <(grep -v '^[[:space:]]*#' /tmp/test_cache.yaml)
echo "--- new header"
sed -n '1,10p' /tmp/test_cache.yaml
docker compose -f /tmp/test_cache.yaml config -q && echo COMPOSE_OK
rm -f /tmp/test_cache.yaml
