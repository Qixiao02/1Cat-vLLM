#!/bin/bash
# dryrun_startup.sh : dry-run mk_prod_startup.py (both levers) against the live production compose.
# Writes only to /tmp.
P=/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml
cd /mnt/2t/build/cmp1003 || exit 1
python3 mk_prod_startup.py "$P" /tmp/test_startup.yaml || exit 1
echo "--- non-comment diff (< old  > new)"
diff <(grep -v '^[[:space:]]*#' "$P") <(grep -v '^[[:space:]]*#' /tmp/test_startup.yaml)
echo "--- new header"
sed -n '1,12p' /tmp/test_startup.yaml
echo "--- command tail"
sed -n '/^    command:/,$p' /tmp/test_startup.yaml | tail -5
docker compose -f /tmp/test_startup.yaml config -q && echo COMPOSE_OK
rm -f /tmp/test_startup.yaml
