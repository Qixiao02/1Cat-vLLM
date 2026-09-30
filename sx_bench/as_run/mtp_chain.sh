#!/bin/bash
# mtp_chain.sh : MTP (k=4) retest of the fork on GPU 4-7 with the PLE prefill buffer fix (port of upstream #707 plus
# the grouping on the speculative path), same configuration as the trial that ran out of memory on 2026-09-30
# (gpu_memory_utilization 0.87, max_num_seqs 24, 131072 context, 8192-token prefill step). The official comparison
# lane is stopped for the trial and started again afterwards.
set -u
W=/mnt/2t/build/mtp_kv
D=/opt/shixiang-inference/docker-dflash2
cd $W
echo "=== stop official lane on GPU 4-7 $(date +%T)"
docker compose -f $D/compose.official-flashnext-tp4-gpu4567.yaml stop 2>&1 | tail -1
#                 tag          util seqs maxlen batched extra patch
bash $W/mtp_kv.sh p707-u87s24  0.87 24   131072 8192    ""    $W/patch707
echo "=== restart official lane on 8011 $(date +%T)"
docker compose -f $D/compose.official-flashnext-tp4-gpu4567.yaml start 2>&1 | tail -1
echo "MTP_CHAIN_DONE $(date +%T)"
