#!/bin/bash
# check_8031_seq32.sh : confirm the two switched knobs are live in the running production container.
N=shixiang-inference-swift15-flashnext-tp4
echo "--- runtime env ---"
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $N | grep -e MOE_TUNE -e SM70_NVFP4 -e PLE_HOST_GIB
echo "--- cmdline ---"
docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' $N | grep -e max-num-seqs -e max-model-len -e gpu-memory-utilization
echo "--- capture shapes / kv / concurrency lines ---"
docker logs $N 2>&1 | grep -e "no-MTP decode cudagraph request shapes" -e "GPU KV cache size" -e "Maximum concurrency" | awk '!s[$0]++' | cut -c1-170
echo "--- tune lines ---"
docker logs $N 2>&1 | grep -i -e tuned -e autotune | tail -6 | cut -c1-200
echo "--- current traffic ---"
curl -s -m 8 localhost:8031/metrics | grep -e "^vllm:num_requests_running{" -e "^vllm:num_requests_waiting{"
echo "RB_CHECK_DONE"
