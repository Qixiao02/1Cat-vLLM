#!/bin/bash
# Run the weight-load lever A/B after chain_rev.sh has fully finished.
cd /mnt/2t/build/cmp1003 || exit 1
echo "=== chain_load start $(date +%T)"
while pgrep -f 'chain_rev\.sh' >/dev/null 2>&1; do sleep 30; done
echo "chain_rev gone at $(date +%T) (iteration done)"
i=0
while [ $i -lt 240 ]; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | sort -n | tail -1)
  avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$busy" -lt 1500 ] && [ "$avail" -ge 100 ]; then break; fi
  echo "waiting: gpu_max=${busy} MiB MemAvailable=${avail} GiB $(date +%T)"
  sleep 20
  i=$((i + 1))
done
bash /mnt/2t/build/cmp1003/startup_ab_load.sh 2>&1 | tee /mnt/2t/build/cmp1003/startup_ab_load.log
echo "CHAIN_LOAD_DONE $(date +%T)"
