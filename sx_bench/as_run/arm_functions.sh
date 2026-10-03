#!/bin/bash
# arm_functions.sh : the watchdog used by every comparison run (sourced by run_main_final2.sh, run_v2b.sh, run_final2.sh).
# Stops the running arm container if the host's MemAvailable drops under 25 GiB or production 8031 fails its health check three
# times in a row, and creates the file $W/ABORT so the chain stops. $W is the results directory of the run.
watch_arm() {  # watch_arm <container> : background; stops the container on memory shortage or production trouble
  local c=$1 bad=0 n=0 m h
  while docker inspect $c >/dev/null 2>&1; do
    m=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    if [ "$m" -lt 25 ]; then echo "$(date +%T) STOP $c: MemAvailable ${m} GiB" >> $W/watch.log; docker stop -t 20 $c >/dev/null 2>&1; touch $W/ABORT; return; fi
    if [ $((n % 6)) = 0 ]; then
      h=$(curl -s -m 8 -o /dev/null -w '%{http_code}' localhost:8031/health)
      if [ "$h" = 200 ]; then bad=0; else bad=$((bad+1)); fi
      echo "$(date +%T) $c mem_avail=${m}GiB load=$(cut -d' ' -f1 /proc/loadavg) 8031=$h" >> $W/watch.log
      if [ "$bad" -ge 3 ]; then echo "$(date +%T) STOP $c: 8031 health failed" >> $W/watch.log; docker stop -t 20 $c >/dev/null 2>&1; touch $W/ABORT; return; fi
    fi
    n=$((n + 1)); sleep 10
  done
}
