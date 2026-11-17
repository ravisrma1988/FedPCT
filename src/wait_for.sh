#!/bin/bash
# wait_for.sh FILE PATTERN MAXSEC: poll until PATTERN appears in FILE (or timeout), then print its tail.
cd ${FEDPCT_ROOT}
for ((t = 0; t < $3; t += 15)); do
  grep -q -E "$2" "$1" 2>/dev/null && break
  sleep 15
done
date -u
tail -n 40 "$1"
