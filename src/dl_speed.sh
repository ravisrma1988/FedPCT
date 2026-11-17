#!/bin/bash
# Download progress of the CCT stream from curl's own byte counter (/proc/<pid>/io wchar).
TOTAL=112540764160
pid=$(pgrep -f "curl -fsSL --retry 5 --retry-delay 30 https://storage.googleapis.com/public-datasets-lila/caltechcameratraps" | head -1)
tail -n 3 ${FEDPCT_ROOT}/logs/cct_fetch_images.log
[ -z "$pid" ] && { echo "no curl process"; exit 0; }
a=$(awk '/^wchar/ {print $2}' /proc/$pid/io); sleep 20; b=$(awk '/^wchar/ {print $2}' /proc/$pid/io)
rate=$(( (b - a) / 20 ))
echo "downloaded $(( b / 1000000000 )) GB of $(( TOTAL / 1000000000 )) GB ($(( 100 * b / TOTAL ))%)"
echo "speed $(( rate / 1000000 )) MB/s"
[ "$rate" -gt 0 ] && echo "eta $(( (TOTAL - b) / rate / 60 )) min"
date -u
