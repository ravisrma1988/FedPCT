#!/bin/bash
# Stream the CCT image archive (~105 GB) straight into tar: the .tar.gz is never stored on disk.
# TLS verification on. Retries on transient failure restart the stream (tar overwrites).
set -uo pipefail
cd ${FEDPCT_ROOT}/data/cct
mkdir -p images
URL=https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/cct_images.tar.gz
for attempt in 1 2 3; do
  echo "attempt $attempt $(date -u)"
  # despite the .gz name the object is an uncompressed tar (112,540,764,160 bytes; checked header)
  if curl -fsSL --retry 5 --retry-delay 30 "$URL" | tar -x -C images; then
    echo "CCT_IMAGES_DONE $(date -u)"; find images -type f -name '*.jpg' | wc -l; exit 0
  fi
  echo "stream failed $(date -u)"; sleep 60
done
echo "CCT_IMAGES_FAILED"; exit 1
