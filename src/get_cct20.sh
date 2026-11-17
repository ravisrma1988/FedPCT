#!/bin/bash
# Download + unpack Caltech Camera Traps ECCV'18 (CCT20) from LILA's GCS bucket.
set -euo pipefail
cd ${FEDPCT_ROOT}/data
mkdir -p cct20 && cd cct20
BASE=https://storage.googleapis.com/public-datasets-lila/caltechcameratraps
for f in eccv_18_annotations.tar.gz eccv_18_all_images_sm.tar.gz; do
  curl -sS -C - --retry 5 -o "$f" "$BASE/$f"
  echo "downloaded $f $(stat -c %s "$f") bytes"
done
tar -xzf eccv_18_annotations.tar.gz
tar -xzf eccv_18_all_images_sm.tar.gz
echo "unpacked: $(find . -name '*.jpg' | wc -l) jpg files"
echo DONE
