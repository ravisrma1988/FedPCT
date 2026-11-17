#!/bin/bash
# Caltech Camera Traps (LILA BC, CDLA-Permissive 1.0): image-level labels + MegaDetector v5a results.
# TLS verification on (curl default). Downloads go to data/cct only.
set -euo pipefail
mkdir -p ${FEDPCT_ROOT}/data/cct
cd ${FEDPCT_ROOT}/data/cct
curl -fsSL -o caltech_camera_traps.json.zip https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip
curl -fsSL -o mdv5a.json.zip https://lila.science/public/lila-md-results/caltech-camera-traps_mdv5a.0.0_results.json.zip
ls -la
sha256sum *.zip | tee SHA256SUMS_meta
unzip -o -q caltech_camera_traps.json.zip
unzip -o -q mdv5a.json.zip
ls -la
