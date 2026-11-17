#!/bin/bash
# Download iWildCam v2.0 from the HF mirror (emmilly/iwildcam-wilds-v2; unofficial, see notes),
# check its SHA-256, extract safely, and verify against the WILDS v2.0 statistics.
set -euo pipefail
cd ${FEDPCT_ROOT}/data
URL=https://huggingface.co/datasets/emmilly/iwildcam-wilds-v2/resolve/main/iwildcam_v2.0.tar.gz
SHA=05c872f46eee44315f79bc90f4098c69ae3799a2b95116db02292ea1b4e370d2
curl -sSL -C - --retry 5 -o iwildcam_v2.0.tar.gz "$URL"
echo "downloaded $(stat -c %s iwildcam_v2.0.tar.gz) bytes"
echo "$SHA  iwildcam_v2.0.tar.gz" | sha256sum -c -
python - <<'EOF'
import tarfile
with tarfile.open("iwildcam_v2.0.tar.gz") as t:
    t.extractall("iwildcam_v2.0", filter="data")  # blocks absolute paths / traversal / devices
print("extracted")
EOF
touch iwildcam_v2.0/RELEASE_v2.0.txt
python ${FEDPCT_ROOT}/src/verify_iwildcam.py
echo DONE
