#!/bin/bash
# Wait for the iWildCam download/verify to finish, then extract BioCLIP features.
set -uo pipefail
cd ${FEDPCT_ROOT}
until grep -q '^DONE' logs/get_iwildcam.log; do
  pgrep -f get_iwildcam.sh > /dev/null || { echo "download/verify script stopped without DONE"; exit 1; }
  sleep 30
done
export HF_HOME=${FEDPCT_ROOT}/.hf PYTHONPATH=src
python src/extract_iwildcam.py > logs/extract_iwildcam.log 2>&1 || { echo "extract failed"; exit 1; }
echo PIPELINE_DONE
