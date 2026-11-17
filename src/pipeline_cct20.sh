#!/bin/bash
# Wait for the CCT20 download/unpack to finish, then extract features and run the baselines.
set -uo pipefail
cd ${FEDPCT_ROOT}
until grep -q '^DONE' logs/get_cct20.log; do
  pgrep -f get_cct20.sh > /dev/null || { echo "download script died"; exit 1; }
  sleep 30
done
export HF_HOME=${FEDPCT_ROOT}/.hf PYTHONPATH=src
python src/extract_cct20.py > logs/extract_cct20.log 2>&1 || { echo "extract failed"; exit 1; }
python src/run_cct20.py 5 > logs/run_cct20.log 2>&1 || { echo "run failed"; exit 1; }
echo PIPELINE_DONE
