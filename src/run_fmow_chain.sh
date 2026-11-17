#!/bin/bash
# FMoW confirmatory chain (PREREG_FMOW.md): wait for the lambda selection, then the development dry run,
# the freeze and the one-time lockbox evaluation, strictly in this order; stops at the first failure.
set -uo pipefail
cd ${FEDPCT_ROOT}
export PYTHONPATH=src
while pgrep -f "pers_lambda.py 5 fmow" > /dev/null; do sleep 30; done
test -s results/pers_lambda_fmow_dinov2.json || { echo "CHAIN_FAIL lambda selection"; exit 1; }
python src/lockbox_eval_fmow.py dev > logs/final_eval_fmow_dev.log 2>&1 || { echo "CHAIN_FAIL dev"; exit 1; }
python src/freeze_fmow.py > logs/freeze_fmow.log 2>&1 || { echo "CHAIN_FAIL freeze"; exit 1; }
python src/lockbox_eval_fmow.py lockbox > logs/final_eval_fmow_lockbox.log 2>&1 || { echo "CHAIN_FAIL lockbox"; exit 1; }
echo CHAIN_DONE
