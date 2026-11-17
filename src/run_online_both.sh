#!/bin/bash
# Online evaluation on validated time splits, LAC then RAPS.
cd ${FEDPCT_ROOT}
export PYTHONPATH=src
python src/online.py 5 lac > logs/online_lac.log 2>&1
python src/online.py 5 raps > logs/online_raps.log 2>&1
echo BOTH_DONE >> logs/online_raps.log
