#!/bin/bash
# Protocol v2 with the cluster-bootstrap test: time split (primary), then seq split (secondary).
cd ${FEDPCT_ROOT}
export PYTHONPATH=src
python -c "import pandas as pd; print('NaT datetimes:', pd.to_datetime(pd.read_csv('features/iwildcam_meta.csv').datetime, errors='coerce').isna().sum())" > logs/run_v2_time.log
python src/run_iwildcam.py 5 time >> logs/run_v2_time.log 2>&1
python src/run_iwildcam.py 5 seq > logs/run_v2_seq.log 2>&1
echo BOTH_DONE >> logs/run_v2_seq.log
