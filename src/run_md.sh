#!/bin/bash
# 4 MegaDetector processes sharing the MIG slice.
cd ${FEDPCT_ROOT}
source .venv_md/bin/activate
for i in 0 1 2 3; do
  python src/md_run.py $i 4 > logs/md_shard$i.log 2>&1 &
done
wait
echo MD_ALL_DONE > logs/md_done.flag
