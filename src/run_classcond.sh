#!/bin/bash
# Run the pre-registered FedPCT-CC analysis on both datasets (PREREG_CLASSCOND.md).
cd ${FEDPCT_ROOT}/src
python classcond_dev.py iwildcam > ../logs/classcond_dev_iwildcam.log 2>&1
python classcond_dev.py cct > ../logs/classcond_dev_cct.log 2>&1
echo ALL_DONE >> ../logs/classcond_dev_cct.log
