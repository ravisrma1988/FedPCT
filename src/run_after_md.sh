#!/bin/bash
# After MegaDetector: crop features, then the backbone + novelty checks on crop features.
cd ${FEDPCT_ROOT}
export HF_HOME=${FEDPCT_ROOT}/.hf PYTHONPATH=src
until [ -f logs/md_done.flag ]; do sleep 60; done
CROP_TEST=200 python src/crop_features.py > logs/crop_test200.log 2>&1 || { echo "crop test failed"; exit 1; }
python src/crop_features.py > logs/crop_features.log 2>&1 || { echo "crop failed"; exit 1; }
python src/compare_backbones.py 5 bioclip2 bioclip2crop > logs/compare_backbones_crop.log 2>&1
python src/diag_novelty.py bioclip2crop 3 > logs/diag_novelty_crop.log 2>&1
echo AFTER_MD_DONE >> logs/diag_novelty_crop.log
