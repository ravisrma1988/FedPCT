#!/bin/bash
# Second-backbone analysis (PREREG_BACKBONE.md), end to end. Extraction uses the project venv
# (transformers); everything else the unchanged system environment.
set -euo pipefail
cd ${FEDPCT_ROOT}
export HF_HOME=${FEDPCT_ROOT}/.hf PYTHONPATH=src
.venv_md/bin/python src/crop_features_dinov2.py iwildcam
.venv_md/bin/python src/crop_features_dinov2.py cct
[ -L features/iwildcam_cct_dinov2crop.npy ] || ln -s ${FEDPCT_ROOT}/features/cct_dinov2crop.npy features/iwildcam_cct_dinov2crop.npy
python src/pers_lambda.py 5 dinov2crop
python src/cct_run.py src/pers_lambda.py 5 cct_dinov2crop
python src/backbone_eval.py iwildcam
python src/backbone_eval.py cct
echo RUN_BACKBONE_DONE
