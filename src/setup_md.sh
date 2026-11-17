#!/bin/bash
# Isolated venv for MegaDetector inference. megadetector and ultralytics-yolov5 are installed with
# --no-deps (their pins, e.g. protobuf<=3.20.1, only matter for training/logging); the light runtime
# deps are added under a constraints file that forbids replacing the NGC torch/torchvision/numpy.
set -euo pipefail
cd ${FEDPCT_ROOT}
[ -d .venv_md ] || python -m venv --system-site-packages .venv_md
source .venv_md/bin/activate
python -c "import torch, torchvision, numpy; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\nnumpy=={numpy.__version__}')" > md_constraints.txt
pip install -q --no-deps megadetector==10.0.28 ultralytics-yolov5==0.1.1 2>&1 | grep -viE 'deprecat|egg' | tail -3 || true
pip install -q -c md_constraints.txt jsonpickle humanfriendly opencv-python-headless send2trash dill \
  "fastquadtree>=1.1.2" pyyaml seaborn thop clipboard 2>&1 | grep -viE 'deprecat|egg' | tail -3 || true
python -c "import torch; print('torch unchanged:', torch.__version__, '| cuda', torch.cuda.is_available())"
mkdir -p models
[ -f models/md_v5a.0.0.pt ] || curl -sSL --retry 3 -o models/md_v5a.0.0.pt \
  https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt
stat -c '%s bytes' models/md_v5a.0.0.pt
sha256sum models/md_v5a.0.0.pt
echo SETUP_DONE
