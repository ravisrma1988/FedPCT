#!/bin/bash
# DALI extraction for BioCLIP (check against PIL features) and BioCLIP-2.
cd ${FEDPCT_ROOT}
export HF_HOME=${FEDPCT_ROOT}/.hf PYTHONPATH=src
python src/extract_dali.py hf-hub:imageomics/bioclip bioclip_dali 512 > logs/extract_bioclip_dali.log 2>&1
python - >> logs/extract_bioclip_dali.log 2>&1 <<'EOF'
import numpy as np
a = np.load("features/iwildcam_bioclip.npy").astype(np.float32)
b = np.load("features/iwildcam_bioclip_dali.npy").astype(np.float32)
c = (a * b).sum(1)
print(f"PIL vs DALI cosine: mean {c.mean():.4f}  p1 {np.quantile(c, 0.01):.4f}  min {c.min():.4f}")
EOF
python src/extract_dali.py hf-hub:imageomics/bioclip-2 bioclip2 256 > logs/extract_bioclip2.log 2>&1
echo EXTRACT_DONE >> logs/extract_bioclip2.log
