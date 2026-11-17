"""Extract frozen BioCLIP features for all iWildCam v2.0 images, in metadata.csv row order.

Outputs in features/:
  iwildcam_bioclip.npy   (N, 512) float16, L2-normalised
  iwildcam_meta.csv      copy of metadata.csv (row i <-> feature row i)
"""
import os
import shutil

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader

from extract_cct20 import Images

ROOT = "${FEDPCT_ROOT}"
DATA = f"{ROOT}/data/iwildcam_v2.0"


def main():
    meta = pd.read_csv(f"{DATA}/metadata.csv")
    model, _, tf = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    model = model.cuda().eval()
    dl = DataLoader(Images(list(meta.filename), f"{DATA}/train", tf), batch_size=512,
                    num_workers=32, pin_memory=True)
    feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, x in enumerate(dl):
            f = model.encode_image(x.cuda(non_blocking=True))
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).half().cpu())
            if i % 50 == 0:
                print(f"batch {i}/{len(dl)}", flush=True)
    os.makedirs(f"{ROOT}/features", exist_ok=True)
    np.save(f"{ROOT}/features/iwildcam_bioclip.npy", torch.cat(feats).numpy())
    shutil.copy(f"{DATA}/metadata.csv", f"{ROOT}/features/iwildcam_meta.csv")
    print("saved", len(meta), "features")


if __name__ == "__main__":
    main()
