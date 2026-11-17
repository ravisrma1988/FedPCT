"""DINOv2 ViT-L/14 features of the FMoW-WILDS images (label-free): 224x224 RGB input (the WILDS
release is already 224x224), ImageNet mean/std, pooled CLS output (pooler_output, 1024-d),
L2-normalised, float16, rows in the order of features/fmow_meta.csv. Same model and preprocessing
as the DINOv2 crop features of the camera-trap backbone analysis, applied to whole images.
Usage: python fmow_features.py
"""
import os

import numpy as np
import pandas as pd
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from transformers import Dinov2Model

ROOT = "${FEDPCT_ROOT}"
MODEL, SIZE, BATCH = "facebook/dinov2-large", 224, 256
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OUT = f"{ROOT}/features/fmow_dinov2.npy"
assert not os.path.exists(OUT), "FMoW features exist"
meta = pd.read_csv(f"{ROOT}/features/fmow_meta.csv", usecols=["row", "filename"])
assert (meta.row.to_numpy() == np.arange(len(meta))).all()
files = [f"{ROOT}/data/fmow/images/{f}" for f in meta.filename]
model = Dinov2Model.from_pretrained(MODEL).cuda().eval()


def source(info):
    i = info.idx_in_epoch % len(files)
    return np.fromfile(files[i], dtype=np.uint8), np.array([i], np.int64)


@pipeline_def(batch_size=BATCH, num_threads=16, device_id=0, prefetch_queue_depth=4)
def pipe():
    png, idx = fn.external_source(source=source, num_outputs=2, batch=False, parallel=False)
    x = fn.decoders.image(png, device="mixed", output_type=types.RGB)
    x = fn.resize(x, size=(SIZE, SIZE), interp_type=types.INTERP_CUBIC, antialias=True)
    x = fn.crop_mirror_normalize(x, dtype=types.FLOAT, mean=[255 * m for m in MEAN], std=[255 * s for s in STD],
                                 output_layout="CHW")
    return x, idx


p = pipe()
p.build()
it = DALIGenericIterator(p, ["img", "idx"], size=len(files), last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=False)
out = np.zeros((len(files), 1024), np.float16)
seen = np.zeros(len(files), bool)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for k, b in enumerate(it):
        i = b[0]["idx"].squeeze(-1).cpu().numpy()
        f = torch.nn.functional.normalize(model(pixel_values=b[0]["img"]).pooler_output.float(), dim=-1)
        out[i] = f.cpu().numpy().astype(np.float16)
        seen[i] = True
        if k % 200 == 0:
            print(f"batch {k}: {seen.sum()} / {len(files)}", flush=True)
assert seen.all(), f"{(~seen).sum()} images missing"
np.save(OUT, out)
print("FMOW_FEATURES_DONE", out.shape)
