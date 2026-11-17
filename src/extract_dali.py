"""GPU-bound feature extraction: DALI decodes JPEGs on the GPU (nvJPEG), resizes and normalises
there, and the open_clip image tower runs in bf16. Rows follow iwildcam_meta.csv order.

Usage: python extract_dali.py <hf-hub model id> <output tag> [batch]
Outputs: features/iwildcam_<tag>.npy (N, D) float16, L2-normalised; features/iwildcam_<tag>.json
Preprocessing mirrors open_clip's eval transform (shorter side -> image size, bicubic; centre crop;
mean/std from the model's preprocess config). DALI's resize differs slightly from PIL's, so the
BioCLIP DALI features are compared with the existing PIL ones as a check.
"""
import json
import sys
import time

import numpy as np
import open_clip
import pandas as pd
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

ROOT = "${FEDPCT_ROOT}"
IMG_DIR = f"{ROOT}/data/iwildcam_v2.0/train"
MODEL, TAG = sys.argv[1], sys.argv[2]
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 256

meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
files = [f"{IMG_DIR}/{f}" for f in meta.filename]

model, _, _ = open_clip.create_model_and_transforms(MODEL)
model = model.cuda().eval()
cfg = getattr(model.visual, "preprocess_cfg", {}) or {}
size = cfg.get("size", model.visual.image_size)
size = size[0] if isinstance(size, (tuple, list)) else size
mean = cfg.get("mean", open_clip.OPENAI_DATASET_MEAN)
std = cfg.get("std", open_clip.OPENAI_DATASET_STD)
print(f"{MODEL}: image size {size}, mean {mean}, std {std}", flush=True)


@pipeline_def(batch_size=BATCH, num_threads=16, device_id=0, prefetch_queue_depth=4)
def pipe():
    jpegs, labels = fn.readers.file(files=files, labels=list(range(len(files))),
                                    random_shuffle=False, name="reader")
    img = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    img = fn.resize(img, resize_shorter=size, interp_type=types.INTERP_CUBIC, antialias=True)
    img = fn.crop_mirror_normalize(img, crop=(size, size), dtype=types.FLOAT,
                                   mean=[255 * m for m in mean], std=[255 * s for s in std],
                                   output_layout="CHW")
    return img, labels


p = pipe()
p.build()
it = DALIGenericIterator(p, ["img", "idx"], reader_name="reader",
                         last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=False)
feats = None
t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i, b in enumerate(it):
        x, idx = b[0]["img"], b[0]["idx"].squeeze(-1).long()
        f = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).half()
        if feats is None:
            feats = torch.zeros(len(files), f.shape[1], dtype=torch.float16, device="cuda")
        feats[idx.cuda()] = f
        if i % 100 == 0:
            done = min((i + 1) * BATCH, len(files))
            print(f"{done}/{len(files)}  {done / (time.time() - t0):.0f} img/s", flush=True)
F = feats.cpu().numpy()
assert np.isfinite(F).all() and (np.abs(np.linalg.norm(F.astype(np.float32), axis=1) - 1) < 1e-2).all()
np.save(f"{ROOT}/features/iwildcam_{TAG}.npy", F)
json.dump(dict(model=MODEL, size=int(size), mean=list(mean), std=list(std), n=len(F), dim=int(F.shape[1]),
               seconds=round(time.time() - t0)), open(f"{ROOT}/features/iwildcam_{TAG}.json", "w"))
print(f"saved {F.shape} in {time.time() - t0:.0f}s", flush=True)
