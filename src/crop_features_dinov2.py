"""DINOv2 ViT-L/14 features of the SAME MegaDetector crops as the BioCLIP-2 features
(PREREG_BACKBONE.md): crop geometry read from md_best*.csv, 224x224 input, ImageNet mean/std,
pooled CLS output (pooler_output, 1024-d), L2-normalised. Label-free.
Usage: python crop_features_dinov2.py iwildcam|cct
"""
import sys

import numpy as np
import pandas as pd
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from transformers import Dinov2Model

ROOT = "${FEDPCT_ROOT}"
MODEL, SIZE, BATCH = "facebook/dinov2-large", 224, 128
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ds = sys.argv[1]
if ds == "iwildcam":
    meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv", usecols=["filename"])
    img, box_csv, out_path = f"{ROOT}/data/iwildcam_v2.0/train", f"{ROOT}/features/md_best.csv", \
        f"{ROOT}/features/iwildcam_dinov2crop.npy"
else:
    meta = pd.read_csv(f"{ROOT}/features/cct_meta.csv", usecols=["filename"])
    img, box_csv, out_path = f"{ROOT}/data/cct/images/cct_images", f"{ROOT}/features/md_best_cct.csv", \
        f"{ROOT}/features/cct_dinov2crop.npy"
box = pd.read_csv(box_csv)
assert len(box) == len(meta) and (box.row.to_numpy() == np.arange(len(meta))).all()
files = [f"{img}/{f}" for f in meta.filename]
anchors = box[["ax", "ay"]].to_numpy(np.float32)
shapes = box[["sx", "sy"]].to_numpy(np.float32)
model = Dinov2Model.from_pretrained(MODEL).cuda().eval()


def source(info):
    i = info.idx_in_epoch % len(files)  # wrap: DALI drops a partial batch on StopIteration
    return (np.fromfile(files[i], dtype=np.uint8), anchors[i].copy(), shapes[i].copy(), np.array([i], np.int64))


@pipeline_def(batch_size=BATCH, num_threads=16, device_id=0, prefetch_queue_depth=4)
def pipe():
    jpg, anc, shp, idx = fn.external_source(source=source, num_outputs=4, batch=False, parallel=False)
    x = fn.decoders.image_slice(jpg, anc, shp, device="mixed", output_type=types.RGB,
                                normalized_anchor=True, normalized_shape=True, axis_names="WH")
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
            print(f"{seen.sum()}/{len(files)}", flush=True)
assert seen.all(), f"{(~seen).sum()} rows not encoded"
assert np.isfinite(out.astype(np.float32)).all()
np.save(out_path, out)
print(f"DINOV2_DONE {ds}", flush=True)
