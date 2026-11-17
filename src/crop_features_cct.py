"""CCT: BioCLIP-2 features of MegaDetector animal crops, same rule as crop_features.py (iWildCam):
best animal box (category 1) with conf >= CONF, MARGIN on each side, square in pixels, clipped;
otherwise the full image's centre square. Boxes: LILA's published MDv5a results (label-free).
Rows follow features/cct_meta.csv. Label-free: species labels are not read here.
Outputs: features/cct_bioclip2crop.npy (N x 768 float16, L2-normalised), features/md_best_cct.csv
Usage: python crop_features_cct.py
"""
import json

import numpy as np
import pandas as pd
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from PIL import Image

import vit_split as vs

ROOT = "${FEDPCT_ROOT}"
IMG = f"{ROOT}/data/cct/images/cct_images"
CONF, MARGIN, BATCH = 0.2, 0.10, 128   # identical to crop_features.py

meta = pd.read_csv(f"{ROOT}/features/cct_meta.csv", usecols=["image_id", "filename"])
md = json.load(open(f"{ROOT}/data/cct/caltech-camera-traps_mdv5a.0.0_results.json"))
assert md["detection_categories"]["1"] == "animal"
det = {im["file"].split("/")[-1]: im.get("detections") or [] for im in md["images"]}
missing = [f for f in meta.filename if f not in det]
assert not missing, f"{len(missing)} images without MegaDetector output"

rows = []
for r, f in enumerate(meta.filename):
    W, H = Image.open(f"{IMG}/{f}").size  # header only
    an = [d for d in det[f] if d["category"] == "1"]
    b = max(an, key=lambda d: d["conf"]) if an else None
    if b is not None and b["conf"] >= CONF:
        x, y, w, h = b["bbox"]
        x0, y0, x1, y1 = (x - MARGIN * w) * W, (y - MARGIN * h) * H, (x + w + MARGIN * w) * W, (y + h + MARGIN * h) * H
        side = min(max(x1 - x0, y1 - y0), W, H)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0 = float(np.clip(cx - side / 2, 0, W - side)); y0 = float(np.clip(cy - side / 2, 0, H - side))
        has = True
    else:
        side = min(W, H); x0, y0 = (W - side) / 2, (H - side) / 2
        has = False
    rows.append((r, has, b["conf"] if b else 0.0, x0 / W, y0 / H, side / W, side / H))
box = pd.DataFrame(rows, columns=["row", "has_animal", "conf", "ax", "ay", "sx", "sy"])
box.to_csv(f"{ROOT}/features/md_best_cct.csv", index=False)
print(f"animal crops: {box.has_animal.mean():.3f} of {len(box)} images", flush=True)

v, size, mean, std = vs.load()
files = [f"{IMG}/{f}" for f in meta.filename]
anchors = box[["ax", "ay"]].to_numpy(np.float32)
shapes = box[["sx", "sy"]].to_numpy(np.float32)


def source(info):
    # wrap instead of StopIteration (DALI drops a partial final batch); the iterator trims padding
    i = info.idx_in_epoch % len(files)
    return (np.fromfile(files[i], dtype=np.uint8), anchors[i].copy(), shapes[i].copy(), np.array([i], np.int64))


@pipeline_def(batch_size=BATCH, num_threads=16, device_id=0, prefetch_queue_depth=4)
def pipe():
    jpg, anc, shp, idx = fn.external_source(source=source, num_outputs=4, batch=False, parallel=False)
    img = fn.decoders.image_slice(jpg, anc, shp, device="mixed", output_type=types.RGB,
                                  normalized_anchor=True, normalized_shape=True, axis_names="WH")
    img = fn.resize(img, size=(size, size), interp_type=types.INTERP_CUBIC, antialias=True)
    img = fn.crop_mirror_normalize(img, dtype=types.FLOAT, mean=[255 * m for m in mean],
                                   std=[255 * s for s in std], output_layout="CHW")
    return img, idx


p = pipe()
p.build()
it = DALIGenericIterator(p, ["img", "idx"], size=len(files), last_batch_policy=LastBatchPolicy.PARTIAL,
                         auto_reset=False)
out = np.zeros((len(files), 768), np.float16)
seen = np.zeros(len(files), bool)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for k, b in enumerate(it):
        i = b[0]["idx"].squeeze(-1).cpu().numpy()
        f = torch.nn.functional.normalize(v(b[0]["img"]).float(), dim=-1)
        out[i] = f.cpu().numpy().astype(np.float16)
        seen[i] = True
        if k % 200 == 0:
            print(f"{seen.sum()}/{len(files)}", flush=True)
assert seen.all(), f"{(~seen).sum()} rows not encoded"
assert np.isfinite(out.astype(np.float32)).all()
np.save(f"{ROOT}/features/cct_bioclip2crop.npy", out)
print("CCT_CROP_FEATURES_DONE", flush=True)
