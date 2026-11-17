"""BioCLIP-2 features of MegaDetector animal crops (GPU decode + crop with DALI).

Rule (fixed before results): best animal box (category 1) with conf >= CONF -> expand by MARGIN on
each side, make square in pixel space, clip to the image; otherwise the full image's centre square
(same view as the earlier whole-image features). Both resized to the model's input size.
Outputs: features/iwildcam_bioclip2crop.npy (N x 768 float16, L2-normalised, meta row order),
         features/md_best.csv (row, has_animal, conf, box used).
Usage: python crop_features.py
"""
import glob
import json

import numpy as np
import pandas as pd
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from PIL import Image

import vit_split as vs

ROOT = "${FEDPCT_ROOT}"
IMG = f"{ROOT}/data/iwildcam_v2.0/train"
CONF, MARGIN, BATCH = 0.2, 0.10, 128

import os  # noqa: E402

TEST = int(os.environ.get("CROP_TEST", "0"))  # >0: pipeline test on the first TEST rows that MD has done
meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
best = {}
for path in glob.glob(f"{ROOT}/results/md/shard*.jsonl"):
    for line in open(path):
        d = json.loads(line)
        an = [x for x in d["detections"] if x["category"] == "1"]
        best[d["row"]] = max(an, key=lambda x: x["conf"]) if an else None
test_rows = sorted(best)[:TEST]
if TEST:
    meta = meta.loc[test_rows].reset_index(drop=True)
    best = {i: best[r] for i, r in enumerate(test_rows)}
assert len(best) == len(meta), f"MegaDetector output covers {len(best)} of {len(meta)} rows"

rows = []
for r, f in enumerate(meta.filename):
    W, H = Image.open(f"{IMG}/{f}").size  # header only
    b = best[r]
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
if not TEST:
    box.to_csv(f"{ROOT}/features/md_best.csv", index=False)
print(f"animal crops: {box.has_animal.mean():.3f} of {len(box)} images", flush=True)

v, size, mean, std = vs.load()
files = [f"{IMG}/{f}" for f in meta.filename]
anchors = box[["ax", "ay"]].to_numpy(np.float32)
shapes = box[["sx", "sy"]].to_numpy(np.float32)


def source(info):
    # wrap instead of StopIteration: DALI drops a partial final batch on StopIteration; the
    # iterator's size + LastBatchPolicy.PARTIAL trims the padded samples instead
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
if TEST:
    # crops of animal images should differ from the whole-image features; centre-square views
    # (no detection) should be close to them (same view, DALI vs DALI)
    ref = np.load(f"{ROOT}/features/iwildcam_bioclip2.npy")[test_rows]
    cos = (out.astype(np.float32) * ref.astype(np.float32)).sum(1)
    print(f"TEST ok: {TEST} rows; cos to whole-image features: no-animal {cos[~box.has_animal].mean():.3f}, "
          f"animal crops {cos[box.has_animal.to_numpy()].mean():.3f}", flush=True)
else:
    np.save(f"{ROOT}/features/iwildcam_bioclip2crop.npy", out)
    print("CROP_FEATURES_DONE", flush=True)
