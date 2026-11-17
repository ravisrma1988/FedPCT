"""MegaDetector v5a over iWildCam (rows of iwildcam_meta.csv), shard i of n, size 1280.
Keeps all detections with conf >= 0.05 (category 1 animal, 2 person, 3 vehicle), so thresholds can
be chosen later on tune data. Image loading runs in a thread pool to keep the GPU busy.
Usage (inside .venv_md): python md_run.py <shard> <n_shards>
Output: results/md/shard<i>.jsonl  (one line per image: row, detections)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from megadetector.detection.run_detector import load_detector
from PIL import Image

ROOT = "${FEDPCT_ROOT}"
shard, n = int(sys.argv[1]), int(sys.argv[2])
meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
rows = list(range(shard, len(meta), n))
os.makedirs(f"{ROOT}/results/md", exist_ok=True)
out_path = f"{ROOT}/results/md/shard{shard}.jsonl"
done = set()
if os.path.exists(out_path):  # resume
    done = {json.loads(l)["row"] for l in open(out_path)}
rows = [r for r in rows if r not in done]
det = load_detector(f"{ROOT}/models/md_v5a.0.0.pt")
files = meta.filename.to_numpy()


def load(r):
    return r, Image.open(f"{ROOT}/data/iwildcam_v2.0/train/{files[r]}").convert("RGB")


t0 = time.time()
with open(out_path, "a") as f, ThreadPoolExecutor(4) as pool:
    for i, (r, im) in enumerate(pool.map(load, rows)):
        o = det.generate_detections_one_image(im, str(r), detection_threshold=0.05, image_size=1280)
        f.write(json.dumps(dict(row=int(r), detections=o.get("detections", []))) + "\n")
        if i % 2000 == 0:
            f.flush()
            print(f"shard {shard}: {i}/{len(rows)}  {(i + 1) / (time.time() - t0):.1f} img/s", flush=True)
print(f"MD_SHARD_DONE {shard}", flush=True)
