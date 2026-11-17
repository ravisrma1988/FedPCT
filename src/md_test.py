"""MegaDetector smoke test: API, throughput and agreement between inference sizes 1280 and 640
on 200 iWildCam images (run inside .venv_md)."""
import inspect
import time

import numpy as np
import pandas as pd
from megadetector.detection.run_detector import load_detector
from PIL import Image

ROOT = "${FEDPCT_ROOT}"
det = load_detector(f"{ROOT}/models/md_v5a.0.0.pt")
print("generate_detections_one_image", inspect.signature(det.generate_detections_one_image))
meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv").sample(200, random_state=0)
imgs = [Image.open(f"{ROOT}/data/iwildcam_v2.0/train/{f}").convert("RGB") for f in meta.filename]
res = {}
for size in (1280, 640):
    t = time.time()
    out = [det.generate_detections_one_image(im, str(i), detection_threshold=0.1, image_size=size)
           for i, im in enumerate(imgs)]
    dt = time.time() - t
    top = [max((d["conf"] for d in o.get("detections", []) if d["category"] == "1"), default=0.0) for o in out]
    res[size] = np.array(top)
    print(f"size {size}: {len(imgs) / dt:.1f} img/s; images with animal conf>=0.2: {(np.array(top) >= 0.2).mean():.2f}")
agree = ((res[1280] >= 0.2) == (res[640] >= 0.2)).mean()
print(f"animal/no-animal agreement 1280 vs 640 at conf 0.2: {agree:.3f}")
empty = meta.y.to_numpy() == 0
print(f"label 'empty' share {empty.mean():.2f}; animal detected (640) on empty: {(res[640][empty] >= 0.2).mean():.2f}, "
      f"on non-empty: {(res[640][~empty] >= 0.2).mean():.2f}")
print("example output:", out[0])
