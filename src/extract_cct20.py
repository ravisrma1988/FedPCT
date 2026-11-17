"""Extract frozen BioCLIP image features for all CCT20 images (once; reused by every experiment).

Outputs in features/:
  cct20_bioclip.npy   (N, 512) float16, L2-normalised
  cct20_meta.csv      one row per image: file_name, location, seq_id, split, label (single-label
                      images only; multi-species images are dropped and counted in the log)
"""
import collections
import glob
import json
import os

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = "${FEDPCT_ROOT}"
DATA = f"{ROOT}/data/cct20"


def load_meta():
    rows, dropped = [], 0
    for path in sorted(glob.glob(f"{DATA}/eccv_18_annotation_files/*.json")):
        split = os.path.basename(path).replace("_annotations.json", "")
        d = json.load(open(path))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        labels = collections.defaultdict(set)
        for a in d["annotations"]:
            labels[a["image_id"]].add(cats[a["category_id"]])
        for im in d["images"]:
            lab = labels.get(im["id"], set())
            if len(lab) != 1:
                dropped += 1
                continue
            rows.append(dict(file_name=im["file_name"], location=str(im["location"]),
                             seq_id=im["seq_id"], split=split, label=lab.pop()))
    print(f"kept {len(rows)} single-label images, dropped {dropped} (multi/no label)")
    return pd.DataFrame(rows)


class Images(Dataset):
    def __init__(self, files, img_dir, tf):
        self.files, self.img_dir, self.tf = files, img_dir, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(os.path.join(self.img_dir, self.files[i])).convert("RGB"))


def main():
    meta = load_meta()
    img_dir = os.path.dirname(glob.glob(f"{DATA}/**/{meta.file_name.iloc[0]}", recursive=True)[0])
    missing = [f for f in meta.file_name if not os.path.exists(os.path.join(img_dir, f))]
    print(f"image dir {img_dir}; missing files: {len(missing)}")
    meta = meta[~meta.file_name.isin(set(missing))].reset_index(drop=True)

    model, _, tf = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    model = model.cuda().eval()
    dl = DataLoader(Images(list(meta.file_name), img_dir, tf), batch_size=512,
                    num_workers=24, pin_memory=True)
    feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, x in enumerate(dl):
            f = model.encode_image(x.cuda(non_blocking=True))
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).half().cpu())
            if i % 20 == 0:
                print(f"batch {i}/{len(dl)}", flush=True)
    os.makedirs(f"{ROOT}/features", exist_ok=True)
    np.save(f"{ROOT}/features/cct20_bioclip.npy", torch.cat(feats).numpy())
    meta.to_csv(f"{ROOT}/features/cct20_meta.csv", index=False)
    print("saved", len(meta), "features")
    print(meta.groupby("location").label.nunique().to_string())


if __name__ == "__main__":
    main()
