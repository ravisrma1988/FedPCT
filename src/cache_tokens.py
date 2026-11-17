"""Cache BioCLIP-2 block-22 tokens for every train-partition row of the time splits (union over
seeds), ordered by client. Training of the top blocks then never touches JPEGs or the trunk.

Outputs: features/tokens_b22.f16 (memmap, (N, L, D) float16), features/tokens_b22_index.csv
(cache position -> meta row, client). Checks on the first batch that trunk+Top == full encoder.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

import vit_split as vs
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
IMG_DIR = f"{ROOT}/data/iwildcam_v2.0/train"

parts = [load_split("time", s)[0] for s in range(5)]
rows = pd.concat([p.loc[p.part == "train", ["row", "client"]] for p in parts]).drop_duplicates("row")
rows = rows.sort_values(["client", "row"]).reset_index(drop=True)
meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
files = [f"{IMG_DIR}/{f}" for f in meta.filename.to_numpy()[rows.row.to_numpy()]]

v, size, mean, std = vs.load()
top = vs.Top(v).cuda().eval()
with torch.no_grad():
    probe = vs.trunk(v, torch.zeros(1, 3, size, size, device="cuda"))
L, D = probe.shape[1], probe.shape[2]
print(f"rows {len(rows)}  tokens {L}x{D}  -> {len(rows) * L * D * 2 / 2**30:.1f} GB", flush=True)

mm = np.lib.format.open_memmap(f"{ROOT}/features/tokens_b22.npy", mode="w+", dtype=np.float16,
                               shape=(len(rows), L, D))
it = vs.dali_iter(files, size, mean, std, batch=128)
t0, checked = time.time(), False
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i, b in enumerate(it):
        x, idx = b[0]["img"], b[0]["idx"].squeeze(-1).long().cpu().numpy()
        tok = vs.trunk(v, x)
        if not checked:
            full = torch.nn.functional.normalize(v(x).float(), dim=-1)
            split = top(tok).float()
            cos = (full * split).sum(-1)
            print(f"split check: cos(full, trunk+top) min {cos.min():.5f}", flush=True)
            assert cos.min() > 0.999
            checked = True
        mm[idx] = tok.to(torch.float16).cpu().numpy()
        if i % 50 == 0:
            done = min((i + 1) * 128, len(rows))
            print(f"{done}/{len(rows)}  {done / (time.time() - t0):.0f} img/s", flush=True)
mm.flush()
rows.to_csv(f"{ROOT}/features/tokens_b22_index.csv", index_label="pos")
json.dump(dict(model=vs.MODEL, cut=vs.CUT, n=len(rows), L=L, D=D, seconds=round(time.time() - t0)),
          open(f"{ROOT}/features/tokens_b22.json", "w"))
print("CACHE_DONE", flush=True)
