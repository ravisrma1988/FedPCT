"""Are CCT train/test near-duplicates (crop cos > 0.98) re-saved copies of the same photo, or real
recurrences? Compares full-frame perceptual hashes (64-bit dHash, 9x8 grey) of matched pairs with
random same-site, same-class train/test pairs. Outcome-blind (no model, no method result).
Usage: python diag_cct_dupcheck.py
"""
import json

import numpy as np
import torch
from PIL import Image

import cct_env

cct_env.use_cct()
import splits  # noqa: E402

IMG = f"{cct_env.ROOT}/data/cct/images/cct_images"
meta, X = cct_env.load_meta()
info0 = json.load(open(f"{cct_env.SPLIT_DIR}/time_seed0.json"))
splits.UNK_POOL_TRAIN_RANGE = tuple(info0["unk_pool_train_range"])
s, _ = splits.build(meta, 0, "time", X)
F = torch.from_numpy(X[s.row.to_numpy()]).float().cuda()
rng = np.random.default_rng(0)


def dhash(f):
    g = np.asarray(Image.open(f"{IMG}/{f}").convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


pairs, rand = [], []
for k, g in s.groupby("client"):
    a, b = g.index[g.part == "train"], g.index[g.part == "test"]
    if len(a) == 0 or len(b) == 0:
        continue
    mx, arg = (F[b] @ F[a].T).max(1)
    hit = np.flatnonzero(mx.cpu().numpy() > 0.98)
    for i in rng.choice(hit, min(len(hit), 15), replace=False) if len(hit) else []:
        bi, ai = b[i], a[arg[i].item()]
        same = a[(s.loc[a, "y"] == s.at[bi, "y"]).to_numpy()]
        if len(same) == 0:  # e.g. a held-out unknown species: no same-class train image
            continue
        pairs.append((s.at[bi, "filename"], s.at[ai, "filename"]))
        rand.append((s.at[bi, "filename"], s.at[rng.choice(same), "filename"]))
d_hit = np.array([(dhash(x) != dhash(y)).sum() for x, y in pairs])
d_rand = np.array([(dhash(x) != dhash(y)).sum() for x, y in rand])
print(f"{len(pairs)} near-dup pairs: dHash Hamming distance median {np.median(d_hit):.0f}, "
      f"share <= 4 (same photo) {np.mean(d_hit <= 4):.2f}, share <= 10 {np.mean(d_hit <= 10):.2f}")
print(f"{len(rand)} random same-site same-class pairs: median {np.median(d_rand):.0f}, "
      f"share <= 4 {np.mean(d_rand <= 4):.2f}, share <= 10 {np.mean(d_rand <= 10):.2f}")
