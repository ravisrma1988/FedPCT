"""What are the CCT test near-duplicates of train (cos > 0.98, same site)? Outcome-blind: no model,
no method result; only class names of the matched images and the time gap to their match.
Rebuilds the seed-0 partition in memory with the same split code (nothing is saved).
Usage: python diag_cct_neardup.py
"""
import json

import numpy as np
import pandas as pd
import torch

import cct_env

cct_env.use_cct()
import splits  # noqa: E402

meta, X = cct_env.load_meta()
info0 = json.load(open(f"{cct_env.SPLIT_DIR}/time_seed0.json"))
print("seed 0 before drop:", info0.get("near_dup_before_drop"))
splits.UNK_POOL_TRAIN_RANGE = tuple(info0["unk_pool_train_range"])
s, info = splits.build(meta, 0, "time", X)
F = torch.from_numpy(X[s.row.to_numpy()]).float().cuda()
recs = []
for k, g in s.groupby("client"):
    a, b = g.index[g.part == "train"], g.index[g.part == "test"]
    if len(a) == 0 or len(b) == 0:
        continue
    sim = F[b] @ F[a].T
    mx, arg = sim.max(1)
    hit = (mx > 0.98).cpu().numpy()
    j = a[arg.cpu().numpy()]
    for bi, ji, h in zip(b, j, hit):
        if h:
            recs.append((s.at[bi, "y"], s.at[ji, "y"], s.at[bi, "datetime"] - s.at[ji, "datetime"], s.at[bi, "seq_id"]))
d = pd.DataFrame(recs, columns=["y_test", "y_train", "gap", "seq"])
names = meta.drop_duplicates("y").set_index("y").class_name
n_test = int((s.part == "test").sum())
print(f"test images {n_test}; near-dup of train {len(d)} ({100 * len(d) / n_test:.1f}%) in {d.seq.nunique()} sequences "
      f"of {s[s.part == 'test'].seq_id.nunique()}")
print("class of near-dup test images:", d.y_test.map(names).value_counts().head(8).to_dict())
print("share empty:", round(float((d.y_test.map(names) == "empty").mean()), 3),
      "| same class as matched train image:", round(float((d.y_test == d.y_train).mean()), 3))
print("time gap to matched train image (days): quantiles 0/5/25/50/75%:",
      [round(d.gap.quantile(q).total_seconds() / 86400, 1) for q in (0, .05, .25, .5, .75)])
te = s[s.part == "test"]
print("test class mix overall (top 6):", te.y.map(names).value_counts(normalize=True).round(3).head(6).to_dict())
