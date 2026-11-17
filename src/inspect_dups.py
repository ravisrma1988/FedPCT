"""Inspect images whose BioCLIP features are identical: are the files identical too?"""
import hashlib

import numpy as np
import pandas as pd

ROOT = "${FEDPCT_ROOT}"
meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
X = np.load(f"{ROOT}/features/iwildcam_bioclip.npy")
meta["fh"] = [hashlib.md5(v.tobytes()).hexdigest() for v in X]
g = meta.groupby("fh")
dup = meta[meta.fh.isin(g.size()[g.size() > 1].index)].copy()
print(f"feature-duplicate groups: {dup.fh.nunique()}, images involved: {len(dup)}")


def file_md5(f):
    return hashlib.md5(open(f"{ROOT}/data/iwildcam_v2.0/train/{f}", "rb").read()).hexdigest()


dup["file_md5"] = dup.filename.map(file_md5)
summ = dup.groupby("fh").agg(n=("y", "size"), files_identical=("file_md5", lambda s: s.nunique() == 1),
                             same_location=("location_remapped", lambda s: s.nunique() == 1),
                             same_seq=("seq_id", lambda s: s.nunique() == 1),
                             same_label=("y", lambda s: s.nunique() == 1),
                             wilds_splits=("split", lambda s: ",".join(sorted(set(s)))),
                             times=("datetime", lambda s: s.nunique()))
print(summ.describe(include="all").T[["count", "unique", "top", "freq"]].to_string())
print("\nexamples:")
print(dup.sort_values("fh")[["fh", "filename", "location_remapped", "seq_id", "datetime", "y", "split"]]
      .head(12).to_string(index=False))
