"""Location / class statistics for choosing the protocol-v2 thresholds."""
import numpy as np
import pandas as pd

m = pd.read_csv("${FEDPCT_ROOT}/features/iwildcam_meta.csv")
cats = pd.read_csv("${FEDPCT_ROOT}/data/iwildcam_v2.0/categories.csv")
print("categories.csv columns:", list(cats.columns), "| y=0 is:", cats.iloc[0].to_dict())

loc = m.groupby("location_remapped").agg(n=("y", "size"), n_seq=("seq_id", "nunique"),
                                         n_cls=("y", "nunique"),
                                         split=("split", lambda s: s.mode()[0]))
q = [0, .1, .25, .5, .75, .9, 1]
print("\nimages per location quantiles:", np.quantile(loc.n, q).astype(int))
for t in [100, 250, 500, 1000]:
    print(f"  locations with >= {t} images: {(loc.n >= t).sum()}  (by majority split: "
          f"{loc[loc.n >= t].split.value_counts().to_dict()})")
print("does any location appear in >1 WILDS split?",
      (m.groupby("location_remapped").split.nunique() > 1).sum(), "locations")

cls = m.groupby("y").agg(n=("y", "size"), n_loc=("location_remapped", "nunique"),
                         n_train=("split", lambda s: (s == "train").sum()))
print("\nclass sizes quantiles:", np.quantile(cls.n, q).astype(int))
print("classes with n_train < 10:", (cls.n_train < 10).sum())
for lo, hi in [(20, 200), (50, 500), (100, 1000)]:
    c = cls[(cls.n >= lo) & (cls.n <= hi) & (cls.n_loc >= 3)]
    print(f"  candidate rare unknowns: {lo}<=n<={hi} & >=3 locations -> {len(c)} classes, "
          f"{c.n.sum()} images")
print("\nlargest classes:", cls.n.nlargest(8).to_dict())
