"""Label-free structure of FMoW-WILDS v1.1 before any protocol decision (no model, no features).
Reports: rows per WILDS split, countries with >= 250 images, per-country stream lengths and time
spans, object instances (img_path folder) and how often one object's views would fall into more
than one partition under the per-site 40/20/40 time split.
"""
import numpy as np
import pandas as pd

ROOT = "${FEDPCT_ROOT}/data/fmow"
m = pd.read_csv(f"{ROOT}/rgb_metadata.csv")
m["row"] = np.arange(len(m))
m["t"] = pd.to_datetime(m.timestamp, format="ISO8601")
m["obj"] = m.img_path.str.rsplit("/", n=1).str[0]
print("rows", len(m), "| split counts", m.split.value_counts().to_dict())
print("missing country", m.country_code.isna().sum(), "| classes", m.category.nunique())
print("years", m.t.dt.year.min(), "-", m.t.dt.year.max())
v = m.groupby("obj").size()
print("objects", len(v), "| views per object: median", v.median(), "mean", round(v.mean(), 2), "max", v.max())
print("objects spanning >1 country", (m.groupby("obj").country_code.nunique() > 1).sum())
n = m.groupby("country_code").size().sort_values(ascending=False)
keep = n[n >= 250].index
print(f"countries {len(n)}; with >=250 images {len(keep)}, covering {n[keep].sum()} images "
      f"({100 * n[keep].sum() / len(m):.1f}%)")
print("images per kept country: quantiles", np.percentile(n[keep], [0, 10, 25, 50, 75, 90, 100]).astype(int).tolist())
k = m[m.country_code.isin(keep)].copy()
k["q"] = k.groupby("country_code").t.rank(method="first", pct=True)
k["part"] = np.where(k.q <= 0.4, "train", np.where(k.q <= 0.6, "cal", "test"))
span = k.groupby("country_code").t.agg(lambda x: (x.max() - x.min()).days / 365.25)
print("time span per country (years): quantiles", np.round(np.percentile(span, [0, 25, 50, 75, 100]), 1).tolist())
cut = k.groupby(["country_code", "part"]).t.agg(["min", "max"]).unstack()
print("median year of test-period start:", cut[("min", "test")].dt.year.median())
po = k.groupby("obj").part.nunique()
print(f"objects in >1 partition: {(po > 1).mean() * 100:.1f}% of objects")
tst = k[k.part == "test"]
tr_objs = set(k.loc[k.part == "train", "obj"])
print(f"test images whose object has a view in the same country's train partition: "
      f"{100 * tst.obj.isin(tr_objs).mean():.1f}%")
first = k.groupby("obj").t.transform("min")
k["q_obj"] = k.assign(t0=first).groupby("country_code").t0.rank(method="dense", pct=True)
po2 = k.assign(part2=np.where(k.q_obj <= 0.4, "train", np.where(k.q_obj <= 0.6, "cal", "test")))
print("if whole objects are partitioned by their first view: test share",
      f"{100 * (po2.part2 == 'test').mean():.1f}%, objects per country median",
      int(k.groupby("country_code").obj.nunique().median()))
print("objects in >1 split folder? n/a; time span of one object (years) quantiles", np.round(np.percentile(k.groupby("obj").t.agg(lambda x: (x.max() - x.min()).days / 365.25), [50, 75, 90, 99]), 1).tolist())
print("top countries", n.head(12).to_dict())
