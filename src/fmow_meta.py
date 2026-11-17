"""FMoW-WILDS v1.1 metadata table for the shared split code (label-free rules only).

Rules (fixed 2026-11-15, before any FMoW feature, model or result):
  * site      = country (country_code); the split code keeps sites with >= 250 images
  * sequence  = object instance (the img_path folder, e.g. train/airport/airport_12). All views of an
                object stay in one partition, and the per-site time split orders objects by their
                FIRST view. This prevents the main leakage path of FMoW: 67% of test images would
                otherwise have another view of the same object in the same country's train partition.
  * time      = image timestamp (UTC)
  * seen / unseen sites: FMoW has no site-level split, so 25% of the countries with >= 250 images are
                drawn as unseen with rng 20261115 (stratified by the UN region of country_code_mapping.csv;
                countries without a region form their own stratum). Unseen sites contribute no training data.
  * image file: images/rgb_img_<row>.png, row = position in rgb_metadata.csv (WILDS convention)
Output: features/fmow_meta.csv, reports/fmow_meta_report.json
Usage: python -I fmow_meta.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "${FEDPCT_ROOT}"
DATA = f"{ROOT}/data/fmow"
OUT, REP = f"{ROOT}/features/fmow_meta.csv", f"{ROOT}/reports/fmow_meta_report.json"
MIN_IMAGES, UNSEEN_SHARE, RNG = 250, 0.25, 20261115
assert not os.path.exists(OUT), "FMoW meta exists; never rebuild"

m = pd.read_csv(f"{DATA}/rgb_metadata.csv")
m["row"] = np.arange(len(m))
m["datetime"] = pd.to_datetime(m.timestamp, format="ISO8601").dt.tz_convert(None)
obj = m.img_path.str.rsplit("/", n=1).str[0]
m["seq_id"] = pd.factorize(obj)[0]
assert (m.groupby("seq_id").country_code.nunique() == 1).all(), "an object spans two countries"
countries = sorted(m.country_code.unique())
m["location_remapped"] = m.country_code.map({c: i for i, c in enumerate(countries)})
cats = sorted(m.category.unique())
m["y"] = m.category.map({c: i for i, c in enumerate(cats)})
m["image_id"] = m.row
m["filename"] = [f"rgb_img_{r}.png" for r in m.row]
assert all(os.path.exists(f"{DATA}/images/{f}") for f in m.filename.iloc[[0, len(m) // 2, len(m) - 1]])

reg = pd.read_csv(f"{DATA}/country_code_mapping.csv", encoding="latin-1", keep_default_na=False)
region = dict(zip(reg["alpha-3"], reg["region"].replace("", "Other")))
n = m.groupby("country_code").size()
sites = sorted(n[n >= MIN_IMAGES].index)
rng = np.random.default_rng(RNG)
unseen = []
strata = pd.Series({c: region.get(c, "Other") or "Other" for c in sites})
for r, cs in strata.groupby(strata):
    k = int(round(UNSEEN_SHARE * len(cs)))
    unseen += rng.choice(sorted(cs.index), k, replace=False).tolist() if k else []
m["split"] = np.where(m.country_code.isin(unseen), "test", "train")   # split code: seen = any 'train' row

m[["row", "image_id", "filename", "y", "category", "location_remapped", "country_code", "seq_id",
   "datetime", "split"]].to_csv(OUT, index=False)
rep = dict(created="2026-11-15", rows=len(m), classes=len(cats), countries=len(countries),
           sites_ge_250=len(sites), unseen_sites=sorted(unseen), n_unseen=len(unseen),
           strata={r: int((strata == r).sum()) for r in sorted(strata.unique())},
           objects=int(m.seq_id.nunique()), years=[int(m.datetime.dt.year.min()), int(m.datetime.dt.year.max())],
           class_names=cats, country_codes=countries)
json.dump(rep, open(REP, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in ("class_names", "country_codes")}))
print("FMOW_META_DONE")
