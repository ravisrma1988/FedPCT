"""CCT metadata table for the shared split code, built with label-free integrity checks FIRST.
Nothing here looks at results; species labels are only used to drop multi-label photos.

Rules (fixed before any CCT result):
  * multi-category images (several species in one photo) are dropped (single-label protocol)
  * byte-identical files: one copy kept; if the copies disagree on the label, all are dropped
  * sequences whose frames carry more than one location are dropped (cannot be assigned to a site)
  * timestamps: rows with unparseable dates, or dates outside [2008-01-01, 2019-12-31] (CCT collection
    period with margin), are dropped; a sequence whose frames span more than 1 day is dropped
    (broken camera clock / merged sequences), since time order is the basis of the split
  * seen / unseen sites: CCT's official location-level train / val split (CaltechCameraTrapsSplits_v0)
  * the 20 CCT20 pilot locations are recorded (they must stay out of the CCT lockbox)
Output: features/cct_meta.csv (rows in the order of the feature file), reports/cct_meta_report.json
Usage: python -I cct_meta.py
"""
import collections
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = "${FEDPCT_ROOT}"
D = f"{ROOT}/data/cct"
IMG = f"{D}/images/cct_images"
os.makedirs(f"{ROOT}/reports", exist_ok=True)
meta = json.load(open(f"{D}/caltech_images_20210113.json"))
splits = json.load(open(f"{D}/CaltechCameraTrapsSplits_v0.json"))
rep = {}

cname = {c["id"]: c["name"] for c in meta["categories"]}
cats = collections.defaultdict(set)
for a in meta["annotations"]:
    cats[a["image_id"]].add(a["category_id"])
df = pd.DataFrame(meta["images"])
df["ncat"] = df.id.map(lambda i: len(cats[i]))
df["category_id"] = df.id.map(lambda i: min(cats[i]))
rep["images_in"] = len(df)
rep["multi_category_dropped"] = int((df.ncat > 1).sum())
df = df[df.ncat == 1].copy()

exists = df.file_name.map(lambda f: os.path.exists(f"{IMG}/{f}"))
rep["missing_files"] = int((~exists).sum())
assert exists.all(), f"{(~exists).sum()} image files missing"


def md5(f):
    h = hashlib.md5()
    with open(f"{IMG}/{f}", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


with ThreadPoolExecutor(32) as ex:
    df["md5"] = list(ex.map(md5, df.file_name, chunksize=256))
nlab = df.groupby("md5").category_id.transform("nunique")
dup_conflict = nlab > 1
dup_extra = ~dup_conflict & df.md5.duplicated(keep="first")
rep["byte_duplicates_conflicting_labels_dropped"] = int(dup_conflict.sum())
rep["byte_duplicates_extra_copies_dropped"] = int(dup_extra.sum())
df = df[~dup_conflict & ~dup_extra].copy()

nloc = df.groupby("seq_id").location.transform("nunique")
rep["rows_in_multi_location_sequences_dropped"] = int((nloc > 1).sum())
df = df[nloc == 1].copy()

df["datetime"] = pd.to_datetime(df.date_captured, errors="coerce")
bad = df.datetime.isna() | (df.datetime < "2008-01-01") | (df.datetime > "2019-12-31")
rep["bad_or_out_of_range_dates_dropped"] = int(bad.sum())
rep["year_counts_before_filter"] = {str(k): int(v) for k, v in df.datetime.dt.year.value_counts().sort_index().items()}
df = df[~bad].copy()
span = df.groupby("seq_id").datetime.transform(lambda t: t.max() - t.min())
long_seq = span > pd.Timedelta(days=1)
rep["rows_in_sequences_spanning_over_1_day_dropped"] = int(long_seq.sum())
df = df[~long_seq].copy()

train_locs = {str(l) for l in splits["splits"]["train"]}
val_locs = {str(l) for l in splits["splits"]["val"]}
assert not train_locs & val_locs, "official split has overlapping locations"
df["split"] = np.where(df.location.astype(str).isin(train_locs), "train",
                       np.where(df.location.astype(str).isin(val_locs), "val", "none"))
rep["rows_with_location_in_neither_official_split"] = int((df.split == "none").sum())

c20 = set()
for f in os.listdir(f"{ROOT}/data/cct20/eccv_18_annotation_files"):
    if f.endswith(".json"):
        c20 |= {str(im["location"]) for im in json.load(open(f"{ROOT}/data/cct20/eccv_18_annotation_files/{f}"))["images"]}
rep["cct20_pilot_locations"] = sorted(c20, key=int)

df["location_remapped"] = df.location.astype(int)
df["y"] = df.category_id.astype(int)
df["class_name"] = df.y.map(cname)
out = df.rename(columns={"id": "image_id", "file_name": "filename"})[
    ["image_id", "filename", "y", "class_name", "location_remapped", "seq_id", "datetime", "split", "md5"]
].reset_index(drop=True)
out.to_csv(f"{ROOT}/features/cct_meta.csv", index=False)
rep["rows_out"] = len(out)
rep["locations_out"] = int(out.location_remapped.nunique())
rep["locations_ge_250"] = int((out.groupby("location_remapped").size() >= 250).sum())
rep["seen_locations_ge_250"] = int((out[out.split == "train"].groupby("location_remapped").size() >= 250).sum())
rep["unseen_locations_ge_250"] = int((out[out.split == "val"].groupby("location_remapped").size() >= 250).sum())
json.dump(rep, open(f"{ROOT}/reports/cct_meta_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
