"""Single source of truth for iWildCam data partitions (protocol v2).

Splits are built ONCE, validated with hard assertions, and saved with a SHA-256. Experiments
must call load_split(); nothing else may partition the data.

Build order (each step uses only what is allowed at that point):
  0. clean      = drop multi-species photos, dedupe identical photos (see clean())
  1. clients    = locations with >= MIN_IMAGES images      (image counts only, no labels)
  2. partition  = per site, by whole sequences              (no labels)
                  time: earliest 40% of sequences -> train, next 20% -> cal, latest 40% -> test
                  seq : random 40/20/40 over sequences (seeded)
  3. unknowns   = N_UNK species drawn from a pool defined on the TRAIN partition only
  4. train rows of unknown species are dropped; their cal/test rows are kept as UNK
Rules for later code (enforced by structure, stated here so they are not forgotten):
  * hyper-parameters (incl. FedOSCP's lambda) may be chosen on cal data of a separate
    "tune" slice only, never on test; conformal quantiles use the remaining cal data
  * label-shift weights / class proportions come from cal labels only

Usage:  python splits.py --seeds 5 --modes time seq
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = "${FEDPCT_ROOT}"
SPLIT_DIR = f"{ROOT}/splits"
MIN_IMAGES = 250
N_UNK = 8
UNK_POOL_TRAIN_RANGE = (20, 200)   # images of the species in the train partition
UNK_POOL_MIN_SITES = 3             # sites where the species occurs in the train partition
UNK = -1


def _partition_site(g, mode, seed):
    if mode == "time":
        q = g.groupby("seq_id").datetime.transform("min").rank(method="dense", pct=True).to_numpy()
        return np.where(q <= 0.4, "train", np.where(q <= 0.6, "cal", "test"))
    out = np.array(["train"] * len(g), dtype=object)
    tr, rest = next(GroupShuffleSplit(1, test_size=0.6, random_state=seed).split(g, groups=g.seq_id))
    r = g.iloc[rest]
    if r.seq_id.nunique() < 2:
        out[rest] = "test"
        return out
    cal, te = next(GroupShuffleSplit(1, test_size=2 / 3, random_state=seed + 1).split(r, groups=r.seq_id))
    out[rest[cal]], out[rest[te]] = "cal", "test"
    return out


def clean(meta, X):
    """Deduplicate before any splitting. iWildCam stores a multi-species photo as several rows
    (one per species, different filenames, identical pixels). Groups of rows with identical
    features are treated as one photo:
      * labels disagree -> multi-species photo -> dropped (single-label protocol, as for CCT20)
      * labels agree    -> exact duplicate      -> one row kept
    """
    h = pd.Series([hashlib.md5(v.tobytes()).hexdigest() for v in X], index=meta.index)
    nlab = meta.y.groupby(h).transform("nunique")
    multi = nlab > 1
    dup_extra = ~multi & h.duplicated(keep="first")
    report = dict(rows_in=len(meta), multi_species_rows_dropped=int(multi.sum()),
                  multi_species_photos=int(h[multi].nunique()),
                  exact_duplicate_rows_dropped=int(dup_extra.sum()))
    out = meta[~multi & ~dup_extra]
    report["rows_out"] = len(out)
    return out, report


def build(meta, seed, mode, X=None):
    rng = np.random.default_rng(seed)
    meta, clean_report = clean(meta, X)
    n_img = meta.groupby("location_remapped").size()
    locs = sorted(n_img[n_img >= MIN_IMAGES].index)
    m = meta[meta.location_remapped.isin(locs)].copy()
    m["row"] = m.index
    m["client"] = m.location_remapped.map({l: k for k, l in enumerate(locs)})
    m["datetime"] = pd.to_datetime(m.datetime)
    m["part"] = ""
    for l, g in m.groupby("location_remapped"):
        m.loc[g.index, "part"] = _partition_site(g, mode, seed * 1000 + int(l))
    # Carve a "tune" slice (1/3 of each site's cal sequences) for hyper-parameter selection.
    # Methods without hyper-parameters calibrate on tune+cal, so all methods see the same data.
    cal_seqs = m.loc[m.part == "cal", "seq_id"].unique()
    tune_seqs = set(rng.choice(cal_seqs, len(cal_seqs) // 3, replace=False))
    m.loc[(m.part == "cal") & m.seq_id.isin(tune_seqs), "part"] = "tune"
    seen_locs = set(meta.loc[meta.split == "train", "location_remapped"])
    m["seen"] = m.location_remapped.isin(seen_locs)

    # Unknown pool from the train partition of seen sites only.
    trn = m[(m.part == "train") & m.seen]
    st = trn.groupby("y").agg(n=("y", "size"), sites=("client", "nunique"))
    pool = st[(st.n >= UNK_POOL_TRAIN_RANGE[0]) & (st.n <= UNK_POOL_TRAIN_RANGE[1])
              & (st.sites >= UNK_POOL_MIN_SITES)].index
    unk = sorted(int(u) for u in rng.choice(pool, N_UNK, replace=False))
    # Known label space = classes present in the (unknown-free) train partition of seen sites.
    known = sorted(int(c) for c in set(trn.y) - set(unk))
    cidx = {c: i for i, c in enumerate(known)}
    # Unknowns: "designed" (held-out species) and "natural" (species that simply never occur in
    # the train partition). Both are unknown to the model; they are reported separately.
    m["unk_type"] = np.where(m.y.isin(unk), "designed", np.where(m.y.isin(known), "", "natural"))
    m["is_unknown"] = m.unk_type != ""
    m["label"] = [UNK if u else cidx[int(c)] for c, u in zip(m.y, m.is_unknown)]
    m = m[(m.part != "train") | (m.seen & ~m.is_unknown)]  # drop unused train rows
    info = dict(seed=seed, mode=mode, n_clients=len(locs), locations=[int(l) for l in locs],
                unknown_y=unk, known_y=known, unk_pool_size=int(len(pool)), cleaning=clean_report)
    return m[["row", "client", "location_remapped", "seq_id", "datetime", "part", "seen",
              "y", "label", "is_unknown", "unk_type", "image_id", "filename"]].reset_index(drop=True), info


def validate(s, info, X):
    """Hard checks; any violation raises. Returns soft diagnostics (near-duplicates)."""
    parts_per_seq = s.groupby("seq_id").part.nunique()
    assert (parts_per_seq == 1).all(), f"{(parts_per_seq > 1).sum()} sequences span partitions"
    for col in ["image_id", "filename", "row"]:
        assert s[col].is_unique, f"duplicate {col}"
    tr = s[s.part == "train"]
    assert not tr.is_unknown.any(), "unknown species in train partition"
    assert tr.seen.all(), "unseen-site rows in train partition"
    assert set(s[s.part != "train"].label) <= set(range(len(info["known_y"]))) | {UNK}
    if info["mode"] == "time":
        start = s.groupby("seq_id").datetime.transform("min")
        for k, g in s.assign(start=start).groupby("client"):
            t = g.replace({"part": {"tune": "cal"}}).groupby("part").start.agg(["min", "max"])
            if {"train", "cal"} <= set(t.index):
                assert t.loc["train", "max"] <= t.loc["cal", "min"], f"client {k}: train after cal"
            if {"cal", "test"} <= set(t.index):
                assert t.loc["cal", "max"] <= t.loc["test", "min"], f"client {k}: cal after test"
    # Exact duplicate feature vectors across partitions.
    F = X[s.row.to_numpy()]
    h = pd.Series([hashlib.md5(v.tobytes()).hexdigest() for v in F])
    cross = s.assign(h=h).groupby("h").part.nunique()
    assert (cross <= 1).all(), f"{(cross > 1).sum()} identical images across partitions"

    # Soft: near-duplicates (cos > 0.98) between train and test at the same site.
    near, by_cls = 0, {}
    Ft = torch.from_numpy(F).float().cuda()
    for k, g in s.groupby("client"):
        a, b = g.index[g.part == "train"], g.index[g.part == "test"]
        if len(a) == 0 or len(b) == 0:
            continue
        mx = (Ft[b] @ Ft[a].T).max(1).values.cpu().numpy()
        hit = mx > 0.98
        near += int(hit.sum())
        for c in s.loc[b[hit], "y"]:
            by_cls[int(c)] = by_cls.get(int(c), 0) + 1
    n_test = int((s.part == "test").sum())
    return dict(test_near_dup_of_train=near, test_rows=n_test,
                near_dup_pct=round(100 * near / max(n_test, 1), 2),
                near_dup_top_classes=dict(sorted(by_cls.items(), key=lambda x: -x[1])[:5]))


def path(mode, seed):
    return f"{SPLIT_DIR}/{mode}_seed{seed}"


def save(s, info, diag):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    p = path(info["mode"], info["seed"])
    s.to_csv(f"{p}.csv.gz", index=False)
    info["sha256"] = hashlib.sha256(open(f"{p}.csv.gz", "rb").read()).hexdigest()
    info["diagnostics"] = diag
    info["counts"] = {f"{pt}/{t or 'known'}": int(n)
                      for (pt, t), n in s.groupby(["part", "unk_type"]).size().items()}
    json.dump(info, open(f"{p}.json", "w"), indent=1)
    return info


def load_split(mode, seed):
    """The only way experiments obtain a partition. Verifies the saved hash."""
    p = path(mode, seed)
    info = json.load(open(f"{p}.json"))
    assert hashlib.sha256(open(f"{p}.csv.gz", "rb").read()).hexdigest() == info["sha256"], \
        f"{p}.csv.gz changed since it was validated"
    return pd.read_csv(f"{p}.csv.gz", parse_dates=["datetime"]), info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["time", "seq"])
    a = ap.parse_args()
    meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
    X = np.load(f"{ROOT}/features/iwildcam_bioclip.npy")
    assert len(meta) == len(X)
    for mode in a.modes:
        for seed in range(a.seeds):
            s, info = build(meta, seed, mode, X)
            info = save(s, info, validate(s, info, X))
            print(f"{mode} seed {seed}: OK  clients={info['n_clients']} unknown={info['unknown_y']} "
                  f"(pool {info['unk_pool_size']}) counts={info['counts']} "
                  f"near-dup test->train={info['diagnostics']['near_dup_pct']}% "
                  f"top={info['diagnostics']['near_dup_top_classes']}", flush=True)
