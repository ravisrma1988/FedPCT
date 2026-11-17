"""Build, validate and save the FMoW time splits with the frozen split code (splits.build / validate /
save), then draw the FMoW lockbox. Run once; refuses to overwrite.

Leakage safeguards (fixed 2026-11-15, before any FMoW model or result):
  * same split code as iWildCam and CCT: whole sequences (here: object instances), per-site time order
    of their first view (train 40% / cal 20% / test 40% of objects), tune = 1/3 of cal objects,
    held-out classes from the train partition only, exact feature duplicates removed, hard assertions
  * near-duplicate rule (as CCT): validate() reports test images with a cos > 0.98 match among the SAME
    site's train images. If that rate exceeds 2% for a seed, those test OBJECTS are dropped
    (label-free) and the split is re-validated; both rates are recorded.
  * lockbox: 30% of sites, stratified by seen/unseen, rng 20261115, drawn here before any FMoW result.
Usage: python splits_fmow.py
"""
import json
import os

import numpy as np
import pandas as pd

import fmow_env

fmow_env.use_fmow()
import splits  # noqa: E402  (patched by use_fmow)

NEAR_DUP_MAX_PCT = 2.0
assert not os.path.exists(f"{fmow_env.SPLIT_DIR}/time_seed0.json"), "FMoW splits exist; never rebuild"
os.makedirs(fmow_env.SPLIT_DIR, exist_ok=True)
meta, X = fmow_env.load_meta()
meta["datetime"] = pd.to_datetime(meta.datetime, format="ISO8601").dt.strftime("%Y-%m-%d %H:%M:%S.%f")  # one format


def build(seed):
    base = splits.UNK_POOL_TRAIN_RANGE
    for hi in (base[1],) + fmow_env.POOL_FALLBACKS:
        splits.UNK_POOL_TRAIN_RANGE = (base[0], hi)
        try:
            return splits.build(meta, seed, "time", X), splits.UNK_POOL_TRAIN_RANGE
        except ValueError as e:  # fewer than N_UNK classes in the pool; anything else is re-raised
            if "larger sample than population" not in str(e):
                raise
            continue
        finally:
            splits.UNK_POOL_TRAIN_RANGE = base
    raise RuntimeError("no held-out class pool")


def drop_near_dup_test_objects(s):
    import torch
    F = torch.from_numpy(X[s.row.to_numpy()]).float().cuda()
    drop = set()
    for k, g in s.groupby("client"):
        a, b = g.index[g.part == "train"], g.index[g.part == "test"]
        if len(a) and len(b):
            hit = torch.cat([(F[b[i:i + 4096]] @ F[a].T).max(1).values for i in range(0, len(b), 4096)]).cpu().numpy() > 0.98
            drop |= set(s.loc[b[hit], "seq_id"])
    return s[~s.seq_id.isin(drop)].reset_index(drop=True), len(drop)


for seed in range(5):
    (s, info), rng_used = build(seed)
    info["dataset"] = "fmow"
    info["unk_pool_train_range"] = list(rng_used)
    diag = splits.validate(s, info, X)
    if diag["near_dup_pct"] > NEAR_DUP_MAX_PCT:
        info["near_dup_before_drop"] = diag
        s, n = drop_near_dup_test_objects(s)
        info["near_dup_test_objects_dropped"] = n
        diag = splits.validate(s, info, X)
    info = splits.save(s, info, diag)
    print(f"fmow time seed {seed}: OK clients={info['n_clients']} unknown={info['unknown_y']} "
          f"(pool {info['unk_pool_size']}, range {rng_used}) near-dup={diag['near_dup_pct']}% "
          f"dropped_objects={info.get('near_dup_test_objects_dropped', 0)} counts={info['counts']}", flush=True)

s, info = splits.load_split("time", 0)
seen = s.groupby("client").seen.first().reindex(range(info["n_clients"])).to_numpy()
locs = np.array(info["locations"])
assert not os.path.exists(fmow_env.LOCKBOX), "FMoW lockbox exists; never redraw"
rng = np.random.default_rng(20261115)
pick = []
for flag in (True, False):
    cand = np.array(sorted(locs[seen == flag]))
    pick += rng.choice(cand, int(round(0.3 * len(cand))), replace=False).tolist()
json.dump(dict(created="2026-11-15", dataset="fmow", rule="30% of sites per seen/unseen stratum, rng 20261115",
               locations=sorted(int(x) for x in pick)), open(fmow_env.LOCKBOX, "w"), indent=1)
print(f"FMoW lockbox: {len(pick)} of {len(locs)} sites (seen {int(np.isin(pick, locs[seen]).sum())}, "
      f"unseen {int(np.isin(pick, locs[~seen]).sum())})")
print("FMOW_SPLITS_DONE")
