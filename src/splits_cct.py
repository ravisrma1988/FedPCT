"""Build, validate and save the CCT time splits with the frozen iWildCam split code (splits.build /
validate / save), then draw the CCT lockbox. Run once; refuses to overwrite.

Leakage safeguards (fixed before any CCT result):
  * same split code as iWildCam: whole sequences, per-site time order (train 40% / cal 20% /
    test 40%), tune = 1/3 of cal sequences, unknowns from the train partition only, exact feature
    duplicates removed, hard assertions in validate()
  * near-duplicate rule: validate() reports test images with a cos > 0.98 match among the SAME
    site's train images. If that rate exceeds 2% for a seed, such test SEQUENCES are dropped
    (label-free) and the split is re-validated; both rates are recorded.
  * lockbox: 30% of sites, stratified by seen/unseen, drawn with rng 20261102 from sites that are
    NOT CCT20 pilot locations; drawn here, before any CCT result exists; never redrawn.
Usage: python splits_cct.py
"""
import json
import os

import numpy as np

import cct_env

cct_env.use_cct()
import splits  # noqa: E402  (patched by use_cct)

NEAR_DUP_MAX_PCT = 2.0
assert not os.path.exists(f"{cct_env.SPLIT_DIR}/time_seed0.json"), "CCT splits exist; never rebuild"
os.makedirs(cct_env.SPLIT_DIR, exist_ok=True)
meta, X = cct_env.load_meta()
c20 = set(json.load(open(f"{cct_env.ROOT}/reports/cct_meta_report.json"))["cct20_pilot_locations"])


def build(seed):
    try:
        return splits.build(meta, seed, "time", X), splits.UNK_POOL_TRAIN_RANGE
    except ValueError:  # fewer than N_UNK species in the iWildCam pool range
        old = splits.UNK_POOL_TRAIN_RANGE
        splits.UNK_POOL_TRAIN_RANGE = (old[0], cct_env.POOL_FALLBACK_MAX)
        try:
            return splits.build(meta, seed, "time", X), splits.UNK_POOL_TRAIN_RANGE
        finally:
            splits.UNK_POOL_TRAIN_RANGE = old


def drop_near_dup_test_seqs(s):
    import torch
    F = torch.from_numpy(X[s.row.to_numpy()]).float().cuda()
    drop = set()
    for k, g in s.groupby("client"):
        a, b = g.index[g.part == "train"], g.index[g.part == "test"]
        if len(a) and len(b):
            hit = (F[b] @ F[a].T).max(1).values.cpu().numpy() > 0.98
            drop |= set(s.loc[b[hit], "seq_id"])
    return s[~s.seq_id.isin(drop)].reset_index(drop=True), len(drop)


for seed in range(5):
    (s, info), rng_used = build(seed)
    info["dataset"] = "cct"
    info["unk_pool_train_range"] = list(rng_used)
    diag = splits.validate(s, info, X)
    if diag["near_dup_pct"] > NEAR_DUP_MAX_PCT:
        info["near_dup_before_drop"] = diag
        s, n = drop_near_dup_test_seqs(s)
        info["near_dup_test_sequences_dropped"] = n
        diag = splits.validate(s, info, X)
    info = splits.save(s, info, diag)
    print(f"cct time seed {seed}: OK clients={info['n_clients']} unknown={info['unknown_y']} "
          f"(pool {info['unk_pool_size']}, range {rng_used}) near-dup={diag['near_dup_pct']}% "
          f"dropped_seqs={info.get('near_dup_test_sequences_dropped', 0)} counts={info['counts']}", flush=True)

# lockbox: drawn from the seed-0 split's site list (identical across seeds: sites depend on counts only)
s, info = splits.load_split("time", 0)
seen = s.groupby("client").seen.first().reindex(range(info["n_clients"])).to_numpy()
locs = np.array(info["locations"])
assert not os.path.exists(cct_env.LOCKBOX), "CCT lockbox exists; never redraw"
rng = np.random.default_rng(20261102)
pick = []
for flag in (True, False):
    cand = np.array([l for l in locs[seen == flag] if str(l) not in c20])
    n = int(round(0.3 * (seen == flag).sum()))
    pick += rng.choice(cand, min(n, len(cand)), replace=False).tolist()
json.dump(dict(created="2026-11-02", dataset="cct",
               rule="30% of sites per seen/unseen stratum, excluding CCT20 pilot locations, rng 20261102",
               locations=sorted(int(x) for x in pick)), open(cct_env.LOCKBOX, "w"), indent=1)
print(f"CCT lockbox: {len(pick)} of {len(locs)} sites (seen {int(np.isin(pick, locs[seen]).sum())}, "
      f"unseen {int(np.isin(pick, locs[~seen]).sum())}); CCT20 sites among clients: "
      f"{sum(str(l) in c20 for l in locs)}")
print("CCT_SPLITS_DONE")
