"""Lockbox: 30% of sites held out from all method development (created 2026-11-01, after the
baseline/online/EB/persistence experiments, which used all sites). Stratified by seen/unseen.
Development code must call dev_mask(); lockbox sites are evaluated ONCE, with a pre-specified
method and settings, at the end.
Usage: python lockbox.py   (writes splits/lockbox_sites.json; refuses to overwrite)
"""
import json
import os

import numpy as np

ROOT = "${FEDPCT_ROOT}"
PATH = f"{ROOT}/splits/lockbox_sites.json"


def lockbox_locations():
    return set(json.load(open(PATH))["locations"])


def dev_mask(info, seen=None):
    """Boolean (K,) over clients of a split: True = development site."""
    lb = lockbox_locations()
    return np.array([loc not in lb for loc in info["locations"]])


if __name__ == "__main__":
    assert not os.path.exists(PATH), "lockbox already exists; never redraw it"
    from splits import load_split

    s, info = load_split("time", 0)
    seen = s.groupby("client").seen.first().reindex(range(info["n_clients"])).to_numpy()
    locs = np.array(info["locations"])
    rng = np.random.default_rng(20261101)
    pick = []
    for flag in [True, False]:
        cand = locs[seen == flag]
        pick += rng.choice(cand, int(round(0.3 * len(cand))), replace=False).tolist()
    json.dump(dict(created="2026-11-01", rule="30% of sites, stratified by seen/unseen, rng 20261101",
                   locations=sorted(int(x) for x in pick)), open(PATH, "w"), indent=1)
    print(f"lockbox: {len(pick)} of {len(locs)} sites "
          f"(seen {int(np.isin(pick, locs[seen]).sum())}, unseen {int(np.isin(pick, locs[~seen]).sum())})")
