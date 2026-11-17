"""Point the frozen iWildCam code at FMoW-WILDS without editing it (as cct_env.py does for CCT):
split directory, lockbox file, held-out-class rule and feature file name. Every FMoW script calls
use_fmow() before anything else.

FMoW-specific rules (fixed 2026-11-15, before any FMoW feature, model or result):
  * held-out classes: N_UNK = 4 of the 62 categories. Pool = the iWildCam rule (20-200 train images,
    >= 3 train sites, seen sites only); if fewer than 4 classes qualify, the upper bound is raised to
    2,000, then to 20,000, then removed, in that order (FMoW classes are large, so a fallback is expected).
  * features: DINOv2 ViT-L/14 whole-image features, features/fmow_dinov2.npy, exposed to the frozen
    scripts as features/iwildcam_fmow_dinov2.npy (symlink), tag "fmow_dinov2".
"""
import os

import numpy as np
import pandas as pd

ROOT = "${FEDPCT_ROOT}"
TAG = "fmow_dinov2"
SPLIT_DIR = f"{ROOT}/splits_fmow"
LOCKBOX = f"{SPLIT_DIR}/lockbox_sites.json"
N_UNK = 4
POOL_FALLBACKS = (2000, 20000, 10 ** 9)


def use_fmow():
    import lockbox
    import splits
    splits.SPLIT_DIR = SPLIT_DIR
    splits.N_UNK = N_UNK
    lockbox.PATH = LOCKBOX
    # FMoW timestamps are stored with and without milliseconds, which pandas' inferred parse in
    # load_split leaves as strings; parse them as datetimes so time order never depends on string order.
    if not getattr(splits.load_split, "_fmow", False):
        orig = splits.load_split

        def load_split(mode, seed):
            s, info = orig(mode, seed)
            s["datetime"] = pd.to_datetime(s.datetime, format="ISO8601")
            return s, info
        load_split._fmow = True
        splits.load_split = load_split
    link = f"{ROOT}/features/iwildcam_{TAG}.npy"
    if not os.path.islink(link):
        assert not os.path.exists(link)
        os.symlink(f"{ROOT}/features/fmow_dinov2.npy", link)


def load_meta():
    m = pd.read_csv(f"{ROOT}/features/fmow_meta.csv")
    X = np.load(f"{ROOT}/features/fmow_dinov2.npy")
    assert len(m) == len(X)
    return m, X
