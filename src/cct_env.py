"""Point the frozen iWildCam code at the CCT data without editing it: split directory, lockbox file,
unknown-species rule and feature file name. Every CCT script calls use_cct() before anything else.

CCT-specific rules (fixed 2026-11-02, before any CCT split, feature or result existed):
  * unknown pool: the iWildCam rule (20-200 train images, >= 3 train sites, seen sites only); if
    fewer than 2 species qualify, the upper bound is raised to 2000. N_UNK = 2 (CCT has 21 non-empty
    classes; iWildCam's 8 of ~180 classes would be < 1 here; 2 keeps a designed open-set component).
  * features: BioCLIP-2 MegaDetector crops, features/cct_bioclip2crop.npy, exposed to the frozen
    scripts under the name features/iwildcam_cct_bioclip2crop.npy (symlink), tag "cct_bioclip2crop".
"""
import os

import numpy as np
import pandas as pd

ROOT = "${FEDPCT_ROOT}"
TAG = "cct_bioclip2crop"
SPLIT_DIR = f"{ROOT}/splits_cct"
LOCKBOX = f"{SPLIT_DIR}/lockbox_sites.json"
N_UNK = 2
POOL_FALLBACK_MAX = 2000


def use_cct():
    import lockbox
    import splits
    splits.SPLIT_DIR = SPLIT_DIR
    splits.N_UNK = N_UNK
    lockbox.PATH = LOCKBOX
    link = f"{ROOT}/features/iwildcam_{TAG}.npy"
    if not os.path.islink(link):
        assert not os.path.exists(link)
        os.symlink(f"{ROOT}/features/cct_bioclip2crop.npy", link)


def load_meta():
    m = pd.read_csv(f"{ROOT}/features/cct_meta.csv")
    X = np.load(f"{ROOT}/features/cct_bioclip2crop.npy")
    assert len(m) == len(X)
    return m, X
