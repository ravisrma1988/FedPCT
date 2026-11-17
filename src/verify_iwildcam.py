"""Check the extracted iWildCam v2.0 against the WILDS v2.0 statistics and the wilds loader."""
import os

import pandas as pd

ROOT = "${FEDPCT_ROOT}/data/iwildcam_v2.0"
EXPECTED = {"train": 129809, "val": 14961, "test": 42791, "id_val": 7314, "id_test": 8154}

print("top-level:", sorted(os.listdir(ROOT))[:20])
meta = pd.read_csv(f"{ROOT}/metadata.csv")
print("columns:", list(meta.columns))
sizes = meta.split.value_counts().to_dict()
print("split sizes:", sizes, "| match WILDS:", sizes == EXPECTED)
loc_col = "location_remapped" if "location_remapped" in meta else "location"
print(f"total {len(meta)}, classes in y: {meta.y.nunique()}, locations: {meta[loc_col].nunique()}")
img_dir = f"{ROOT}/train"
missing = sum(not os.path.exists(f"{img_dir}/{f}") for f in meta.filename)
print(f"missing image files: {missing}")

from wilds import get_dataset  # noqa: E402

ds = get_dataset("iwildcam", root_dir="${FEDPCT_ROOT}/data", download=False)
print("wilds loader OK:", len(ds), "examples,", ds.n_classes, "classes")
