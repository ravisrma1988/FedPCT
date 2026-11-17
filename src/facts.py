"""Facts requested by the pre-submission audit, read from the frozen splits (no model runs):
class composition (empty / non-animal share) per partition, number of known classes, unknown
calibration counts, test known-image counts per site, cleaning accounting.
Usage: python facts.py iwildcam|cct
"""
import json
import sys

import numpy as np
import pandas as pd

ds = sys.argv[1]
if ds == "cct":
    import cct_env
    cct_env.use_cct()
    names = pd.read_csv("${FEDPCT_ROOT}/features/cct_meta.csv").drop_duplicates("y").set_index("y").class_name.to_dict()
else:
    c = pd.read_csv("${FEDPCT_ROOT}/data/iwildcam_v2.0/categories.csv")
    names = c[c.y != 99999].set_index("y").name.to_dict()
from splits import load_split  # noqa: E402

NONANIMAL = {"empty", "car", "vehicle", "human", "start", "end"}
out = {}
for seed in range(5):
    s, info = load_split("time", seed)
    s["cname"] = s.y.map(names)
    s["nonanimal"] = s.cname.str.lower().isin(NONANIMAL)
    known = s.label >= 0
    te = (s.part == "test") & known
    cal_unk = s.part.isin(["cal"]) & (s.label < 0)
    per_site = s[te].groupby("client").size()
    out[seed] = dict(
        n_known_classes=len(info["known_y"]),
        nonanimal_known_classes=sorted({names[y] for y in info["known_y"] if names[y].lower() in NONANIMAL}),
        test_known_images=int(te.sum()), test_known_nonanimal_share=round(float(s[te].nonanimal.mean()), 3),
        all_images_nonanimal_share=round(float(s.nonanimal.mean()), 3),
        cal_unknown_images=int(cal_unk.sum()), tune_unknown_images=int(((s.part == "tune") & (s.label < 0)).sum()),
        test_unknown_images=int(((s.part == "test") & (s.label < 0)).sum()),
        designed_unknown=[names[u] for u in info["unknown_y"]],
        test_known_per_site_median=float(per_site.median()), test_known_per_site_p10=float(per_site.quantile(.1)),
        test_known_per_site_p90=float(per_site.quantile(.9)), n_clients=info["n_clients"],
        cleaning=info.get("cleaning"), near_dup_dropped=info.get("near_dup_test_sequences_dropped"))
print(json.dumps(out, indent=1, default=str))
