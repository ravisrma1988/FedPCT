"""Class and image accounting for Online Resource 1 and the manuscript (audit 2, items 1.1, 1.2, 3.6).
For each seed: how every labelled source class is used (known, designed held-out, natural held-out =
present only outside the seen-site training partitions, or absent from the retained data), image counts
by partition and role, and which partitions each site category contributes.
Usage: python class_accounting.py iwildcam|cct
"""
import json
import sys

import pandas as pd

ds = sys.argv[1]
ROOT = "${FEDPCT_ROOT}"
if ds == "cct":
    import cct_env
    cct_env.use_cct()
    meta = pd.read_csv(f"{ROOT}/features/cct_meta.csv")
    NAMES = meta.drop_duplicates("y").set_index("y").class_name.to_dict()
    all_classes = sorted(NAMES)
else:
    c = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
    NAMES = c[c.y != 99999].set_index("y").name.to_dict()
    meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
    all_classes = sorted(NAMES)
from lockbox import dev_mask  # noqa: E402
from splits import load_split  # noqa: E402

out = {"n_labelled_classes": len(all_classes), "images_in_meta": int(len(meta))}
per_class = {}
for seed in range(5):
    s, info = load_split("time", seed)
    known, designed = set(info["known_y"]), set(info["unknown_y"])
    present = set(s.y)
    natural = present - known - designed
    absent = set(all_classes) - present
    seen = s.groupby("client").seen.first()
    lock = ~dev_mask(info)
    site_cat = {k: ("lockbox" if lock[k] else "development") + "/" + ("seen" if seen.get(k, False) else "unseen")
                for k in range(info["n_clients"])}
    s["sitecat"] = s.client.map(site_cat)
    s["role"] = s.apply(lambda r: "known" if r.y in known else ("designed" if r.y in designed else "natural"), axis=1)
    use = s.groupby(["sitecat", "part"]).size().unstack(fill_value=0)
    out[f"seed{seed}"] = dict(
        known=len(known), designed=len(designed), natural_heldout=len(natural), absent_from_retained=len(absent),
        natural_classes=sorted(NAMES[y] for y in natural), absent_classes_sample=sorted(NAMES[y] for y in absent)[:15],
        images_by_part_role={f"{p}/{r}": int(n) for (p, r), n in s.groupby(["part", "role"]).size().items()},
        sites_by_category=pd.Series(site_cat).value_counts().to_dict(),
        partitions_by_site_category=use.to_dict(orient="index"),
        retained_rows=int(len(s)))
    if seed == 0:
        tr = s[s.part == "train"]
        for y in sorted(present | absent):
            role = "known" if y in known else "designed held-out" if y in designed else "natural held-out" if y in natural else "absent"
            sub = s[s.y == y]
            per_class[NAMES[y]] = dict(role_seed0=role, train=int((sub.part == "train").sum()),
                                       tune=int((sub.part == "tune").sum()), cal=int((sub.part == "cal").sum()),
                                       test=int((sub.part == "test").sum()))
out["per_class_seed0"] = per_class
json.dump(out, open(f"{ROOT}/results/class_accounting_{ds}.json", "w"), indent=1)
for k in [f"seed{i}" for i in range(5)]:
    v = out[k]
    print(ds, k, "known", v["known"], "designed", v["designed"], "natural", v["natural_heldout"], "absent", v["absent_from_retained"],
          "| retained rows", v["retained_rows"])
print("seed0 natural classes:", out["seed0"]["natural_classes"][:20])
print("seed0 images by part/role:", out["seed0"]["images_by_part_role"])
print("seed0 sites:", out["seed0"]["sites_by_category"])
print("seed0 partitions by site category:", json.dumps(out["seed0"]["partitions_by_site_category"]))
