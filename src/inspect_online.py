"""Why are online set sizes so large? Per-site stream length vs set size, and score distribution."""
import json

import numpy as np

rows = json.load(open("${FEDPCT_ROOT}/results/online_time.json"))[0]
key = "qt_pooled_init|r=0.3|eta=0.02"
n = np.array([r["n"] for r in rows])
size = np.array([r[key]["size"] for r in rows])
cov = np.array([r[key]["cov"] for r in rows])
static = np.array([r["static_pooled"]["size"] for r in rows])
print(f"{key}: sites={len(rows)}")
for lo, hi in [(0, 200), (200, 500), (500, 1000), (1000, 1e9)]:
    m = (n >= lo) & (n < hi)
    if m.any():
        print(f"  stream length {lo:>5}-{hi:<9}: sites={m.sum():3d}  size online={size[m].mean():6.1f} "
              f"static={static[m].mean():5.1f}  cov={cov[m].mean():.3f}")
print("  sites with mean set size > 50:", int((size > 50).sum()), "of", len(rows))
