"""Check how categories.csv maps y to names."""
import pandas as pd

ROOT = "${FEDPCT_ROOT}"
c = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
m = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
print("rows", len(c), "unique y", c.y.nunique(), "unique category_id", c.category_id.nunique())
print("duplicate-y rows:\n", c[c.y.duplicated(keep=False)].sort_values("y").head(12).to_string())
print("\nmetadata (y, category_id) pairs unique per y:",
      (m.groupby("y").category_id.nunique() == 1).all(), "| distinct y in metadata:", m.y.nunique())
pairs = m[["y", "category_id"]].drop_duplicates()
j = pairs.merge(c, on="category_id", how="left", suffixes=("", "_cat"))
print("metadata categories found in categories.csv by category_id:", j.name.notna().sum(), "of", len(j))
print("y in metadata equals y in categories.csv for the same category_id:", (j.y == j.y_cat).mean())
