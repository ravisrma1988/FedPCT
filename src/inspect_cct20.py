"""Print the structure of the CCT20 annotation files (split sizes, keys, locations, classes)."""
import collections
import glob
import json
import sys

root = sys.argv[1]
for path in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
    d = json.load(open(path))
    imgs, anns = d.get("images", []), d.get("annotations", [])
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    locs = collections.Counter(str(i.get("location")) for i in imgs)
    cls = collections.Counter(cats.get(a["category_id"], a["category_id"]) for a in anns)
    print(f"\n{path}\n  images={len(imgs)} annotations={len(anns)} locations={len(locs)}")
    if imgs:
        print("  image keys:", sorted(imgs[0]), "| example:", {k: imgs[0][k] for k in list(imgs[0])[:6]})
    if anns:
        print("  ann keys:", sorted(anns[0]))
    print("  locations:", dict(locs.most_common()))
    print("  classes:", dict(cls.most_common()))
