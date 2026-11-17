"""Inspect CCT metadata: are location, timestamp and sequence id present per image? Counts per site,
classes, empties, multi-label images; MegaDetector results coverage. Read-only.
Usage: python -I cct_inspect.py <metadata json> <md results json>
"""
import collections
import json
import sys

meta = json.load(open(sys.argv[1]))
print("top-level keys:", list(meta))
print("info:", meta.get("info"))
imgs, anns, cats = meta["images"], meta["annotations"], meta["categories"]
print(f"images {len(imgs)}  annotations {len(anns)}  categories {len(cats)}")
print("image fields:", sorted({k for im in imgs[:2000] for k in im}))
print("example image:", imgs[0])
print("annotation fields:", sorted({k for a in anns[:2000] for k in a}))
for f in ("location", "datetime", "date_captured", "seq_id", "frame_num", "seq_num_frames"):
    print(f"  field {f:<15} present in {sum(f in im for im in imgs)} / {len(imgs)} images")
loc = collections.Counter(str(im.get("location")) for im in imgs)
print(f"locations {len(loc)}; with >= 250 images: {sum(v >= 250 for v in loc.values())}")
print("images per location (top 5 / bottom 5):", loc.most_common(5), loc.most_common()[-5:])
cname = {c["id"]: c["name"] for c in cats}
per_img = collections.defaultdict(set)
for a in anns:
    per_img[a["image_id"]].add(a["category_id"])
cls = collections.Counter(cname[c] for s in per_img.values() for c in s)
print("classes:", cls.most_common())
print("images with >1 category:", sum(len(s) > 1 for s in per_img.values()),
      "| images without annotation:", len(imgs) - len(per_img))
md = json.load(open(sys.argv[2]))
print("MD keys:", list(md), "| info:", md.get("info"))
print(f"MD images {len(md['images'])}; example:", md["images"][0])
ids = {im["file_name"] for im in imgs}
mdf = {im["file"] for im in md["images"]}
print("example metadata file_name:", imgs[0]["file_name"], "| example MD file:", next(iter(mdf)))
