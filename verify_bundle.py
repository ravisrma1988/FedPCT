"""Integrity and provenance check of this repository. Needs only the Python 3 standard library.

    python verify_bundle.py

1. Every file listed in SHA256_BUNDLE.txt is hashed and compared with the listed value (files as distributed).
2. Every file hashed in the freeze records of the three lockbox evaluations is compared with SHA256_ORIGINAL.txt,
   the hashes of the files before path anonymization. Agreement shows that the stored files are the frozen
   versions. Frozen files that are not redistributed (image features of the source datasets) are listed
   separately and are not counted as mismatches.
3. Files whose distributed content differs from the original are listed. The only differences are the
   replacement of local directory paths described in README.md.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = ("splits/prereg_freeze.json", "splits_cct/prereg_freeze.json", "splits_fmow/prereg_freeze.json",
           "results/LOCKBOX_RUN.json", "results/LOCKBOX_RUN_cct.json", "results/LOCKBOX_RUN_fmow.json")


def sha(rel):
    with open(os.path.join(HERE, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def listing(name):
    out = {}
    with open(os.path.join(HERE, name), encoding="utf8") as f:
        for line in f:
            h, p = line.split(maxsplit=1)
            out[p.strip()[2:]] = h
    return out


bundle, original = listing("SHA256_BUNDLE.txt"), listing("SHA256_ORIGINAL.txt")
bad = [p for p, h in bundle.items() if not os.path.exists(os.path.join(HERE, p)) or sha(p) != h]
print(f"1. distributed files: {len(bundle) - len(bad)} of {len(bundle)} match SHA256_BUNDLE.txt")
for p in bad:
    print("   MISMATCH", p)

problems = 0
for rec in RECORDS:
    path = os.path.join(HERE, rec)
    if not os.path.exists(path):
        print(f"2. {rec}: not present")
        continue
    with open(path, encoding="utf8") as f:
        d = json.load(f)
    frozen = d.get("sha256") or d["freeze"]
    ok, absent = 0, []
    for p, h in frozen.items():
        if p not in original:
            absent.append(p)
        elif original[p] == h:
            ok += 1
        else:
            problems += 1
            print("   FREEZE MISMATCH", rec, p)
    print(f"2. {rec}: {ok} of {len(frozen) - len(absent)} frozen files equal the stored originals"
          + (f", {len(absent)} not redistributed ({', '.join(absent)})" if absent else ""))

changed = sorted(p for p in original if p in bundle and original[p] != bundle[p])
print(f"3. {len(changed)} files differ from the originals only by path anonymization")
sys.exit(1 if bad or problems else 0)
