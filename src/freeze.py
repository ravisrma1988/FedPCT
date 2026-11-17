"""Freeze the pre-registration: record SHA-256 of PREREG.md and every file the final evaluation
uses (list in lockbox_eval.HASHED). Refuses to overwrite an existing freeze.
Usage: python freeze.py
"""
import json
import os
from datetime import datetime, timezone

from lockbox_eval import DONE, FREEZE, HASHED, sha

assert not os.path.exists(FREEZE), "already frozen; never re-freeze"
assert not os.path.exists(DONE), "lockbox already evaluated"
rec = dict(frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), sha256={p: sha(p) for p in HASHED})
json.dump(rec, open(FREEZE, "w"), indent=1)
for p, h in rec["sha256"].items():
    print(f"{h}  {p}")
print("FROZEN", rec["frozen_utc"])
