"""Freeze the CCT pre-registration: SHA-256 of PREREG.md, PREREG_CCT.md and every file the CCT
evaluation uses (lockbox_eval_cct.le.HASHED). Refuses to overwrite or to run after the lockbox.
Usage: python freeze_cct.py
"""
import json
import os
from datetime import datetime, timezone

import lockbox_eval_cct  # noqa: F401  (patches lockbox_eval constants for CCT)
from lockbox_eval_cct import le

assert not os.path.exists(le.FREEZE), "CCT already frozen; never re-freeze"
assert not os.path.exists(le.DONE), "CCT lockbox already evaluated"
rec = dict(frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), sha256={p: le.sha(p) for p in le.HASHED})
json.dump(rec, open(le.FREEZE, "w"), indent=1)
for p, h in rec["sha256"].items():
    print(f"{h}  {p}")
print("CCT FROZEN", rec["frozen_utc"])
