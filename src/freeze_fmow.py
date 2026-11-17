"""Freeze the FMoW pre-registration: SHA-256 of PREREG.md, PREREG_FMOW.md and every file the FMoW
evaluation uses (lockbox_eval_fmow.le.HASHED). Refuses to overwrite or to run after the lockbox.
Usage: python freeze_fmow.py
"""
import json
import os
from datetime import datetime, timezone

import lockbox_eval_fmow  # noqa: F401  (patches lockbox_eval constants for FMoW)
from lockbox_eval_fmow import le

assert not os.path.exists(le.FREEZE), "FMoW already frozen; never re-freeze"
assert not os.path.exists(le.DONE), "FMoW lockbox already evaluated"
rec = dict(frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), sha256={p: le.sha(p) for p in le.HASHED})
json.dump(rec, open(le.FREEZE, "w"), indent=1)
for p, h in rec["sha256"].items():
    print(f"{h}  {p}")
print("FMOW FROZEN", rec["frozen_utc"])
