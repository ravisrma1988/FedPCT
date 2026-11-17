"""Final evaluation on CCT with the frozen lockbox_eval.py functions (identical code path).
  python lockbox_eval_cct.py dev       dry run on CCT development sites
  python lockbox_eval_cct.py lockbox   one-time CCT lockbox evaluation; refuses unless frozen and
                                       unchanged (splits_cct/prereg_freeze.json), refuses to run twice
"""
import json
import os
import sys

import cct_env

cct_env.use_cct()
import lockbox_eval as le  # noqa: E402

le.TAG = cct_env.TAG
le.FREEZE = f"{cct_env.SPLIT_DIR}/prereg_freeze.json"
le.DONE = f"{le.ROOT}/results/LOCKBOX_RUN_cct.json"
le.HASHED = le.HASHED[:]
le.HASHED.remove("results/pers_lambda_bioclip2crop.json")
le.HASHED.remove("splits/lockbox_sites.json")
le.HASHED += ["PREREG_CCT.md", "src/lockbox_eval_cct.py", "src/cct_env.py", "src/cct_run.py", "src/pers_lambda.py",
              f"results/pers_lambda_{cct_env.TAG}.json", "splits_cct/lockbox_sites.json"]

if __name__ == "__main__":
    which = sys.argv[1]
    assert which in ("dev", "lockbox")
    if which == "lockbox":
        assert os.path.exists(le.FREEZE), "CCT pre-registration not frozen"
        frozen = json.load(open(le.FREEZE))["sha256"]
        bad = [p for p in le.HASHED if frozen.get(p) != le.sha(p)]
        assert not bad, f"files changed since freeze: {bad}"
        assert not os.path.exists(le.DONE), "CCT lockbox already evaluated; it is run once"
        json.dump(dict(freeze=frozen), open(le.DONE, "w"), indent=1)
    sel = json.load(open(f"{le.ROOT}/results/pers_lambda_{le.TAG}.json"))
    allr = []
    for sd in le.SEEDS:
        lam_star = {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in le.RATES}
        allr.append(le.run(sd, which, lam_star))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{le.ROOT}/results/final_eval_cct_{which}.json", "w"), indent=1, default=float)
    le.report(allr, f"CCT {which}")
    print(f"FINAL_EVAL_DONE cct {which}")
