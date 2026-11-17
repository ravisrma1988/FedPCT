"""Final evaluation on FMoW-WILDS with the frozen lockbox_eval.py functions (identical code path as
iWildCam and CCT), followed by the pre-specified secondary comparison with stronger online trackers
(online_baselines.py), on the same sites, seeds and labelling draws.
  python lockbox_eval_fmow.py dev       dry run on FMoW development sites
  python lockbox_eval_fmow.py lockbox   one-time FMoW lockbox evaluation; refuses unless frozen and
                                        unchanged (splits_fmow/prereg_freeze.json), refuses to run twice
"""
import json
import os
import sys

import fmow_env

fmow_env.use_fmow()
import lockbox_eval as le  # noqa: E402
import online_baselines as ob  # noqa: E402

le.TAG = fmow_env.TAG
le.FREEZE = f"{fmow_env.SPLIT_DIR}/prereg_freeze.json"
le.DONE = f"{le.ROOT}/results/LOCKBOX_RUN_fmow.json"
le.HASHED = le.HASHED[:]
le.HASHED.remove("results/pers_lambda_bioclip2crop.json")
le.HASHED.remove("splits/lockbox_sites.json")
le.HASHED += ["PREREG_FMOW.md", "src/lockbox_eval_fmow.py", "src/fmow_env.py", "src/fmow_meta.py",
              "src/fmow_features.py", "src/splits_fmow.py", "src/online_baselines.py", "src/pers_lambda.py",
              "features/fmow_meta.csv", "features/fmow_dinov2.npy", f"results/pers_lambda_{fmow_env.TAG}.json", "splits_fmow/lockbox_sites.json"]
le.HASHED += [f"splits_fmow/time_seed{s}.json" for s in le.SEEDS]

if __name__ == "__main__":
    which = sys.argv[1]
    assert which in ("dev", "lockbox")
    if which == "lockbox":
        assert os.path.exists(le.FREEZE), "FMoW pre-registration not frozen"
        frozen = json.load(open(le.FREEZE))["sha256"]
        bad = [p for p in le.HASHED if frozen.get(p) != le.sha(p)]
        assert not bad, f"files changed since freeze: {bad}"
        assert not os.path.exists(le.DONE), "FMoW lockbox already evaluated; it is run once"
        json.dump(dict(freeze=frozen), open(le.DONE, "w"), indent=1)
    sel = json.load(open(f"{le.ROOT}/results/pers_lambda_{le.TAG}.json"))
    lam = [{rate: sel[sd][f"r={rate}"]["lam_star"] for rate in le.RATES} for sd in le.SEEDS]
    allr = []
    for sd in le.SEEDS:
        allr.append(le.run(sd, which, lam[sd]))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{le.ROOT}/results/final_eval_fmow_{which}.json", "w"), indent=1, default=float)
    le.report(allr, f"FMoW {which}")
    sec = []
    for sd in le.SEEDS:
        sec.append(ob.run(sd, lam[sd], which))
    json.dump(sec, open(f"{le.ROOT}/results/online_baselines_fmow_{which}.json", "w"), indent=1, default=float)
    ob.report(sec, f"FMoW {which.upper()} sites (secondary comparison)")
    print(f"FINAL_EVAL_DONE fmow {which}")
