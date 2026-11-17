"""Second-backbone robustness analysis (PREREG_BACKBONE.md): frozen lockbox_eval.py functions with
the DINOv2 crop features. Development sites (primary) and lockbox sites (post hoc, not
confirmatory; the lockbox one-shot guard is not used because the confirmatory runs are done).
Requires results/pers_lambda_<tag>.json from the pre-registered lambda rule.
Usage: python backbone_eval.py iwildcam|cct
"""
import json
import sys

ds = sys.argv[1]
if ds == "cct":
    import cct_env
    cct_env.use_cct()
import lockbox_eval as le  # noqa: E402

le.TAG = "cct_dinov2crop" if ds == "cct" else "dinov2crop"
sel = json.load(open(f"{le.ROOT}/results/pers_lambda_{le.TAG}.json"))
for which in ("dev", "lockbox"):
    allr = []
    for sd in le.SEEDS:
        lam_star = {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in le.RATES}
        allr.append(le.run(sd, which, lam_star))
    json.dump(allr, open(f"{le.ROOT}/results/backbone_{le.TAG}_{which}.json", "w"), indent=1, default=float)
    le.report(allr, f"{ds} {which}{' (POST HOC)' if which == 'lockbox' else ''} DINOv2")
print(f"BACKBONE_DONE {ds}")
