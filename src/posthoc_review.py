"""POST HOC analyses requested by the pre-submission review of 2026-11-16 (lockbox sites, already used
once, so not confirmatory). Method, seeds, label draws, step sizes, proximal weights and thresholds are
those of lockbox_eval.py and posthoc_animal.py; only the evaluation is extended.
  * B1: the iWildCam class 'motorcycle' is a vehicle and is now grouped with the non-animal classes
    (posthoc_animal.py grouped only 'empty' and 'car' as non-animal)
  * A2: mean coverage shortfall over sites, mean_k max(0, 0.9 - cov_k), next to the flagged share
  * A3: static frontier of the GLOBAL classifier (FCP at alpha' in FRONTIER_G) and the flagged share of
    this frontier interpolated at the mean set size of FedPCT, per seed (clipping is recorded)
Usage: python posthoc_review.py iwildcam|cct|fmow
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

ds = sys.argv[1]
ROOT = "${FEDPCT_ROOT}"
if ds == "cct":
    import cct_env
    cct_env.use_cct()
    NAMES = pd.read_csv(f"{ROOT}/features/cct_meta.csv").drop_duplicates("y").set_index("y").class_name.to_dict()
    TAG = "cct_bioclip2crop"
elif ds == "fmow":
    import fmow_env
    fmow_env.use_fmow()
    NAMES = None
    TAG = fmow_env.TAG
else:
    c = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
    NAMES = c[c.y != 99999].set_index("y").name.to_dict()
    TAG = "bioclip2crop"
import cp_baselines as cp  # noqa: E402
import lockbox_eval as le  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from pers_frontier import FRONTIER  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

NONANIMAL = {"empty", "car", "motorcycle"}
FRONTIER_G = tuple(FRONTIER) + (0.02, 0.01, 0.005)
RATES = (0.1, 0.3, 1.0)
sel = json.load(open(f"{ROOT}/results/pers_lambda_{TAG}.json"))


def masked(sites, animal):
    return [dict(st, score=np.where(animal[st["idx"]], st["score"], np.nan)) for st in sites]


def per_site(ev):
    return [dict(cov=r["cov"], size=r["size"], sig=bool(r["sig"]) if r["cov"] == r["cov"] else None, seen=bool(r["seen"])) for r in ev]


def summ(rows):
    v = [x for x in rows if x["sig"] is not None]
    cov = np.array([x["cov"] for x in v])
    return dict(sig=100 * np.mean([x["sig"] for x in v]), size=float(np.mean([x["size"] for x in v])),
                cov_mean=float(cov.mean()), below_obs=100 * float(np.mean(cov < 0.9)),
                shortfall=float(np.mean(np.maximum(0, 0.9 - cov))), cov_median=float(np.median(cov)), n=len(v))


rec = {}
for seed in range(5):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    mask = ~dev_mask(info)
    X = torch.from_numpy(np.load(f"{le.ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    animal = None if NAMES is None else ~s.y.map(NAMES).str.lower().isin(NONANIMAL).to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    cal = dict(S=S_all[cm], y=y[cm], client=client[cm])
    tau_g = [cp.thresholds("pooled", cal, a, C, K)[0] for a in FRONTIER_G]
    tau0 = tau_g[0]
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and mask[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    one = torch.zeros(1, device=DEV)
    pad, nk = pack(base)
    fixed = torch.stack([torch.full((len(base), pad.shape[1]), t, device=DEV) for t in tau_g])
    evg = site_eval(fixed, base, seen_t)
    r = dict(fcp_all=per_site(evg[0]), gfront={str(a): per_site(e) for a, e in zip(FRONTIER_G, evg)})
    if animal is not None:
        r["fcp_animal"] = per_site(site_eval(fixed[:1], masked(base, animal), seen_t)[0])
        r["n_animal"] = [int(np.sum(animal[st["idx"]] & ~np.isnan(st["score"]))) for st in base]
        r["n_known"] = [int(np.sum(~np.isnan(st["score"]))) for st in base]
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad2, nk2, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, sel[seed][f"r={rate}"]["lam_star"], seed, score_fn)
        taus = track2(pad2, nk2, W[None], torch.full((len(sites),), tau0, device=DEV), torch.tensor([eta], device=DEV), one)
        r[f"ours_all|{rate}"] = per_site(site_eval(taus, sites, seen_t)[0])
        if animal is not None:
            r[f"ours_animal|{rate}"] = per_site(site_eval(taus, masked(sites, animal), seen_t)[0])
    rec[seed] = r
    print(f"seed {seed} done", flush=True)

out = {"dataset": ds, "n_sites": len(rec[0]["fcp_all"]), "nonanimal": sorted(NONANIMAL), "frontier_g": list(FRONTIER_G)}
keys = ["fcp_all"] + [f"ours_all|{q}" for q in RATES]
if NAMES is not None:
    keys += ["fcp_animal"] + [f"ours_animal|{q}" for q in RATES]
    out["animal_share_of_known_test_seed0"] = float(np.sum(rec[0]["n_animal"]) / np.sum(rec[0]["n_known"]))
for k in keys:
    per = [summ(rec[sd][k]) for sd in rec]
    out[k] = {m: float(np.mean([p[m] for p in per])) for m in per[0]}
    out[k]["sig_seeds"] = [round(p["sig"], 1) for p in per]
# global static frontier and matched-size comparison
out["gfront"] = {a: {m: float(np.mean([summ(rec[sd]["gfront"][a])[m] for sd in rec])) for m in ("sig", "size", "cov_mean", "shortfall")}
                 for a in rec[0]["gfront"]}
for q in RATES:
    d, clip, fr = [], [], []
    for sd in rec:
        front = sorted((summ(v) for v in rec[sd]["gfront"].values()), key=lambda x: x["size"])
        o = summ(rec[sd][f"ours_all|{q}"])
        fs = float(np.interp(o["size"], [f["size"] for f in front], [f["sig"] for f in front]))
        fr.append(fs)
        d.append(o["sig"] - fs)
        clip.append(o["size"] > front[-1]["size"] or o["size"] < front[0]["size"])
    out[f"matched_global|{q}"] = dict(frontier_sig=float(np.mean(fr)), diff=float(np.mean(d)), diff_seeds=[round(x, 1) for x in d],
                                      clipped_seeds=int(sum(clip)))


def paired(a, b, B=2000, rng=np.random.default_rng(0)):
    ok = [all(rec[sd][a][i]["sig"] is not None and rec[sd][b][i]["sig"] is not None for sd in rec) for i in range(len(rec[0][a]))]
    dc = np.mean([[y_["cov"] - x["cov"] for x, y_, o in zip(rec[sd][a], rec[sd][b], ok) if o] for sd in rec], 0)
    dd = np.mean([[float(y_["sig"]) - float(x["sig"]) for x, y_, o in zip(rec[sd][a], rec[sd][b], ok) if o] for sd in rec], 0)
    bc = [rng.choice(dc, len(dc)).mean() for _ in range(B)]
    bd = [rng.choice(dd, len(dd)).mean() for _ in range(B)]
    return dict(cov_diff=float(dc.mean()), cov_ci=[float(np.quantile(bc, .025)), float(np.quantile(bc, .975))],
                sig_diff_pts=float(100 * dd.mean()), sig_ci_pts=[float(100 * np.quantile(bd, .025)), float(100 * np.quantile(bd, .975))],
                n_sites=len(dc))


if NAMES is not None:
    for t in ("all", "animal"):
        for q in (0.3, 1.0):
            out[f"paired_{t}|{q}"] = paired(f"fcp_{t}", f"ours_{t}|{q}")
    out["sites_without_animal_images_seed0"] = int(sum(x["sig"] is None for x in rec[0]["fcp_animal"]))
json.dump(out, open(f"{ROOT}/results/posthoc_review_{ds}.json", "w"), indent=1, default=float)
json.dump(rec, open(f"{ROOT}/results/posthoc_review_{ds}_persite.json", "w"), default=float)
print(json.dumps(out, indent=1, default=float))
print("REVIEW_DONE")
