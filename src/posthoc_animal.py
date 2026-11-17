"""POST HOC analysis requested by the pre-submission audit (lockbox sites, already used once, so not
confirmatory). The method is unchanged (same functions, seeds and label draws as lockbox_eval.py);
only the evaluation is extended:
  * all-class metrics (must reproduce the published lockbox tables: sanity check)
  * animal-only metrics: coverage and set size over known-class images whose class is an animal
    (excluding 'empty' and, in CCT, 'car'); thresholds and models are exactly those of the method
  * per-site coverage distribution: share of sites with observed coverage below 0.9, median and
    10th percentile of site coverage
  * paired site-level comparison FedPCT minus FCP: per site, mean over seeds of the coverage
    difference and of the deficiency indicator difference; 95% bootstrap interval over sites
Usage: python posthoc_animal.py iwildcam|cct
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

ds = sys.argv[1]
if ds == "cct":
    import cct_env
    cct_env.use_cct()
    NAMES = pd.read_csv("${FEDPCT_ROOT}/features/cct_meta.csv").drop_duplicates("y").set_index("y").class_name.to_dict()
else:
    c = pd.read_csv("${FEDPCT_ROOT}/data/iwildcam_v2.0/categories.csv")
    NAMES = c[c.y != 99999].set_index("y").name.to_dict()
import cp_baselines as cp  # noqa: E402
import lockbox_eval as le  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

NONANIMAL = {"empty", "car"}
TAG = "cct_bioclip2crop" if ds == "cct" else "bioclip2crop"
RATES = (0.3, 1.0)
sel = json.load(open(f"${FEDPCT_ROOT}/results/pers_lambda_{TAG}.json"))


def masked(sites, animal):
    return [dict(st, score=np.where(animal[st["idx"]], st["score"], np.nan)) for st in sites]


def per_site(ev):
    # sites with no evaluable image (e.g. no animal images in the test period) get cov = NaN
    return [dict(cov=r["cov"], size=r["size"], sig=bool(r["sig"]) if r["cov"] == r["cov"] else None, seen=r["seen"]) for r in ev]


rec = {}
for seed in range(5):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    mask = ~dev_mask(info)                                   # lockbox sites
    X = torch.from_numpy(np.load(f"{le.ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    animal = ~s.y.map(NAMES).str.lower().isin(NONANIMAL).to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    tau0 = cp.thresholds("pooled", dict(S=S_all[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and mask[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    one = torch.zeros(1, device=DEV)
    # FCP: global classifier, fixed pooled threshold
    pad, nk = pack(base)
    fixed = torch.full((1, len(base), pad.shape[1]), tau0, device=DEV)
    r = dict(fcp_all=per_site(site_eval(fixed, base, seen_t)[0]),
             fcp_animal=per_site(site_eval(fixed, masked(base, animal), seen_t)[0]),
             n_animal=[int(np.sum(animal[st["idx"]] & ~np.isnan(st["score"]))) for st in base],
             n_known=[int(np.sum(~np.isnan(st["score"]))) for st in base])
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, sel[seed][f"r={rate}"]["lam_star"], seed, score_fn)
        taus = track2(pad, nk, W[None], torch.full((len(sites),), tau0, device=DEV), torch.tensor([eta], device=DEV), one)
        r[f"ours_all|{rate}"] = per_site(site_eval(taus, sites, seen_t)[0])
        r[f"ours_animal|{rate}"] = per_site(site_eval(taus, masked(sites, animal), seen_t)[0])
    rec[seed] = r
    print(f"seed {seed} done", flush=True)


def agg(key):
    v = [[x for x in rec[sd][key] if x["sig"] is not None] for sd in rec]  # drop sites without images
    sig = [100 * np.mean([x["sig"] for x in v_]) for v_ in v]
    cov = [np.array([x["cov"] for x in v_]) for v_ in v]
    return dict(sig=np.mean(sig), sig_seeds=[round(x, 1) for x in sig], size=np.mean([np.mean([x["size"] for x in v_]) for v_ in v]),
                cov_mean=np.mean([c.mean() for c in cov]), below_obs=100 * np.mean([np.mean(c < 0.9) for c in cov]),
                cov_median=np.mean([np.median(c) for c in cov]), cov_p10=np.mean([np.quantile(c, .1) for c in cov]))


def paired(a, b, B=2000, rng=np.random.default_rng(0)):
    """Per site: mean over seeds of (b - a) coverage and deficiency; 95% bootstrap CI over sites."""
    ok = [all(rec[sd][a][i]["sig"] is not None and rec[sd][b][i]["sig"] is not None for sd in rec) for i in range(len(rec[0][a]))]
    dc = np.mean([[y_["cov"] - x["cov"] for x, y_, o in zip(rec[sd][a], rec[sd][b], ok) if o] for sd in rec], 0)
    dd = np.mean([[float(y_["sig"]) - float(x["sig"]) for x, y_, o in zip(rec[sd][a], rec[sd][b], ok) if o] for sd in rec], 0)
    bc = [rng.choice(dc, len(dc)).mean() for _ in range(B)]
    bd = [rng.choice(dd, len(dd)).mean() for _ in range(B)]
    return dict(cov_diff=float(dc.mean()), cov_ci=[float(np.quantile(bc, .025)), float(np.quantile(bc, .975))],
                sig_diff_pts=float(100 * dd.mean()), sig_ci_pts=[float(100 * np.quantile(bd, .025)), float(100 * np.quantile(bd, .975))],
                n_sites=len(dc))


out = {"dataset": ds, "n_sites": len(rec[0]["fcp_all"]),
       "animal_share_of_known_test": float(np.sum(rec[0]["n_animal"]) / np.sum(rec[0]["n_known"])),
       "animal_images_per_site_median": float(np.median(rec[0]["n_animal"]))}
for k in ["fcp_all", "fcp_animal"] + [f"ours_{t}|{r}" for t in ("all", "animal") for r in RATES]:
    out[k] = agg(k)
for t in ("all", "animal"):
    for r in RATES:
        out[f"paired_{t}|{r}"] = paired(f"fcp_{t}", f"ours_{t}|{r}")
out["sites_without_animal_images"] = int(sum(x["sig"] is None for x in rec[0]["fcp_animal"]))
json.dump(out, open(f"${FEDPCT_ROOT}/results/posthoc_animal_{ds}.json", "w"), indent=1, default=float)
json.dump(rec, open(f"${FEDPCT_ROOT}/results/posthoc_animal_{ds}_persite.json", "w"), default=float)
print(json.dumps(out, indent=1, default=float))
print("POSTHOC_DONE")
