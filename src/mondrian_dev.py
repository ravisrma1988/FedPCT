"""Exploratory FedPCT-D (PREREG_MONDRIAN.md): two tracked thresholds per site, one for frames in which
MegaDetector found an animal (conf >= 0.2) and one for the other frames. Development sites only.
FedPCT is recomputed on the same streams as the reference; its all-class numbers must reproduce the
development results (sanity check).
Usage: python mondrian_dev.py iwildcam|cct
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
    MD = pd.read_csv(f"{ROOT}/features/md_best_cct.csv").set_index("row").has_animal
else:
    c = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
    NAMES = c[c.y != 99999].set_index("y").name.to_dict()
    MD = pd.read_csv(f"{ROOT}/features/md_best.csv").set_index("row").has_animal
import cp_baselines as cp  # noqa: E402
import lockbox_eval as le  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online2 import ETA_L, build_streams, labels_for, pinball, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

TAG = "cct_bioclip2crop" if ds == "cct" else "bioclip2crop"
RATES = (0.3, 1.0)
sel = json.load(open(f"{ROOT}/results/pers_lambda_{TAG}.json"))


def site_metrics(S, score, tau_img, seq, keep):
    kn = ~np.isnan(score) & keep
    if kn.sum() == 0:
        return None
    hit = score[kn] <= tau_img[kn]
    return dict(cov=float(hit.mean()), size=float((S[kn] <= tau_img[kn, None]).sum(1).mean()),
                sig=bool(cp.cluster_upper_bound(hit, seq[kn]) < 1 - ALPHA))


rec = {}
for seed in range(5):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    animal = ~s.y.map(NAMES).str.lower().isin({"empty", "car"}).to_numpy()
    grp = MD.reindex(s.row).to_numpy(bool)                       # detector group, known at prediction time
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    tau0 = cp.thresholds("pooled", dict(S=S_all[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    calk = cm & (y != cp.UNK)
    true_cal = cp.true_scores(S_all[calk], y[calk])
    tau0_g = {g: cp.conformal_quantile(true_cal[grp[calk] == g], ALPHA) for g in (False, True)}
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and dev[k]]
    base = [test_all[k] for k in keep]
    one = torch.zeros(1, device=DEV)
    r = {"tau0": float(tau0), "tau0_g": {str(k): float(v) for k, v in tau0_g.items()}}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, sel[seed][f"r={rate}"]["lam_star"], seed, score_fn)
        taus = track2(pad, nk, W[None], torch.full((len(sites),), tau0, device=DEV), torch.tensor([eta], device=DEV), one)[0].cpu().numpy()
        taus_g = {}
        for g in (False, True):
            sg = [dict(st, score=np.where(grp[st["idx"]] == g, st["score"], np.nan)) for st in sites]
            pg, ng = pack(sg)
            taus_g[g] = track2(pg, ng, W[None], torch.full((len(sites),), tau0_g[g], device=DEV),
                               torch.tensor([eta], device=DEV), one)[0].cpu().numpy()
        res = {m: {"all": [], "animal": []} for m in ("fedpct", "fedpct_d")}
        for k, st in enumerate(sites):
            codes = st["seq_codes"]
            t1 = taus[k, codes]
            gi = grp[st["idx"]]
            t2 = np.where(gi, taus_g[True][k, codes], taus_g[False][k, codes])
            an = animal[st["idx"]]
            for m, t in (("fedpct", t1), ("fedpct_d", t2)):
                res[m]["all"].append(site_metrics(st["S"], st["score"], t, st["seq"], np.ones_like(an)))
                res[m]["animal"].append(site_metrics(st["S"], st["score"], t, st["seq"], an))
        r[f"r={rate}"] = res
    rec[seed] = r
    print(f"seed {seed} done (tau0 {tau0:.3f}, detected {tau0_g[True]:.3f}, not detected {tau0_g[False]:.3f})", flush=True)

out = {}
for rate in RATES:
    for m in ("fedpct", "fedpct_d"):
        for cls in ("all", "animal"):
            per_seed = [[x for x in rec[sd][f"r={rate}"][m][cls] if x is not None] for sd in rec]
            out[f"{m}|{cls}|{rate}"] = dict(
                sig=float(np.mean([100 * np.mean([x["sig"] for x in v]) for v in per_seed])),
                sig_seeds=[round(100 * np.mean([x["sig"] for x in v]), 1) for v in per_seed],
                cov=float(np.mean([np.mean([x["cov"] for x in v]) for v in per_seed])),
                below_obs=float(np.mean([100 * np.mean([x["cov"] < 0.9 for x in v]) for v in per_seed])),
                size=float(np.mean([np.mean([x["size"] for x in v]) for v in per_seed])),
                n_sites=len(per_seed[0]))
json.dump({"summary": out, "seeds": rec}, open(f"{ROOT}/results/mondrian_dev_{ds}.json", "w"), default=float)
print(f"\n== FedPCT-D exploratory, {ds} development sites, five seeds")
for k, v in out.items():
    print(f"{k:24s} flagged {v['sig']:5.1f}% {v['sig_seeds']} | cov {v['cov']:.3f} | obs<0.9 {v['below_obs']:5.1f}% | size {v['size']:.2f} | sites {v['n_sites']}")
print("MONDRIAN_DONE")
