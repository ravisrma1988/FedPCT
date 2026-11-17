"""FedPCT-CC (PREREG_CLASSCOND.md): one tracked threshold for animal classes and one for non-animal
classes (empty, car) per site; class c enters the set when s(x, c) <= tau_g(c). Development sites only.
FedPCT is recomputed on the same streams as the reference; its all-class numbers must reproduce the
development results (sanity check). Refuses to run if the protocol or this script changed after freezing.
Usage: python classcond_dev.py iwildcam|cct
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

ds = sys.argv[1]
ROOT = "${FEDPCT_ROOT}"
frozen = json.load(open(f"{ROOT}/splits/prereg_classcond.json"))
for f, h in (("PREREG_CLASSCOND.md", frozen["PREREG_CLASSCOND.md"]), ("src/classcond_dev.py", frozen["classcond_dev.py"])):
    assert hashlib.sha256(open(f"{ROOT}/{f}", "rb").read()).hexdigest() == h, f"{f} changed after freezing"
if ds == "cct":
    import cct_env
    cct_env.use_cct()
    NAMES = pd.read_csv(f"{ROOT}/features/cct_meta.csv").drop_duplicates("y").set_index("y").class_name.to_dict()
else:
    c = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
    NAMES = c[c.y != 99999].set_index("y").name.to_dict()
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
NONANIMAL = {"empty", "car"}
sel = json.load(open(f"{ROOT}/results/pers_lambda_{TAG}.json"))


def site_metrics(S, score, tau_cls, tau_img, seq, keep):
    """tau_cls (n, C): per-image, per-class threshold; tau_img (n,): threshold of the true class."""
    kn = ~np.isnan(score) & keep
    if kn.sum() == 0:
        return None
    hit = score[kn] <= tau_img[kn]
    return dict(cov=float(hit.mean()), size=float((S[kn] <= tau_cls[kn]).sum(1).mean()),
                sig=bool(cp.cluster_upper_bound(hit, seq[kn]) < 1 - ALPHA))


rec = {}
for seed in range(5):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    dev = dev_mask(info)
    cls_animal = np.array([NAMES[c].lower() not in NONANIMAL for c in info["known_y"]])   # by label index
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    img_animal = np.where(y >= 0, cls_animal[np.clip(y, 0, None)], False)                 # true class animal
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
    tau0_g = {g: cp.conformal_quantile(true_cal[img_animal[calk] == g], ALPHA) for g in (False, True)}
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
            sg = [dict(st, score=np.where(img_animal[st["idx"]] == g, st["score"], np.nan)) for st in sites]
            pg, ng = pack(sg)
            taus_g[g] = track2(pg, ng, W[None], torch.full((len(sites),), tau0_g[g], device=DEV),
                               torch.tensor([eta], device=DEV), one)[0].cpu().numpy()
        res = {m: {"all": [], "animal": [], "nonanimal": []} for m in ("fedpct", "fedpct_cc")}
        for k, st in enumerate(sites):
            codes = st["seq_codes"]
            n = len(st["idx"])
            t1 = taus[k, codes]
            ta, tn = taus_g[True][k, codes], taus_g[False][k, codes]
            an = img_animal[st["idx"]]
            cls1 = np.repeat(t1[:, None], C, 1)
            cls2 = np.where(cls_animal[None, :], ta[:, None], tn[:, None])
            img2 = np.where(an, ta, tn)
            for m, tc, ti in (("fedpct", cls1, t1), ("fedpct_cc", cls2, img2)):
                res[m]["all"].append(site_metrics(st["S"], st["score"], tc, ti, st["seq"], np.ones(n, bool)))
                res[m]["animal"].append(site_metrics(st["S"], st["score"], tc, ti, st["seq"], an))
                res[m]["nonanimal"].append(site_metrics(st["S"], st["score"], tc, ti, st["seq"], ~an))
        r[f"r={rate}"] = res
    rec[seed] = r
    print(f"seed {seed} done (tau0 {tau0:.3f}, animal {tau0_g[True]:.3f}, non-animal {tau0_g[False]:.3f})", flush=True)

out = {}
for rate in RATES:
    for m in ("fedpct", "fedpct_cc"):
        for cls in ("all", "animal", "nonanimal"):
            per_seed = [[x for x in rec[sd][f"r={rate}"][m][cls] if x is not None] for sd in rec]
            out[f"{m}|{cls}|{rate}"] = dict(
                sig=float(np.mean([100 * np.mean([x["sig"] for x in v]) for v in per_seed])),
                sig_seeds=[round(100 * np.mean([x["sig"] for x in v]), 1) for v in per_seed],
                cov=float(np.mean([np.mean([x["cov"] for x in v]) for v in per_seed])),
                below_obs=float(np.mean([100 * np.mean([x["cov"] < 0.9 for x in v]) for v in per_seed])),
                size=float(np.mean([np.mean([x["size"] for x in v]) for v in per_seed])),
                n_sites=[len(v) for v in per_seed])
a, b = out["fedpct|animal|1.0"], out["fedpct_cc|animal|1.0"]
red = [x - y_ for x, y_ in zip(a["sig_seeds"], b["sig_seeds"])]
size_ratio = out["fedpct_cc|all|1.0"]["size"] / out["fedpct|all|1.0"]["size"]
crit = dict(animal_reduction_pts=a["sig"] - b["sig"], seeds_reduced=int(sum(x > 0 for x in red)), size_ratio=size_ratio,
            met=bool(a["sig"] - b["sig"] >= 5 and sum(x > 0 for x in red) >= 4 and size_ratio <= 1.25))
json.dump({"summary": out, "criterion": crit, "seeds": rec}, open(f"{ROOT}/results/classcond_dev_{ds}.json", "w"), default=float)
print(f"\n== FedPCT-CC, {ds} development sites, five seeds")
for k, v in out.items():
    print(f"{k:26s} flagged {v['sig']:5.1f}% {v['sig_seeds']} | cov {v['cov']:.3f} | obs<0.9 {v['below_obs']:5.1f}% | size {v['size']:.2f} | sites {v['n_sites']}")
print("criterion (r=1.0):", crit)
print("CLASSCOND_DONE")
