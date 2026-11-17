"""POST HOC sensitivity analyses of the lockbox evaluation (lockbox already used once, so not confirmatory).

The method is unchanged: same functions, seeds, label draws, step sizes, proximal weights and thresholds as
lockbox_eval.py and posthoc_animal.py. Only the evaluation of the stored per-image coverage hits is extended.
  * reproduction check: the sequence-bootstrap flag must reproduce the published flagged shares
  * multiplicity: one-sided bootstrap p-value per site for H0 coverage >= 0.9, Benjamini-Hochberg at q = 0.05
    across the lockbox sites of each seed
  * serial dependence: temporal block bootstrap that resamples blocks of b consecutive sequences (b = 5, 10, 20)
  * per-class coverage over lockbox test images, by animal or non-animal class and by class frequency
Usage: python posthoc_sensitivity.py iwildcam|cct
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
import cp_baselines as cp  # noqa: E402
import lockbox_eval as le  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online2 import ETA_L, build_streams, labels_for, pinball, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

if ds == "cct":
    NAMES = pd.read_csv(f"{le.ROOT}/features/cct_meta.csv").drop_duplicates("y").set_index("y").class_name.to_dict()
else:
    c = pd.read_csv(f"{le.ROOT}/data/iwildcam_v2.0/categories.csv")
    NAMES = c[c.y != 99999].set_index("y").name.to_dict()
NONANIMAL = {"empty", "car"}
TAG = "cct_bioclip2crop" if ds == "cct" else "bioclip2crop"
RATES = (0.3, 1.0)
B, BLOCKS = 2000, (5, 10, 20)
sel = json.load(open(f"{le.ROOT}/results/pers_lambda_{TAG}.json"))


def boot_cov(hit, groups, seed=0):
    """Bootstrap distribution of coverage, resampling whole groups (as cp.cluster_upper_bound, same RNG)."""
    _, g = np.unique(groups, return_inverse=True)
    hits = np.bincount(g, weights=hit.astype(float))
    cnts = np.bincount(g).astype(float)
    if len(cnts) < 2:
        return None
    W = np.random.default_rng(seed).multinomial(len(cnts), np.full(len(cnts), 1 / len(cnts)), B)
    return (W @ hits) / (W @ cnts)


def site_stats(hit, seq, codes):
    """Flags for one site: sequence bootstrap (published rule), its p-value, and block bootstraps."""
    bc = boot_cov(hit, seq)
    if bc is None:
        # a single sequence: cp.cluster_upper_bound returns 1.0, so the published rule never flags such a site
        if len(hit) == 0:
            return None
        return dict(cov=float(hit.mean()), n=int(len(hit)), n_seq=1, sig=False, p=1.0,
                    **{f"sig_block{b}": False for b in BLOCKS})
    out = dict(cov=float(hit.mean()), n=int(len(hit)), n_seq=int(len(np.unique(seq))),
               sig=bool(np.quantile(bc, 0.95) < 1 - ALPHA),
               p=float((1 + np.sum(bc >= 1 - ALPHA)) / (B + 1)))
    for b in BLOCKS:
        bb = boot_cov(hit, codes // b)
        out[f"sig_block{b}"] = bool(bb is not None and np.quantile(bb, 0.95) < 1 - ALPHA)
    return out


def bh(pvals, q=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    thresh = q * np.arange(1, len(p) + 1) / len(p)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    rej = np.zeros(len(p), bool)
    rej[order[:k]] = True
    return rej


rec = {}
for seed in range(5):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    mask = ~dev_mask(info)
    X = torch.from_numpy(np.load(f"{le.ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    yname = s.y.map(NAMES).str.lower().to_numpy()
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
    base = [test_all[k] for k in keep]
    one = torch.zeros(1, device=DEV)

    def collect(taus, sites):
        rows, cls = [], []
        for k, st in enumerate(sites):
            kn = ~np.isnan(st["score"])
            t = taus[0, k, torch.as_tensor(st["seq_codes"], device=DEV)].cpu().numpy()
            hit = st["score"][kn] <= t[kn]
            rows.append(site_stats(hit, st["seq"][kn], st["seq_codes"][kn]))
            cls.append(dict(y=y[st["idx"]][kn].tolist(), name=yname[st["idx"]][kn].tolist(), hit=hit.tolist()))
        return rows, cls

    pad, nk = pack(base)
    r, classes = {}, {}
    r["fcp"], classes["fcp"] = collect(torch.full((1, len(base), pad.shape[1]), tau0, device=DEV), base)
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, sel[seed][f"r={rate}"]["lam_star"], seed, score_fn)
        taus = track2(pad, nk, W[None], torch.full((len(sites),), tau0, device=DEV), torch.tensor([eta], device=DEV), one)
        r[f"ours|{rate}"], classes[f"ours|{rate}"] = collect(taus, sites)
    train_counts = pd.Series(y[tr]).value_counts().to_dict()
    rec[seed] = dict(sites=r, classes=classes, train_counts={int(k): int(v) for k, v in train_counts.items()})
    print(f"seed {seed} done", flush=True)

# ---- summaries ----
out = {"dataset": ds, "n_lockbox_sites": len(rec[0]["sites"]["fcp"]), "B": B, "blocks": list(BLOCKS)}
for key in rec[0]["sites"]:
    per_seed = []
    for sd in rec:
        v = [x for x in rec[sd]["sites"][key] if x is not None]
        rej = bh([x["p"] for x in v])
        per_seed.append(dict(sig=100 * np.mean([x["sig"] for x in v]), p05=100 * np.mean([x["p"] < 0.05 for x in v]),
                             bh=100 * np.mean(rej), **{f"block{b}": 100 * np.mean([x[f"sig_block{b}"] for x in v]) for b in BLOCKS},
                             cov=np.mean([x["cov"] for x in v]), n=len(v)))
    out[key] = {m: float(np.mean([d[m] for d in per_seed])) for m in per_seed[0]}
    out[key]["sig_seeds"] = [round(d["sig"], 1) for d in per_seed]
    out[key]["bh_seeds"] = [round(d["bh"], 1) for d in per_seed]

# per-class coverage pooled over lockbox sites, averaged over seeds
for key in rec[0]["classes"]:
    rows = []
    for sd in rec:
        df = pd.DataFrame({c: sum((d[c] for d in rec[sd]["classes"][key]), []) for c in ("y", "name", "hit")})
        g = df.groupby("y").agg(cov=("hit", "mean"), n=("hit", "size"), name=("name", "first")).reset_index()
        g["train"] = g.y.map(rec[sd]["train_counts"]).fillna(0)
        g["animal"] = ~g.name.isin(NONANIMAL)
        g = g[g.n >= 20]                      # classes with at least 20 lockbox test images
        an = g[g.animal]
        terc = np.quantile(an.train, [1 / 3, 2 / 3]) if len(an) >= 3 else [0, 0]
        rare, common = an[an.train <= terc[0]], an[an.train > terc[1]]
        rows.append(dict(n_classes=len(g), n_animal_classes=len(an),
                         animal_median_cov=float(an["cov"].median()), animal_share_below=float(100 * np.mean(an["cov"] < 0.9)),
                         nonanimal_cov=float(np.average(g[~g.animal]["cov"], weights=g[~g.animal].n)) if (~g.animal).any() else np.nan,
                         rare_animal_median_cov=float(rare["cov"].median()), common_animal_median_cov=float(common["cov"].median())))
    out[f"classes|{key}"] = {m: float(np.nanmean([d[m] for d in rows])) for m in rows[0]}

json.dump(out, open(f"{le.ROOT}/results/posthoc_sensitivity_{ds}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
print("SENSITIVITY_DONE")
