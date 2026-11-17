"""Exploratory, DEVELOPMENT SITES ONLY: does a per-site starting threshold help FedPCT?

The frozen method starts every site at the pooled conformal quantile tau0 and then tracks it online.
Drift is mostly site-specific (diag_common_drift.py), so a better per-site START may help where
tracking is slow (low labelling rate, short streams). Starts compared (all computed before the test
period from global-head RAPS scores on the calibration partition, so Theorem 1 holds unchanged):

  pooled      tau0 (the frozen method)
  local       the site's own conformal quantile on its calibration images (fallback tau0)
  shrink      tau0 + w_k (q_k - tau0), w_k = s2 / (s2 + v_k): empirical Bayes shrinkage of the local
              offset towards zero; v_k = sequence-bootstrap variance of q_k, s2 = between-site variance
  fedprior    tau0 + p_k, p_k predicted from a label-free site descriptor (mean global-head class
              probabilities over the site's calibration images) by ridge regression across sites,
              cross-fitted over sites (a site's prediction never uses its own offset). Ridge needs only
              the sums X'WX and X'Wo, which sites can send to the server (federated).
  fedprior_eb tau0 + p_k + w_k (q_k - tau0 - p_k), shrinkage towards the federated prior

Each start is followed by Ditto(lambda*) + tracking at the frozen eta (the method), and also reported
with eta = 0 (start alone). Lockbox test streams are never touched: the evaluation mask is the
development mask; calibration partitions of all sites are used, as for tau0 in the frozen protocol.
Usage: python site_prior_dev.py iwildcam|cct
"""
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "${FEDPCT_ROOT}/src")
if sys.argv[1] == "cct":
    import cct_env
    cct_env.use_cct()
import lockbox_eval as le  # noqa: E402

if sys.argv[1] == "cct":
    le.TAG = cct_env.TAG

import cp_baselines as cp  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

ROOT = le.ROOT
MINCAL, NBOOT, NFOLD = 20, 200, 5
STARTS = ("pooled", "local", "shrink", "fedprior", "fedprior_eb")


def site_quantiles(sc, sq, rng):
    """Conformal quantile of one site's calibration true scores and its sequence-bootstrap variance."""
    q = cp.conformal_quantile(sc, ALPHA)
    u, inv = np.unique(sq, return_inverse=True)
    bs = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(u), len(u))
        idx = np.concatenate([np.flatnonzero(inv == p) for p in pick])
        b = cp.conformal_quantile(sc[idx], ALPHA)
        bs.append(b if np.isfinite(b) else sc.max())
    return q, float(np.var(bs))


def starts_for(s, S_all, P_all, y, client, part, seqs, tau0, K, seed):
    rng = np.random.default_rng(50_000 + seed)
    cm = part == "cal"
    off, var, desc, ok = np.zeros(K), np.full(K, np.inf), np.zeros((K, P_all.shape[1])), np.zeros(K, bool)
    for k in range(K):
        m = cm & (client == k)
        if m.sum() == 0:
            continue
        desc[k] = P_all[m].mean(0)                     # label-free descriptor
        kn = m & (y != cp.UNK)
        if kn.sum() < MINCAL:
            continue
        sc = cp.true_scores(S_all[kn], y[kn])
        q, v = site_quantiles(sc, seqs[kn], rng)
        if np.isfinite(q):
            off[k], var[k], ok[k] = q - tau0, v, True
    # between-site variance of the local offsets (method of moments)
    w0 = 1 / var[ok]
    s2_loc = max(0.0, np.average((off[ok] - np.average(off[ok], weights=w0)) ** 2, weights=w0) - np.median(var[ok]))
    # federated ridge prior, cross-fitted over sites
    Z = np.sqrt(desc)                                  # Hellinger-type transform of the class mix
    pred = np.zeros(K)
    idx = np.flatnonzero(ok)
    fold = rng.permutation(len(idx)) % NFOLD
    has_desc = desc.sum(1) > 0
    for f in range(NFOLD):
        tr = idx[fold != f]
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], off[tr], sample_weight=1 / (var[tr] + s2_loc + 1e-9))
        te = np.r_[idx[fold == f], np.flatnonzero(~ok & has_desc)] if f == 0 else idx[fold == f]
        pred[te] = mdl.predict(Z[te])
    res = off[ok] - pred[ok]
    s2_res = max(0.0, np.average(res ** 2, weights=w0) - np.median(var[ok]))
    w_loc = np.where(ok, s2_loc / (s2_loc + var), 0.0)
    w_res = np.where(ok, s2_res / (s2_res + var), 0.0)
    st = dict(pooled=np.full(K, tau0),
              local=np.where(ok, tau0 + off, tau0),
              shrink=tau0 + w_loc * np.where(ok, off, 0.0),
              fedprior=np.where(has_desc, tau0 + pred, tau0),
              fedprior_eb=np.where(has_desc, tau0 + pred + w_res * np.where(ok, off - pred, 0.0), tau0))
    diag = dict(n_ok=int(ok.sum()), s2_loc=s2_loc, s2_res=s2_res, med_var=float(np.median(var[ok])),
                mean_w_loc=float(w_loc[ok].mean()), mean_w_res=float(w_res[ok].mean()),
                r2_prior=float(1 - np.average(res ** 2, weights=w0) / np.average((off[ok] - np.average(off[ok], weights=w0)) ** 2, weights=w0)))
    return {k: np.maximum(v, 0.0) for k, v in st.items()}, diag


def run(seed, lam_star):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    mask = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{le.TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    tau0 = cp.thresholds("pooled", dict(S=S_all[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    st0, diag = starts_for(s, S_all, P_all, y, client, part, seqs, tau0, K, seed)
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and mask[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    out = dict(n_sites=len(keep), n_unseen=int((~seen_t).sum()), diag=diag)
    for rate in le.RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, lam_star[rate], seed, score_fn)
        r = dict(eta=eta)
        for name in STARTS:
            t0 = torch.as_tensor(st0[name][keep], dtype=torch.float32, device=DEV)
            taus = torch.cat([track2(pad, nk, W[None], t0, torch.tensor([e], device=DEV), torch.zeros(1, device=DEV))
                              for e in (eta, 0.0)])
            ev = site_eval(taus, sites, seen_t)
            r[name] = summary(ev[0])
            r[name + "_fixed"] = summary(ev[1])
        out[f"r={rate}"] = r
        print(f"  seed {seed} r={rate} eta={eta}: " + "  ".join(f"{n} {r[n]['sig']:.0f}%/{r[n]['size']:.2f}" for n in STARTS),
              flush=True)
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    sel = json.load(open(f"{ROOT}/results/pers_lambda_{le.TAG}.json"))
    allr = []
    for sd in le.SEEDS:
        allr.append(run(sd, {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in le.RATES}))
        print(f"seed {sd} done  diag {allr[-1]['diag']}", flush=True)
    json.dump(allr, open(f"{ROOT}/results/site_prior_dev_{ds}.json", "w"), indent=1, default=float)
    print(f"\n== site prior, {ds} DEVELOPMENT sites ({allr[0]['n_sites']} sites, {allr[0]['n_unseen']} unseen), 5 seeds")
    print(f"{'start':<14}{'rate':>5}{'track %sig':>12}{'unseen':>8}{'size':>7}{'cov':>7}{'fixed %sig':>12}{'size':>7}   per-seed")
    for rate in le.RATES:
        for n in STARTS:
            g = lambda key, r: np.nanmean([a[f"r={rate}"][key][r] for a in allr])  # noqa: E731
            print(f"{n:<14}{rate:>5}{g(n, 'sig'):11.1f}%{g(n, 'sig_unseen'):7.0f}%{g(n, 'size'):7.2f}{g(n, 'cov'):7.3f}"
                  f"{g(n + '_fixed', 'sig'):11.1f}%{g(n + '_fixed', 'size'):7.2f}   "
                  f"{[round(a[f'r={rate}'][n]['sig']) for a in allr]}")
    print("SITE_PRIOR_DONE")
