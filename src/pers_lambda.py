"""Federated selection of the Ditto strength lambda (so it is not a hand-picked constant), plus the
test sensitivity to lambda for the paper. DEVELOPMENT sites, crop features.

Selection (test never used): each development site replays its own held-out CAL+TUNE block (time
order; the block that precedes test) as a feedback stream at the deployment labelling rate r, with
the same Ditto refit rule, and reports its prequential log-loss on known-class images (prediction
made by the head in force BEFORE that image's label arrives) for every lambda in LAMS. The server
averages the per-site losses weighted by image count and picks lambda*(r). Federated: only one
scalar per (site, lambda) leaves a site. Using cal images here does not touch tau0/theta0, which
are computed from the global head.
Test (report): for every lambda, known part (fixed tau, tracked tau) + unknown flag (fixed theta).
Usage: python pers_lambda.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import cp_baselines as cp
import scores
from lockbox import dev_mask
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from openset import ALPHA_U, MIN_UNK
from pers_select import simulate
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
RATES = (0.1, 0.3, 1.0)
LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)


def run(seed, tag):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_glob = score_fn(P_all)
    tau0 = cp.thresholds("pooled", dict(S=S_glob[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    th0 = cp.conformal_quantile(-(1 - P_all.max(1))[cm & (y == cp.UNK)], ALPHA_U)
    tune_sites = [st for st in build_streams(s, "tune", S_glob, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_glob[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]

    s_sel = s.copy()
    s_sel.loc[s_sel.part == "tune", "part"] = "cal"                     # whole held-out block, time order
    sel_all = build_streams(s_sel, "cal", S_glob, y, seqs)
    sel_sites = [st for k, st in enumerate(sel_all) if st is not None and dev[k]]
    test_all = build_streams(s, "test", S_glob, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and dev[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]

    out = {}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta_star = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sel = {}
        for lam in LAMS:
            rng = np.random.default_rng(30_000 + seed)
            tot = cnt = 0.0
            for st in sel_sites:
                ys = y[st["idx"]]
                P, _, _ = simulate(st, X[st["idx"]], ys, glob, "ditto", rate, "uniform", rng, lam)
                kn = ys != cp.UNK
                tot += -np.log(np.clip(P[kn, ys[kn]], 1e-12, None)).sum()
                cnt += kn.sum()
            sel[lam] = tot / cnt
        lam_star = min(sel, key=sel.get)
        res = {}
        for lam in LAMS:
            rng = np.random.default_rng(10_000 + seed)  # same test label draws as pers_openset.py
            sites, Ls, nov, yy = [], [], [], []
            for st in base:
                ys = y[st["idx"]]
                P, L, _ = simulate(st, X[st["idx"]], ys, glob, "ditto", rate, "uniform", rng, lam)
                S = score_fn(P)
                sites.append(dict(st, S=S, score=cp.true_scores(S, ys)))
                Ls.append(L), nov.append(1 - P.max(1)), yy.append(ys)
            pad, nk = pack(sites)
            W = np.zeros((len(sites), pad.shape[1]), np.float32)
            for k, L in enumerate(Ls):
                W[k, :len(L)] = L
            taus = track2(pad, nk, torch.as_tensor(W, device=DEV)[None].expand(2, -1, -1),
                          torch.full((len(sites),), tau0, device=DEV), torch.tensor([0.0, eta_star], device=DEV),
                          torch.zeros(2, device=DEV))
            ev = site_eval(taus, sites, seen_t)
            rec_n = rec_d = 0
            usig, ff = [], []
            for st, n_, ys in zip(sites, nov, yy):
                flag, unk = -n_ < th0, ys == cp.UNK
                rec_n += flag[unk].sum(); rec_d += unk.sum()
                if unk.sum() >= MIN_UNK:
                    usig.append(cp.cluster_upper_bound(flag[unk], st["seq"][unk]) < 1 - ALPHA_U)
                ff.append(flag[~unk].mean())
            ya, na = np.concatenate(yy), np.concatenate(nov)
            res[str(lam)] = dict(fixed=summary(ev[0]), qt=summary(ev[1]), recall=rec_n / rec_d,
                                 usig=100 * float(np.mean(usig)), ff=100 * float(np.mean(ff)),
                                 auroc=float(roc_auc_score(ya == cp.UNK, na)))
        out[f"r={rate}"] = dict(sel_loss={str(k): v for k, v in sel.items()}, lam_star=lam_star, eta=eta_star, test=res)
        print(f"  seed {seed} r={rate}: lambda* {lam_star} | sel loss " +
              " ".join(f"{k}:{v:.3f}" for k, v in sel.items()), flush=True)
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = [run(sd, tag) for sd in range(n)]
    json.dump(allr, open(f"{ROOT}/results/pers_lambda_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== Ditto lambda: federated selection + test sensitivity, DEV sites, features={tag}, {n} seeds")
    for key in allr[0]:
        ch = [r[key]["lam_star"] for r in allr]
        print(f"\n{key}: lambda* per seed {ch}")
        print(f"  {'lambda':<8}{'sel_loss':>9} | {'fixed %sig':>10}{'size':>6} | {'qt %sig':>8}{'size':>6} | "
              f"{'recall':>7}{'ff%':>7}{'auroc':>7}")
        for lam in map(str, LAMS):
            g = lambda f: np.mean([f(r[key]) for r in allr])  # noqa: E731
            print(f"  {lam:<8}{g(lambda d: d['sel_loss'][lam]):9.4f} | {g(lambda d: d['test'][lam]['fixed']['sig']):9.0f}%"
                  f"{g(lambda d: d['test'][lam]['fixed']['size']):6.2f} | {g(lambda d: d['test'][lam]['qt']['sig']):7.0f}%"
                  f"{g(lambda d: d['test'][lam]['qt']['size']):6.2f} | {g(lambda d: d['test'][lam]['recall']):7.3f}"
                  f"{g(lambda d: d['test'][lam]['ff']):6.1f}%{g(lambda d: d['test'][lam]['auroc']):7.3f}")
        d =[r[key]["test"][str(r[key]["lam_star"])] for r in allr]
        print(f"  selected: fixed %sig {np.mean([x['fixed']['sig'] for x in d]):.0f}% size "
              f"{np.mean([x['fixed']['size'] for x in d]):.2f} | qt %sig {np.mean([x['qt']['sig'] for x in d]):.0f}% size "
              f"{np.mean([x['qt']['size'] for x in d]):.2f} | ff {np.mean([x['ff'] for x in d]):.1f}%")
