"""Does online tracking beat simply using a larger fixed threshold, under the SAME Ditto heads?
Known-class part only, DEVELOPMENT sites, uniform labelling (same draws as pers_openset.py).
  static frontier: tau fixed at the pooled cal quantile for alpha in FRONTIER (eta = 0)
  qt             : tau tracked from tau0(ALPHA) with eta* from the tune replay (as before)
Matched-size comparison, per seed: %sig of the frontier linearly interpolated (in size) at qt's size;
qt wins a seed if its %sig is below the interpolated frontier. Same for coverage-window share.
Usage: python pers_frontier.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from lockbox import dev_mask
from online import DELAY
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from pers_select import simulate
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
RATES = (0.1, 0.3, 1.0)
# fine grid where the tracked sizes fall (3.4-5.2): a coarse grid makes linear interpolation
# overstate the frontier's %sig if the curve is convex, which would flatter tracking
FRONTIER = (0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.06, 0.05, 0.03)


def interp(front, size, key):
    """Frontier value of `key` at `size` (frontier sorted by size; clipped at the ends)."""
    f = sorted(front, key=lambda d: d["size"])
    return float(np.interp(size, [d["size"] for d in f], [d[key] for d in f]))


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
    lam, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam, kreg, seed + 7)  # noqa: E731
    S_glob = score_fn(P_all)
    cal = dict(S=S_glob[cm], y=y[cm], client=client[cm])
    tau_fr = [cp.thresholds("pooled", cal, a, C, K)[0] for a in FRONTIER]
    assert FRONTIER[0] == ALPHA
    tune_sites = [st for st in build_streams(s, "tune", S_glob, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_glob[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    sites_all = build_streams(s, "test", S_glob, y, seqs)
    keep = [k for k, st in enumerate(sites_all) if st is not None and dev[k]]
    base, seen_t = [sites_all[k] for k in keep], seen[keep]

    out = {}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta_star = pos[int(np.argmin(pinball(tt, tune_sites)))]
        rng = np.random.default_rng(10_000 + seed)  # same draws as pers_openset.py (ditto mode)
        sites, Ls = [], []
        for st in base:
            ys = y[st["idx"]]
            P, L, _ = simulate(st, X[st["idx"]], ys, glob, "ditto", rate, "uniform", rng)
            S = score_fn(P)
            sites.append(dict(st, S=S, score=cp.true_scores(S, ys)))
            Ls.append(L)
        pad, nk = pack(sites)
        J = pad.shape[1]
        W = np.zeros((len(sites), J), np.float32)
        for k, L in enumerate(Ls):
            W[k, :len(L)] = L
        W = torch.as_tensor(W, device=DEV)
        G = len(FRONTIER) + 1
        tau0 = torch.tensor(tau_fr + [tau_fr[0]], device=DEV)[:, None].expand(G, len(sites)).contiguous()
        etas = torch.tensor([0.0] * len(FRONTIER) + [eta_star], device=DEV)
        # track2 takes one tau0 per site; run per config so each frontier point has its own start
        taus = torch.cat([track2(pad, nk, W[None], tau0[g], etas[g:g + 1], torch.zeros(1, device=DEV))
                          for g in range(G)])
        ev = [summary(r) for r in site_eval(taus, sites, seen_t)]
        front, qt = ev[:-1], ev[-1]
        out[f"r={rate}"] = dict(frontier={str(a): f for a, f in zip(FRONTIER, front)}, qt=qt, eta=eta_star,
                                front_sig_at_qt_size=interp(front, qt["size"], "sig"),
                                front_win_at_qt_size=interp(front, qt["size"], "win"),
                                front_cov_at_qt_size=interp(front, qt["size"], "cov"))
        print(f"  seed {seed} r={rate}: qt %sig {qt['sig']:.0f} size {qt['size']:.2f} | frontier at that size "
              f"{out[f'r={rate}']['front_sig_at_qt_size']:.0f}", flush=True)
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = [run(sd, tag) for sd in range(n)]
    json.dump(allr, open(f"{ROOT}/results/pers_frontier_fine_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== Ditto heads: static frontier vs online tracking, DEV sites, features={tag}, delay={DELAY}, {n} seeds")
    for key in allr[0]:
        print(f"\n{key}")
        print(f"  {'tau':<16}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'%win<.85':>9}{'size':>7}")
        for a in map(str, FRONTIER):
            m = {k: np.mean([r[key]["frontier"][a][k] for r in allr]) for k in allr[0][key]["frontier"][a]}
            print(f"  {'static a=' + a:<16}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%{m['win']:8.0f}%{m['size']:7.2f}")
        m = {k: np.mean([r[key]["qt"][k] for r in allr]) for k in allr[0][key]["qt"]}
        print(f"  {'qt (tracked)':<16}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%{m['win']:8.0f}%{m['size']:7.2f}")
        d_sig = [r[key]["qt"]["sig"] - r[key]["front_sig_at_qt_size"] for r in allr]
        d_win = [r[key]["qt"]["win"] - r[key]["front_win_at_qt_size"] for r in allr]
        d_cov = [r[key]["qt"]["cov"] - r[key]["front_cov_at_qt_size"] for r in allr]
        print(f"  qt minus frontier at matched size: %sig {np.mean(d_sig):+.1f} (per seed {np.round(d_sig, 1).tolist()}), "
              f"%win {np.mean(d_win):+.1f}, cov {np.mean(d_cov):+.3f}; qt better in "
              f"{sum(d < 0 for d in d_sig)}/{len(d_sig)} seeds")
