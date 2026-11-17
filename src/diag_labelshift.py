"""Hypothesis: much of the per-site deployment drift is label shift (species mix), with class-
conditional score distributions roughly shared across sites. Static set rules, DEVELOPMENT sites
only (lockbox excluded), frontier over alpha':

  pooled        one threshold from pooled cal
  classcond     per-class thresholds from pooled cal (federation needed for per-class data);
                classes with < N_MIN cal images fall back to the pooled threshold
  ls_cal        weighted quantile, weights pi_k^cal(y)/pi_pool(y)  (site's cal species mix)
  ls_em         weights from the site's deployment species mix estimated WITHOUT labels by EM on
                its test predictions (Saerens et al. 2002), transductive over the test period
  ls_oracle     weights from the site's TRUE test species mix  (upper bound, not a method)
Coverage/size on known-class test images; significance by sequence-cluster bootstrap.
Usage: python diag_labelshift.py <feature tag> <n_seeds>
"""
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from lockbox import dev_mask
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT, N_MIN = "${FEDPCT_ROOT}", 30
ALPHAS = (0.1, 0.07, 0.05, 0.03)


def em_prior(P, prior_train, iters=100):
    pi = prior_train.copy()
    for _ in range(iters):
        q = P * (pi / prior_train)[None, :]
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < 1e-6:
            break
        pi = new
    return pi


def wquant(s, w, a):
    return cp.conformal_quantile(s, a, weights=w)


def evaluate(sets_fn, sites, a):
    out = []
    for st in sites:
        sets = sets_fn(st, a)
        hit = sets[np.arange(len(st["y"])), st["y"]]
        out.append(dict(cov=hit.mean(), size=sets.sum(1).mean(), seen=st["seen"],
                        sig=cp.cluster_upper_bound(hit, st["seq"]) < 1 - ALPHA))
    return dict(cov=np.mean([o["cov"] for o in out]), size=np.mean([o["size"] for o in out]),
                sig=100 * np.mean([o["sig"] for o in out]),
                sig_unseen=100 * np.mean([o["sig"] for o in out if not o["seen"]]))


def run(seed, tag):
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    head = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P = probs_of(head, X)
    tm = part == "tune"
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    S = scores.raps(P, lam, kreg, seed + 7)
    cm = (part == "cal") & (y != cp.UNK)
    sc, yc, kc = cp.true_scores(S[cm], y[cm]), y[cm], client[cm]
    pi_pool = np.bincount(yc, minlength=C) + 1.0
    pi_pool /= pi_pool.sum()
    pi_train = np.bincount(y[tr], minlength=C) + 1.0
    pi_train /= pi_train.sum()

    sites = []
    for k in np.flatnonzero(dev):
        m = (part == "test") & (client == k)
        kn = m & (y != cp.UNK)
        if kn.sum() < 10:
            continue
        pi_cal = np.bincount(yc[kc == k], minlength=C) + 1.0
        pi_true = np.bincount(y[kn], minlength=C) + 1.0
        sites.append(dict(S=S[kn], y=y[kn], seq=seqs[kn], seen=bool(seen[k]),
                          pi_cal=pi_cal / pi_cal.sum(), pi_true=pi_true / pi_true.sum(),
                          pi_em=em_prior(P[m], pi_train)))  # EM uses all test images, no labels

    rules = {
        "pooled": lambda st, a: st["S"] <= wquant(sc, None, a),
        "classcond": None,
        "ls_cal": lambda st, a: st["S"] <= wquant(sc, st["pi_cal"][yc] / pi_pool[yc], a),
        "ls_em": lambda st, a: st["S"] <= wquant(sc, st["pi_em"][yc] / pi_pool[yc], a),
        "ls_oracle": lambda st, a: st["S"] <= wquant(sc, st["pi_true"][yc] / pi_pool[yc], a),
    }
    cache = {}

    def classcond(st, a):
        if a not in cache:
            t = np.full(C, wquant(sc, None, a))
            for c in range(C):
                mc = yc == c
                if mc.sum() >= N_MIN:
                    t[c] = wquant(sc[mc], None, a)
            cache[a] = t
        return st["S"] <= cache[a][None, :]

    rules["classcond"] = classcond
    return {name: {a: evaluate(fn, sites, a) for a in ALPHAS} for name, fn in rules.items()}, len(sites)


if __name__ == "__main__":
    tag, n = sys.argv[1], int(sys.argv[2])
    allr = []
    for sd in range(n):
        r, ns = run(sd, tag)
        allr.append(r)
        print(f"seed {sd} done ({ns} dev sites)", flush=True)
    print(f"\n== label-shift hypothesis, static, DEV sites, features={tag}, RAPS, {n} seeds")
    print(f"{'rule':<12}{'alpha':>6}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'size':>7}")
    for name in allr[0]:
        for a in ALPHAS:
            m = {k: np.mean([r[name][a][k] for r in allr]) for k in allr[0][name][a]}
            print(f"{name:<12}{a:6.2f}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%{m['size']:7.2f}")
