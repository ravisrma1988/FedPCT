"""Selective (uncertainty-driven) labelling + federated personalisation + importance-weighted
online conformal tracking. DEVELOPMENT sites only.

Per site, sequence by sequence (time order):
  1. predict with the head in force (global, or Ditto-personalised on feedback received so far)
  2. uncertainty u_j = mean over the sequence of (1 - max prob)
  3. labelling probability (decided before any label is seen, from observed quantities only):
       uniform     p_j = r
       uncertainty p_j = clip(r * u_j / ubar_{<j}, P_MIN_FRAC * r, 1),  ubar = running mean of u
     L_j ~ Bernoulli(p_j); labels of sequence j arrive before sequence j + DELAY + 1
  4. every REFIT_EVERY newly arrived labelled known-class images the Ditto head is refitted
Thresholds (start: pooled cal quantile): eta = 0 (frozen) or eta* (tune replay, as before), with
feedback weights  iw: L_j / p_j  (IM-OCP-style importance weighting, Wang et al. 2025)  or
naive: L_j. Policy and refit hyper-parameters fixed a priori.
Usage: python pers_select.py <n_seeds> <feature tag>
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch

import cp_baselines as cp
import scores
from lockbox import dev_mask
from online import DELAY
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from pers_online import DITTO_LAMBDA, REFIT_EVERY, ditto
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
RATES = (0.1, 0.3)
P_MIN_FRAC = 0.2
POLICIES = ("uniform", "uncertainty")
MODES = ("global", "ditto")


def simulate(st, Xs, ys, glob, mode, rate, policy, rng, lam=DITTO_LAMBDA):
    """Returns P (n, C) under the head in force at prediction time, L (J,), p (J,)."""
    codes = st["seq_codes"]
    J = codes.max() + 1
    bounds = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(codes)]
    known = ys != cp.UNK
    with torch.no_grad():
        P = torch.softmax(glob(Xs * 10), 1)
    Pout = torch.empty_like(P)
    L, p = np.zeros(J, bool), np.zeros(J)
    arrivals, fb, refits, ubar, nu = defaultdict(list), [], 0, 0.0, 0
    for j in range(J):
        for m in arrivals.pop(j, []):
            fb.extend(r for r in range(bounds[m], bounds[m + 1]) if known[r])
        if mode == "ditto" and len(fb) // REFIT_EVERY > refits:
            refits = len(fb) // REFIT_EVERY
            h = ditto(glob, Xs[fb], torch.as_tensor(ys[fb], device=DEV), lam)
            with torch.no_grad():
                P[bounds[j]:] = torch.softmax(h(Xs[bounds[j]:] * 10), 1)
        rows = slice(bounds[j], bounds[j + 1])
        Pout[rows] = P[rows]
        u = float((1 - P[rows].max(1).values).mean())
        if policy == "uniform":
            p[j] = rate
        else:
            ref = ubar if nu > 0 else u
            p[j] = float(np.clip(rate * u / max(ref, 1e-6), P_MIN_FRAC * rate, 1.0))
        ubar, nu = (ubar * nu + u) / (nu + 1), nu + 1
        L[j] = rng.random() < p[j]
        if L[j]:
            arrivals[j + DELAY + 1].append(j)
    return Pout.cpu().numpy(), L, p


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
    tau0 = cp.thresholds("pooled", dict(S=S_glob[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
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
        for mode in MODES:
            for policy in POLICIES:
                rng = np.random.default_rng(10_000 + seed)  # same draws across modes/policies
                sites, Ls, ps = [], [], []
                for st in base:
                    ys = y[st["idx"]]
                    P, L, p = simulate(st, X[st["idx"]], ys, glob, mode, rate, policy, rng)
                    S = score_fn(P)
                    sites.append(dict(st, S=S, score=cp.true_scores(S, ys)))
                    Ls.append(L), ps.append(p)
                pad, nk = pack(sites)
                J = pad.shape[1]
                W = {"iw": np.zeros((len(sites), J), np.float32), "naive": np.zeros((len(sites), J), np.float32)}
                for k, (L, p) in enumerate(zip(Ls, ps)):
                    W["iw"][k, :len(L)] = L / p
                    W["naive"][k, :len(L)] = L
                # IW weights L/p have mean ~1 per sequence, naive weights mean ~r: IW gets eta*r so both
                # take the same expected step (under the uniform policy the two are then identical).
                cfgs = [("frozen", "naive", 0.0), ("naive", "naive", eta_star), ("iw", "iw", eta_star * rate)]
                lab = torch.stack([torch.as_tensor(W[w], device=DEV) for _, w, _ in cfgs])
                taus = track2(pad, nk, lab, torch.full((len(sites),), tau0, device=DEV),
                              torch.tensor([e for _, _, e in cfgs], device=DEV), torch.zeros(len(cfgs), device=DEV))
                ev = site_eval(taus, sites, seen_t)
                rate_real = float(np.mean([L.mean() for L in Ls]))
                for g, (name, _, _) in enumerate(cfgs):
                    out[f"{mode}|{policy}|r={rate}|{name}"] = dict(summary(ev[g]), label_rate=rate_real, eta=eta_star)
            print(f"  seed {seed} r={rate} {mode} done", flush=True)
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = []
    for sd in range(n):
        allr.append(run(sd, tag))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{ROOT}/results/pers_select_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== selective labelling, DEV sites, features={tag}, RAPS, delay={DELAY}, {n} seeds")
    print(f"{'mode|policy|rate|tracker':<40}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'%win<.85':>9}{'size':>7}{'labels':>8}")
    for key in allr[0]:
        m = {k: np.mean([r[key][k] for r in allr]) for k in allr[0][key]}
        print(f"{key:<40}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%{m['win']:8.0f}%{m['size']:7.2f}{m['label_rate']:8.3f}")
