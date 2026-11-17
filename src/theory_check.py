"""Numerical check of THEORY.md on the real development streams (frozen method, BioCLIP-2 crops).
Per site, seed and rate r, with the actual labelling draws, eta*, delay D and RAPS bound B:
  * Lemma 1: every threshold lies in [-(D+1) eta alpha n_max, B + (D+1) eta (1-alpha) n_max]  (must hold)
  * Theorem 1: |miss rate - alpha| <= bound (delta = 0.05 / K, union over sites); at r = 1 deterministic
  * how informative the bound is: median bound, share of sites with bound < 0.05 / < 0.10
Usage: python theory_check.py iwildcam|cct
"""
import json
import sys

import numpy as np
import torch

ds = sys.argv[1]
if ds == "cct":
    import cct_env
    cct_env.use_cct()
import cp_baselines as cp  # noqa: E402
import lockbox_eval as le  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online2 import ETA_L, DELAY, build_streams, labels_for, pinball, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

le.TAG = "cct_bioclip2crop" if ds == "cct" else "bioclip2crop"
sel = json.load(open(f"{le.ROOT}/results/pers_lambda_{le.TAG}.json"))
out = []
for seed in le.SEEDS:
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    X = torch.from_numpy(np.load(f"{le.ROOT}/features/iwildcam_{le.TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    B = 1 + lam_r * max(0, C - kreg)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P)
    tau0 = cp.thresholds("pooled", dict(S=S_all[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    dev = dev_mask(info)
    base = [st for k, st in enumerate(test_all) if st is not None and dev[k]]
    delta = 0.05 / len(base)
    for rate in le.RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, sel[seed][f"r={rate}"]["lam_star"], seed, score_fn)
        taus = track2(pad, nk, W[None], torch.full((len(sites),), tau0, device=DEV), torch.tensor([eta], device=DEV),
                      torch.zeros(1, device=DEV))[0].cpu().numpy()
        nkn, Wn = nk.cpu().numpy(), W.cpu().numpy()
        for k, st in enumerate(sites):
            J = st["seq_codes"].max() + 1
            n = nkn[k, :J]
            N, nmax = n.sum(), n.max()
            if N == 0:
                continue
            kn = ~np.isnan(st["score"])
            miss_rate = float((st["score"][kn] > taus[k, st["seq_codes"][kn]]).mean())
            lo, hi = -(DELAY + 1) * eta * ALPHA * nmax, B + (DELAY + 1) * eta * (1 - ALPHA) * nmax
            lemma_ok = bool((taus[k, :J] >= lo - 1e-6).all() and (taus[k, :J] <= hi + 1e-6).all())
            Wd = B + (DELAY + 1) * eta * nmax
            mart = 0.0 if rate == 1.0 else np.sqrt(2 * np.log(2 / delta) * (n[:max(J - DELAY, 0)] ** 2).sum())
            bound = (Wd / eta + mart) / (rate * N) + n[max(J - DELAY, 0):].sum() / N
            out.append(dict(seed=seed, r=rate, site=int(k), N=int(N), T=int(J), eta=eta, B=B,
                            dev=abs(miss_rate - ALPHA), bound=float(bound), lemma_ok=lemma_ok,
                            labelled_share=float(Wn[k, :J].mean())))
    print(f"seed {seed} done (B={B:.2f}, lam_RAPS={lam_r}, kreg={kreg})", flush=True)

json.dump(out, open(f"{le.ROOT}/results/theory_check_{ds}.json", "w"), indent=1, default=float)
print(f"\n== theory check, {ds} development sites, seeds {list(le.SEEDS)}, delay {DELAY}, delta 0.05/K")
for rate in le.RATES:
    rows = [o for o in out if o["r"] == rate]
    dv, bd = np.array([o["dev"] for o in rows]), np.array([o["bound"] for o in rows])
    print(f"r={rate}: site-streams {len(rows)} | Lemma 1 violations {sum(not o['lemma_ok'] for o in rows)} | "
          f"Theorem 1 violations {(dv > bd).sum()} | bound median {np.median(bd):.3f} "
          f"(share < 0.05: {np.mean(bd < 0.05):.2f}, < 0.10: {np.mean(bd < 0.10):.2f}) | "
          f"actual |miss-alpha| median {np.median(dv):.3f} | median N {np.median([o['N'] for o in rows]):.0f}")
print("THEORY_CHECK_DONE")
