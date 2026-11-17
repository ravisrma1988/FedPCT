"""Tightened theory, checked on the real development streams (POST HOC, exploratory; BioCLIP-2 crops).
  1. Freedman (variance-adaptive) version of Theorem 1 for the frozen method (eta*).
  2. Certification floor per site: the bound's value as eta -> infinity,
       floor = ((D+1) n_max + mart) / (r N) + tail / N,
     i.e. the best guarantee ANY step size could give that site.
  3. "Certified" step size: per site, eta_k chosen from DEPLOYABLE quantities so that the Freedman
     bound would equal EPS if the test stream had N_hat = 2 x (the site's cal+tune known images)
     images (test is 40% of the period, cal+tune 20%); sites whose floor at N_hat exceeds EPS are
     tracked with the frozen eta*. Reported: realised bound (true N), coverage, %sig, set size,
     versus the frozen method on the same sites.
Usage: python theory_check2.py iwildcam|cct
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
from online2 import ETA_L, DELAY, build_streams, labels_for, pinball, site_eval, summary, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

EPS, DELTA_TOT = 0.10, 0.05
D = DELAY
le.TAG = "cct_bioclip2crop" if ds == "cct" else "bioclip2crop"
sel = json.load(open(f"{le.ROOT}/results/pers_lambda_{le.TAG}.json"))


def freedman(n, r, delta):
    if r == 1.0:
        return 0.0
    V = r * (1 - r) * (1 - ALPHA) ** 2 * float((n ** 2).sum())
    b = max(r, 1 - r) * float(n.max()) if len(n) else 0.0
    L = np.log(2 / delta)
    return np.sqrt(2 * V * L) + 2 / 3 * b * L


def bound(B, eta, n, r, delta):
    J, N = len(n), n.sum()
    head, tail = n[:max(J - D, 0)], n[max(J - D, 0):]
    return ((B + (D + 1) * eta * n.max()) / eta + freedman(head, r, delta)) / (r * N) + tail.sum() / N


def floor(n, r, delta):
    J, N = len(n), n.sum()
    head, tail = n[:max(J - D, 0)], n[max(J - D, 0):]
    return ((D + 1) * n.max() + freedman(head, r, delta)) / (r * N) + tail.sum() / N


out = []
for seed in le.SEEDS:
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
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
    keep = [k for k, st in enumerate(test_all) if st is not None and dev[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    delta = DELTA_TOT / len(base)
    held = np.isin(part, ["cal", "tune"]) & (y != cp.UNK)
    n_hat = np.array([2 * int((held & (client == k)).sum()) for k in keep])
    # sequence-length statistics of the held-out block (deployable proxy for the test stream)
    hs = [s[held & (client == k)].groupby("seq_id").size().to_numpy() for k in keep]
    nmax_hat = np.array([h.max() if len(h) else 1 for h in hs])
    nmean_hat = np.array([h.mean() if len(h) else 1.0 for h in hs])
    for rate in le.RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, "ditto", rate, sel[seed][f"r={rate}"]["lam_star"], seed, score_fn)
        nkn = nk.cpu().numpy()
        ns = [nkn[k, :st["seq_codes"].max() + 1] for k, st in enumerate(sites)]
        # certified eta from deployable quantities: n_max and the D-tail are taken from the site's
        # held-out block as a proxy (sequence lengths there), N from N_hat
        eta_c = np.full(len(sites), eta)
        cert = np.zeros(len(sites), bool)
        for k in range(len(sites)):
            nh = n_hat[k]
            proxy = np.full(max(int(nh / max(nmean_hat[k], 1)), 1), nmean_hat[k])
            fl = ((D + 1) * nmax_hat[k] + freedman(proxy, rate, delta)) / (rate * max(nh, 1))
            slack = EPS - fl
            if nh > 0 and slack > 0:
                eta_c[k] = max(B / (slack * rate * nh), eta)
                cert[k] = True
        scale = torch.as_tensor(eta_c / eta, dtype=torch.float32, device=DEV)[:, None]
        taus = torch.cat([track2(pad, nk, w[None], torch.full((len(sites),), tau0, device=DEV),
                                 torch.tensor([eta], device=DEV), torch.zeros(1, device=DEV))
                          for w in (W, W * scale)])
        ev = site_eval(taus, sites, seen_t)
        tc = taus.cpu().numpy()
        for k, (st, n) in enumerate(zip(sites, ns)):
            if n.sum() == 0:
                continue
            kn = ~np.isnan(st["score"])
            dev_frozen = abs(float((st["score"][kn] > tc[0, k, st["seq_codes"][kn]]).mean()) - ALPHA)
            dev_cert = abs(float((st["score"][kn] > tc[1, k, st["seq_codes"][kn]]).mean()) - ALPHA)
            out.append(dict(seed=seed, r=rate, N=int(n.sum()), nmax=int(n.max()), B=B, eta=eta, eta_c=float(eta_c[k]),
                            certified=bool(cert[k]), floor=float(floor(n, rate, delta)),
                            bound_frozen=float(bound(B, eta, n, rate, delta)),
                            bound_cert=float(bound(B, eta_c[k], n, rate, delta)), dev_frozen=dev_frozen, dev_cert=dev_cert,
                            frozen=ev[0][k], cert=ev[1][k]))
    print(f"seed {seed} done", flush=True)

json.dump(out, open(f"{le.ROOT}/results/theory_check2_{ds}.json", "w"), indent=1, default=float)
print(f"\n== tightened theory, {ds} development sites (POST HOC), EPS {EPS}, delta {DELTA_TOT}/K, delay {D}")
for rate in le.RATES:
    R = [o for o in out if o["r"] == rate]
    f = np.array([o["floor"] for o in R])
    bf = np.array([o["bound_frozen"] for o in R])
    print(f"r={rate}: {len(R)} site-streams | Freedman bound (frozen eta) median {np.median(bf):.2f} | "
          f"floor median {np.median(f):.3f}, share floor < 0.10: {np.mean(f < 0.10):.2f}, < 0.20: {np.mean(f < 0.20):.2f} | "
          f"violations frozen {sum(o['dev_frozen'] > o['bound_frozen'] for o in R)} cert {sum(o['dev_cert'] > o['bound_cert'] for o in R)}")
    C_ = [o for o in R if o["certified"]]
    if C_:
        bc = np.array([o["bound_cert"] for o in C_])
        m = lambda key, q: np.mean([o[key][q] for o in C_])  # noqa: E731
        print(f"   certified sites {len(C_)} ({len(C_) / len(R):.0%}): realised bound median {np.median(bc):.3f} "
              f"(< 0.10: {np.mean(bc < 0.10):.2f}) | eta_c median {np.median([o['eta_c'] for o in C_]):.3f} vs eta* "
              f"{np.median([o['eta'] for o in C_]):.3f} | cov {m('frozen', 'cov'):.3f} -> {m('cert', 'cov'):.3f} | "
              f"%sig {100 * m('frozen', 'sig'):.0f}% -> {100 * m('cert', 'sig'):.0f}% | size {m('frozen', 'size'):.2f} -> "
              f"{m('cert', 'size'):.2f}")
print("THEORY_CHECK2_DONE")
