"""Online evaluation v2 (GPU): two-level federated threshold tracking vs baselines vs the static
frontier, on validated time splits, RAPS scores, any feature file.

Per site k, sequence step j (time since calibration at that site):
  tau_kj = g_j + d_kj
  local  : d_k <- d_k + eta_l * lab_km * (miss_km - alpha * n_km)              (own delayed feedback)
  shared : g   <- g   + eta_g * sum_k lab_km (miss_km - alpha n_km) / max(1, sum_k lab_km n_km)
           (server; normalised pooled excess miss rate of all sites whose feedback for step m arrives)
  m = j - DELAY. eta_g = 0 gives per-site tracking (qt); eta_l = 0 gives one shared threshold.
Initial tau: pooled conformal quantile on cal (federation-informed start).
eta selection: replay each site's TUNE sequences (time order) as a stream, initial tau = pooled
quantile on tune, pick (eta_l, eta_g) per method and labelling rate minimising the mean pinball loss
of tau on labelled known-class feedback. Test data never influence the choice; the full grid is
also reported.
Usage: python online2.py <n_seeds> <feature tag, e.g. bioclip2 or bioclip2ft_seed{seed}>
"""
import itertools
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from online import DELAY, RATES, stream_order
from online_gpu import pack
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
ETA_L = (0.0, 0.005, 0.02, 0.05)
ETA_G = (0.0, 0.005, 0.02, 0.05)
FRONTIER = (0.1, 0.07, 0.05, 0.03, 0.02)
METHODS = {  # name -> allowed (eta_l, eta_g)
    "qt_site": [(l, 0.0) for l in ETA_L if l > 0],
    "shared_only": [(0.0, g) for g in ETA_G if g > 0],
    "two_level": [(l, g) for l, g in itertools.product(ETA_L, ETA_G) if l > 0 and g > 0],
}


def track2(pad, nk, lab, tau0, eta_l, eta_g, alpha=ALPHA):
    """pad (K,J,I), nk (K,J), lab (G,K,J) float, tau0 (K,), eta_l/eta_g (G,). Returns tau (G,K,J).
    alpha: target miss rate of the tracked quantity (default: the known-class ALPHA)."""
    G, K, J = lab.shape
    taus = torch.empty(G, K, J, device=DEV)
    d = torch.zeros(G, K, device=DEV)
    g = torch.zeros(G, 1, device=DEV)
    for j in range(J):
        taus[:, :, j] = tau0[None, :] + g + d
        m = j - DELAY
        if m >= 0:
            miss = (pad[None, :, m, :] > taus[:, :, m, None]).sum(-1).float()
            exc = lab[:, :, m] * (miss - alpha * nk[None, :, m])           # (G,K)
            d = d + eta_l[:, None] * exc
            denom = (lab[:, :, m] * nk[None, :, m]).sum(1, keepdim=True).clamp(min=1)
            g = g + eta_g[:, None] * exc.sum(1, keepdim=True) / denom
    return taus


def build_streams(s, part_name, S_all, y, seqs):
    """Per-site streams (time order) for one partition. Returns list of dicts (one per site)."""
    sites = []
    for k in range(s.client.max() + 1):
        idx = np.flatnonzero((s.part.to_numpy() == part_name) & (s.client.to_numpy() == k))
        if len(idx) == 0:
            sites.append(None)
            continue
        idx = idx[stream_order(s.iloc[idx])]
        sq = seqs[idx]
        sites.append(dict(idx=idx, S=S_all[idx], score=cp.true_scores(S_all[idx], y[idx]),
                          seq=sq, seq_codes=np.cumsum(np.r_[0, sq[1:] != sq[:-1]])))
    return sites


def labels_for(sites, rate, rng_seed, J):
    rng = np.random.default_rng(rng_seed)
    lab = np.zeros((len(sites), J), dtype=np.float32)
    for k, st in enumerate(sites):
        u = rng.random(st["seq_codes"].max() + 1)  # same draws for every method/eta
        lab[k, :len(u)] = u < rate
    return torch.as_tensor(lab, device=DEV)


def pinball(taus, sites):
    """Mean pinball loss (level 1-alpha) of the threshold on known-class images, all sites."""
    tot, n = 0.0, 0
    for k, st in enumerate(sites):
        kn = ~np.isnan(st["score"])
        t = taus[:, k, torch.as_tensor(st["seq_codes"][kn], device=DEV)]
        sc = torch.as_tensor(st["score"][kn], dtype=torch.float32, device=DEV)[None, :]
        diff = sc - t
        tot = tot + torch.where(diff >= 0, (1 - ALPHA) * diff, -ALPHA * diff).sum(1)
        n += int(kn.sum())
    return (tot / max(n, 1)).cpu().numpy()


def site_eval(taus, sites, seen):
    """Per (config, site): coverage, size, sig_below (cluster bootstrap), share of 200-windows < 0.85."""
    G = taus.shape[0]
    res = [[None] * len(sites) for _ in range(G)]
    for k, st in enumerate(sites):
        kn = ~np.isnan(st["score"])
        t = taus[:, k, torch.as_tensor(st["seq_codes"], device=DEV)]          # (G, n)
        Ssorted = torch.sort(torch.as_tensor(st["S"], dtype=torch.float32, device=DEV), 1).values
        size = torch.stack([torch.searchsorted(Ssorted, t[g][:, None].contiguous(), right=True)[:, 0]
                            for g in range(G)]).float()[:, torch.as_tensor(kn, device=DEV)].cpu().numpy()
        hit = (torch.as_tensor(st["score"][kn], dtype=torch.float32, device=DEV)[None, :]
               <= t[:, torch.as_tensor(kn, device=DEV)]).cpu().numpy()
        for g in range(G):
            h = hit[g]
            w = [h[i:i + 200].mean() for i in range(0, len(h) - 199, 200)]
            res[g][k] = dict(cov=float(h.mean()), size=float(size[g].mean()), seen=bool(seen[k]),
                             sig=bool(cp.cluster_upper_bound(h, st["seq"][kn]) < 1 - ALPHA),
                             win=float(np.mean(np.array(w) < 0.85)) if w else np.nan)
    return res


def summary(rows):
    v = [r for r in rows if r is not None]
    return dict(cov=np.mean([r["cov"] for r in v]), sig=100 * np.mean([r["sig"] for r in v]),
                sig_unseen=100 * np.mean([r["sig"] for r in v if not r["seen"]]),
                win=100 * np.nanmean([r["win"] for r in v]), size=np.mean([r["size"] for r in v]))


def run(seed, tag):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    Xall = np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    head = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P = probs_of(head, X)
    tm = part == "tune"
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    S_all = scores.raps(P, lam, kreg, seed + 7)

    calm = part == "cal"
    cal = dict(S=S_all[calm], y=y[calm], client=client[calm])
    tau_pooled = cp.thresholds("pooled", cal, ALPHA, C, K)[0]
    tune_all = dict(S=S_all[tm], y=y[tm], client=client[tm])
    tau_tune = cp.thresholds("pooled", tune_all, ALPHA, C, K)[0]

    out = {"raps": [lam, kreg], "acc": float((P[part == "test"].argmax(1) == y[part == "test"])[
        y[part == "test"] != cp.UNK].mean())}
    # static frontier (pooled) on test
    test_sites = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_sites) if st is not None]
    test_sites = [test_sites[k] for k in keep]
    seen_t = seen[keep]
    pad, nk = pack(test_sites)
    J = pad.shape[1]
    fr = torch.tensor([cp.thresholds("pooled", cal, a, C, K)[0] for a in FRONTIER], device=DEV)
    fr_taus = fr[:, None, None].expand(len(FRONTIER), len(test_sites), J).contiguous()
    out["static"] = {a: summary(r) for a, r in zip(FRONTIER, site_eval(fr_taus, test_sites, seen_t))}

    # eta selection on tune streams, then test
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        lab_s = labels_for(test_sites, rate, 10_000 + seed, J)
        for meth, grid in METHODS.items():
            el = torch.tensor([a for a, _ in grid], device=DEV)
            eg = torch.tensor([b for _, b in grid], device=DEV)
            G = len(grid)
            tt = track2(tpad, tnk, lab_t[None].expand(G, -1, -1),
                        torch.full((len(tune_sites),), tau_tune, device=DEV), el, eg)
            choice = int(np.argmin(pinball(tt, tune_sites)))
            ts = track2(pad, nk, lab_s[None].expand(G, -1, -1),
                        torch.full((len(test_sites),), tau_pooled, device=DEV), el, eg)
            ev = site_eval(ts, test_sites, seen_t)
            out[f"{meth}|r={rate}"] = dict(chosen=grid[choice], selected=summary(ev[choice]),
                                           grid={str(grid[g]): summary(ev[g]) for g in range(G)})
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = []
    for sd in range(n):
        allr.append(run(sd, tag))
        print(f"seed {sd} done", flush=True)
    safe = tag.replace("{seed}", "S")
    json.dump(allr, open(f"{ROOT}/results/online2_{safe}.json", "w"), indent=1, default=float)
    print(f"\n== online v2, features={tag}, RAPS, time split, alpha={ALPHA}, delay={DELAY}, {n} seeds; "
          f"acc={np.mean([r['acc'] for r in allr]):.3f}")
    print(f"{'method':<34}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'%win<.85':>9}{'size':>7}  chosen(eta_l,eta_g)")
    for a in FRONTIER:
        m = {k: np.mean([r["static"][a][k] for r in allr]) for k in allr[0]["static"][a]}
        print(f"{'static_pooled a=' + str(a):<34}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%"
              f"{m['win']:8.0f}%{m['size']:7.2f}")
    for key in [k for k in allr[0] if "|r=" in k]:
        m = {k: np.mean([r[key]["selected"][k] for r in allr]) for k in allr[0][key]["selected"]}
        ch = [tuple(r[key]["chosen"]) for r in allr]
        print(f"{key:<34}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%{m['win']:8.0f}%"
              f"{m['size']:7.2f}  {max(set(ch), key=ch.count)} ({ch.count(max(set(ch), key=ch.count))}/{len(ch)})")
