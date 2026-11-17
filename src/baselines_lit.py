"""Literature federated-conformal baselines on the same features / heads / scores / sites as the
method. Known-class part only (these methods have no unknown flag), DEVELOPMENT sites, global FedAvg
head (none of them adapts the model), RAPS scores, alpha = 0.1, static thresholds.

  fcp         pooled conformal quantile over all sites' cal scores (Lu et al., ICML 2023: FCP).
  labelshift  label-shift weighted quantile (Plassier et al., ICML 2023), as in cp_baselines.
  qq          one-shot quantile-of-quantiles (Humbert et al., ICML 2023, FedCP-QQ idea): each site
              sends its local beta-quantile, the server takes the k-th smallest. (beta, k) chosen
              by Monte-Carlo coverage under exchangeability with the ACTUAL per-site cal sizes:
              smallest expected threshold with coverage >= 1 - alpha. Our implementation of the
              idea, not the paper's closed-form rule.
  dr          density-ratio weighted conformal prediction for the target site (Tibshirani et al.
              2019), the shared core of PFCP (Min et al., NeurIPS 2025) and PFWCP (Vejling et al.
              2026). OUR ADAPTATION, not their algorithms: w(x) = p(site | x) / p(rest | x) * n_rest /
              n_site from a logistic site-vs-rest classifier on cal features (2-fold cross-fitting
              by sequence); per test image tau(x) = weighted (1 - alpha) quantile of the pooled cal
              scores with the test point's own weight on +inf. Needs the server to see features:
              an optimistic (non-private) version of the density-ratio approach.
Usage: python baselines_lit.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from lockbox import dev_mask
from online import stream_order
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
# beta fixed a priori at 1 - alpha (each site sends its local conformal-level quantile). Searching
# beta as well was unstable: many (beta, k) pairs have near-identical coverage under
# exchangeability, so Monte-Carlo noise picked k anywhere from 4 to 126 across seeds.
BETAS = (0.9,)
N_MC = 4000
DR_STEPS, DR_WD = 300, 1e-3  # fixed a priori


def qq_choose(sizes, alpha, rng):
    """(beta, k) with MC coverage >= 1 - alpha and smallest expected threshold (uniform scale)."""
    U = [rng.random((N_MC, m)) for m in sizes]
    best = None
    for b in BETAS:
        local = np.stack([np.quantile(u, b, axis=1, method="higher") for u in U], 1)   # (N_MC, K)
        local.sort(1)
        for k in range(len(sizes)):
            t = local[:, k]
            if t.mean() >= 1 - alpha and (best is None or t.mean() < best[2]):
                best = (b, k, t.mean())
    return best


def qq_threshold(S_cal, client_cal, K, beta, k):
    local = [np.quantile(S_cal[client_cal == c], beta, method="higher") for c in range(K) if (client_cal == c).any()]
    return np.sort(local)[k]


def site_classifier(Xc, is_site, folds):
    """Cross-fitted P(site | x) for the cal points; plus a full-data model for test points."""
    def fit(X, t):
        w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
        b = torch.zeros(1, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        pos = t.float().mean().clamp(1e-3, 1 - 1e-3)
        for _ in range(DR_STEPS):
            z = X @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, t.float(), pos_weight=(1 - pos) / pos) + DR_WD * (w ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return lambda Z: torch.sigmoid(Z @ w + b).detach()
    p_cal = torch.empty(len(Xc), device=DEV)
    for f in (0, 1):
        m = torch.as_tensor(folds == f, device=DEV)
        p_cal[m] = fit(Xc[~m], is_site[~m])(Xc[m])
    return p_cal, fit(Xc, is_site)


def weighted_tau(S_cal, w_cal, w_test, alpha):
    """Per test point: smallest s with sum_{s_i <= s} w_i >= (1 - alpha) (sum w + w_test)."""
    o = torch.argsort(S_cal)
    s_sorted, cw = S_cal[o], torch.cumsum(w_cal[o], 0)
    need = (1 - alpha) * (cw[-1] + w_test)
    i = torch.searchsorted(cw, need)
    return torch.where(i < len(s_sorted), s_sorted[i.clamp(max=len(s_sorted) - 1)], torch.tensor(float("inf"), device=DEV))


def evaluate(S, score, tau, seq, seen):
    kn = ~np.isnan(score)
    hit = score[kn] <= tau[kn]
    return dict(cov=float(hit.mean()), size=float((S[kn] <= tau[kn, None]).sum(1).mean()), seen=bool(seen),
                sig=bool(cp.cluster_upper_bound(hit, seq[kn]) < 1 - ALPHA))


def run(seed, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(40_000 + seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    S_all = scores.raps(P, lam, kreg, seed + 7)
    calk = cm & (y != cp.UNK)
    cal = dict(S=S_all[cm], y=y[cm], client=client[cm])
    tau_fcp = cp.thresholds("pooled", cal, ALPHA, C, K)
    tau_ls = cp.thresholds("labelshift", cal, ALPHA, C, K)
    Sc = cp.true_scores(S_all[calk], y[calk])
    sizes = np.bincount(client[calk], minlength=K)
    beta, k, _ = qq_choose(sizes[sizes > 0], ALPHA, rng)
    tau_qq = qq_threshold(Sc, client[calk], K, beta, k)
    Xc = X[torch.as_tensor(calk, device=DEV)]
    Sc_t = torch.as_tensor(Sc, dtype=torch.float32, device=DEV)
    folds = np.unique(seqs[calk], return_inverse=True)[1] % 2  # deterministic, whole sequences per fold
    out = {m: [] for m in ("fcp", "labelshift", "qq", "dr")}
    for c in range(K):
        idx = np.flatnonzero((part == "test") & (client == c))
        if not dev[c] or len(idx) == 0:
            continue
        idx = idx[stream_order(s.iloc[idx])]
        S, sc, sq = S_all[idx], cp.true_scores(S_all[idx], y[idx]), seqs[idx]
        n = len(idx)
        out["fcp"].append(evaluate(S, sc, np.full(n, tau_fcp[c]), sq, seen[c]))
        out["labelshift"].append(evaluate(S, sc, np.full(n, tau_ls[c]), sq, seen[c]))
        out["qq"].append(evaluate(S, sc, np.full(n, tau_qq), sq, seen[c]))
        is_site = torch.as_tensor(client[calk] == c, device=DEV)
        if is_site.sum() < 10:  # too few own cal points to fit a ratio: fall back to pooled
            out["dr"].append(evaluate(S, sc, np.full(n, tau_fcp[c]), sq, seen[c]))
            continue
        p_cal, model = site_classifier(Xc, is_site, folds)
        ratio = (1 - is_site.float().mean()) / is_site.float().mean()
        to_w = lambda p: (p.clamp(1e-4, 1 - 1e-4) / (1 - p.clamp(1e-4, 1 - 1e-4))) * ratio  # noqa: E731
        w_test = to_w(model(X[torch.as_tensor(idx, device=DEV)]))
        tau = weighted_tau(Sc_t, to_w(p_cal), w_test, ALPHA).cpu().numpy()
        out["dr"].append(evaluate(S, sc, tau, sq, seen[c]))
    summ = {}
    for m, rows in out.items():
        summ[m] = dict(cov=np.mean([r["cov"] for r in rows]), size=np.mean([r["size"] for r in rows]),
                       sig=100 * np.mean([r["sig"] for r in rows]),
                       sig_unseen=100 * np.mean([r["sig"] for r in rows if not r["seen"]]))
    summ["qq_choice"] = dict(beta=float(beta), k=int(k))
    print(f"seed {seed}: " + " | ".join(f"{m} %sig {summ[m]['sig']:.0f} size {summ[m]['size']:.2f}" for m in out)
          + f" | qq beta={beta} k={k}", flush=True)
    return summ


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = [run(sd, tag) for sd in range(n)]
    json.dump(allr, open(f"{ROOT}/results/baselines_lit_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== literature baselines (global head, static), DEV sites, features={tag}, RAPS, alpha={ALPHA}, {n} seeds")
    print(f"{'method':<12}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'size':>7}")
    for m in ("fcp", "labelshift", "qq", "dr"):
        a = {k: np.mean([r[m][k] for r in allr]) for k in allr[0][m]}
        print(f"{m:<12}{a['cov']:7.3f}{a['sig']:6.0f}%{a['sig_unseen']:11.0f}%{a['size']:7.2f}")
