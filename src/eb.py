"""Federated empirical-Bayes site thresholds.

Each site sends two numbers computed on its own calibration data: its local conformal quantile q_k
and the sampling variance v_k of that quantile (bootstrap over whole sequences). The server returns
the pooled quantile tau_pool and the between-site variance
    s2_b = max(0, mean_k (q_k - tau_pool)^2 - mean_k v_k)      (method of moments)
and each site uses tau_k = tau_pool + w_k (q_k - tau_pool),  w_k = s2_b / (s2_b + v_k).
Sites with too little data (q_k infinite) get w_k = 0. Communication: 2 scalars per site.
The EB threshold is an initial value; finite-sample validity is not claimed for it (the online
tracker provides the per-site long-run guarantee).
"""
import numpy as np

import cp_baselines as cp

B = 300


def site_stats(score, seq, alpha, rng):
    """score: known-class true-label scores of one site's cal data; seq: their sequence ids."""
    q = cp.conformal_quantile(score, alpha)
    if not np.isfinite(q):
        return q, np.inf
    _, g = np.unique(seq, return_inverse=True)
    G = g.max() + 1
    members = [np.flatnonzero(g == i) for i in range(G)]
    boots = []
    for _ in range(B):
        pick = rng.integers(0, G, G)
        boots.append(cp.conformal_quantile(score[np.concatenate([members[i] for i in pick])], alpha))
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    return q, (np.var(boots) if len(boots) > 10 else np.inf)


def persistence_thresholds(cal, alpha, K, min_n=10):
    """Federated persistence-shrunk site thresholds (no test data).
    cal: dict with S, y, client, time (sequence start, sortable).
    1. tau_pool = pooled conformal quantile; e_k = site's cal miss rate at tau_pool minus alpha.
    2. Each site also sends its excess miss rate on the early and the late half (by time) of its cal
       data; the server regresses late on early across sites (weighted) -> rho (clipped to [0, 1]).
    3. Site threshold = its local cal quantile at level 1 - (alpha + (1 - rho) e_k):
       rho = 0 -> pooled behaviour, rho = 1 -> fully local. Communication: 3 scalars per site."""
    kn = cal["y"] != cp.UNK
    s = cp.true_scores(cal["S"], cal["y"])
    tau_pool = cp.conformal_quantile(s[kn], alpha)
    e, early, late, w = np.full(K, np.nan), np.full(K, np.nan), np.full(K, np.nan), np.zeros(K)
    for k in range(K):
        m = np.flatnonzero(kn & (cal["client"] == k))
        if len(m) < min_n:
            continue
        miss = s[m] > tau_pool
        e[k] = miss.mean() - alpha
        order = np.argsort(cal["time"][m], kind="stable")
        h = len(m) // 2
        early[k], late[k] = miss[order[:h]].mean() - alpha, miss[order[h:]].mean() - alpha
        w[k] = h
    ok = np.isfinite(early)
    ww = w[ok] / w[ok].sum()
    ea, la = early[ok], late[ok]
    me, ml = np.sum(ww * ea), np.sum(ww * la)
    rho = float(np.clip(np.sum(ww * (ea - me) * (la - ml)) / max(np.sum(ww * (ea - me) ** 2), 1e-12), 0, 1))
    tau = np.full(K, tau_pool)
    for k in np.flatnonzero(np.isfinite(e)):
        m = kn & (cal["client"] == k)
        level = float(np.clip(alpha + (1 - rho) * e[k], 1e-3, 0.5))
        tau[k] = min(cp.conformal_quantile(s[m], level), float(np.nanmax(cal["S"])))
    return tau, dict(tau_pool=float(tau_pool), rho=rho, n_ok=int(ok.sum()))


def eb_thresholds(cal, alpha, K, seed=0):
    """cal: dict with S, y, client, seq. Returns (tau (K,), info dict)."""
    rng = np.random.default_rng(seed)
    kn = cal["y"] != cp.UNK
    s = cp.true_scores(cal["S"], cal["y"])
    tau_pool = cp.conformal_quantile(s[kn], alpha)
    q, v = np.full(K, np.inf), np.full(K, np.inf)
    for k in range(K):
        m = kn & (cal["client"] == k)
        if m.sum() >= 5:
            q[k], v[k] = site_stats(s[m], cal["seq"][m], alpha, rng)
    ok = np.isfinite(q) & np.isfinite(v)
    s2_b = max(0.0, float(np.mean((q[ok] - tau_pool) ** 2) - np.mean(v[ok]))) if ok.any() else 0.0
    w = np.where(ok, s2_b / (s2_b + np.where(ok, v, 1.0) + 1e-12), 0.0)
    tau = tau_pool + w * np.where(ok, q - tau_pool, 0.0)
    return tau, dict(tau_pool=float(tau_pool), s2_b=s2_b, mean_w=float(w[ok].mean()) if ok.any() else 0.0,
                     n_ok=int(ok.sum()))
