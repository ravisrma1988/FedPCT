"""Baseline conformal calibrators for the federated open-set setting.

Conventions (shared by all datasets):
  probs   (n, C)  softmax over the C known classes
  S       (n, C)  non-conformity score of every candidate label (LAC: 1 - probs; see scores.py)
  y       (n,)    true class in 0..C-1, or UNK (-1) for species outside the known set
  client  (n,)    client id 0..K-1
A prediction set is a boolean array (n, C+1); column C is the "unknown" label.
"""
import numpy as np

UNK = -1


def true_scores(S, y):
    """Non-conformity of the true label (nan for UNK rows)."""
    s = np.full(len(y), np.nan)
    k = y != UNK
    s[k] = S[k, y[k]]
    return s


def conformal_quantile(scores, alpha, weights=None):
    """Split-conformal threshold; with weights, the Tibshirani et al. (2019) weighted
    quantile with the test point's mass (weight 1) placed at +inf."""
    scores = np.asarray(scores)
    if len(scores) == 0:
        return np.inf
    if weights is None:
        n = len(scores)
        r = int(np.ceil((n + 1) * (1 - alpha)))
        return np.inf if r > n else np.sort(scores)[r - 1]
    order = np.argsort(scores)
    s, w = scores[order], np.asarray(weights)[order]
    cdf = np.cumsum(w) / (w.sum() + 1.0)
    idx = np.searchsorted(cdf, 1 - alpha)
    return np.inf if idx >= len(s) else s[idx]


def class_props(y, C, smooth=1.0):
    known = y[y != UNK]
    c = np.bincount(known, minlength=C) + smooth
    return c / c.sum()


def thresholds(method, cal, alpha, C, K):
    """Return a per-client score threshold tau[k] for the known-class part of the set."""
    s = true_scores(cal["S"] if "S" in cal else 1.0 - cal["probs"], cal["y"])
    known = cal["y"] != UNK
    tau = np.empty(K)
    if method == "local":
        for k in range(K):
            m = known & (cal["client"] == k)
            tau[k] = conformal_quantile(s[m], alpha)
    elif method == "pooled":
        tau[:] = conformal_quantile(s[known], alpha)
    elif method == "labelshift":
        # Importance weights p_k(y)/p_j(y) for a calibration point of client j
        # (label-shift correction in the spirit of Plassier et al., ICML 2023; no DP).
        P = np.stack([class_props(cal["y"][cal["client"] == j], C) for j in range(K)])
        yk, cj = cal["y"][known], cal["client"][known]
        for k in range(K):
            w = P[k, yk] / P[cj, yk]
            tau[k] = conformal_quantile(s[known], alpha, weights=w)
    else:
        raise ValueError(method)
    return tau


def predict_sets(S, client, tau, novelty=None, nov_tau=None):
    """Known label y is in the set iff S[:, y] <= tau[client]; UNK is added iff
    novelty > nov_tau (closed-set baselines pass novelty=None and never add UNK)."""
    n, C = S.shape
    sets = np.zeros((n, C + 1), dtype=bool)
    sets[:, :C] = S <= tau[client][:, None]
    if novelty is not None:
        sets[:, C] = novelty > nov_tau
    return sets


def evaluate(sets, y, client, K):
    C = sets.shape[1] - 1
    target = np.where(y == UNK, C, y)
    hit = sets[np.arange(len(y)), target]
    size = sets.sum(1)
    rows = []
    for k in range(K):
        m = client == k
        if m.sum() == 0:
            continue
        u = m & (y == UNK)
        rows.append(dict(
            client=k, n=int(m.sum()), unk_frac=float(u.sum() / m.sum()),
            coverage=float(hit[m].mean()), cov_known=float(hit[m & ~u].mean()),
            set_size=float(size[m].mean()),
            unk_recall=float(hit[u].mean()) if u.any() else np.nan,
        ))
    return rows


def cluster_upper_bound(hit, groups, level=0.95, B=2000, seed=0):
    """One-sided upper confidence bound on coverage by a cluster (sequence) bootstrap:
    whole sequences are resampled, so near-duplicate frames do not count as independent."""
    _, g = np.unique(groups, return_inverse=True)
    hits = np.bincount(g, weights=hit.astype(float))
    cnts = np.bincount(g).astype(float)
    if len(cnts) < 2:
        return 1.0
    W = np.random.default_rng(seed).multinomial(len(cnts), np.full(len(cnts), 1 / len(cnts)), B)
    return float(np.quantile((W @ hits) / (W @ cnts), level))


def sig_below_clustered(sets, y, client, groups, K, alpha):
    """Per client: True if the known-class coverage is significantly below 1 - alpha
    (cluster-bootstrap 95% upper bound < 1 - alpha)."""
    C = sets.shape[1] - 1
    kn = y != UNK
    hit = sets[np.arange(len(y)), np.where(kn, y, C)]
    out = {}
    for k in range(K):
        m = (client == k) & kn
        if m.sum():
            out[k] = cluster_upper_bound(hit[m], groups[m], seed=k) < 1 - alpha
    return out


def summarize(rows, alpha):
    cov = np.array([r["coverage"] for r in rows])
    return dict(
        mean_cov=cov.mean(), worst_cov=cov.min(),
        worst_cov_known=min(r["cov_known"] for r in rows),
        frac_clients_below=float((cov < 1 - alpha - 0.02).mean()),
        mean_size=np.mean([r["set_size"] for r in rows]),
    )
