"""Non-conformity score matrices S (n, C): S[i, c] = score of candidate label c for image i.

lac  : 1 - p_c
raps : Angelopoulos et al. (ICLR 2021) regularised adaptive prediction sets:
       (mass of labels ranked above c) + u_i * p_c + lam * max(0, rank_c + 1 - kreg),
       u_i ~ U(0, 1) fixed per image by `seed` (randomised version).
RAPS (lam, kreg) are hyper-parameters: choose them with tune_raps() on the TUNE partition only.
"""
import numpy as np

import cp_baselines as cp

RAPS_GRID = [(lam, k) for lam in (0.0, 0.001, 0.01, 0.05, 0.2) for k in (1, 2, 5, 10)]


def lac(P):
    return 1.0 - P


def raps(P, lam, kreg, seed=0):
    n, C = P.shape
    order = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, order, 1)
    above = np.cumsum(Ps, 1) - Ps
    u = np.random.default_rng(seed).random((n, 1))
    pen = lam * np.maximum(0, np.arange(1, C + 1) - kreg)[None, :]
    S = np.empty_like(P)
    np.put_along_axis(S, order, above + u * Ps + pen, 1)
    return S


def tune_raps(P, y, seq, alpha, seed=0):
    """Pick (lam, kreg) minimising mean set size of a POOLED threshold: calibrate on a random half
    of the tune sequences, measure size on the other half. Uses tune data only."""
    rng = np.random.default_rng(seed)
    useq = np.unique(seq)
    half = np.isin(seq, rng.choice(useq, len(useq) // 2, replace=False))
    kn = y != cp.UNK
    best = None
    for lam, k in RAPS_GRID:
        S = raps(P, lam, k, seed)
        tau = cp.conformal_quantile(cp.true_scores(S, y)[half & kn], alpha)
        size = (S[~half & kn] <= tau).sum(1).mean()
        if best is None or size < best[0]:
            best = (size, lam, k)
    return best[1], best[2]
