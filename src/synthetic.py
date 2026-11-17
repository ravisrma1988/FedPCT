"""Synthetic federated open-set data: Gaussian classes, Dirichlet label shift across
clients, per-client covariate offset, and unknown classes present at some clients only.
Used to sanity-check the conformal code before real features exist."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from cp_baselines import UNK


def make(seed=0, K=20, C=10, U=5, d=16, beta=0.3, shift=0.0, unk_clients=0.5,
         unk_max=0.3, n_train=5000, n_cal=(80, 400), n_test=1000, sep=2.5):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, sep / np.sqrt(d) * 2, (C + U, d))
    offsets = rng.normal(0, shift, (K, d))
    props = rng.dirichlet(np.full(C, beta), K) if beta else np.full((K, C), 1 / C)
    has_unk = rng.random(K) < unk_clients
    pi = np.where(has_unk, rng.uniform(0.05, unk_max, K), 0.0)

    def sample(k, n):
        n_unk = rng.binomial(n, pi[k])
        yk = rng.choice(C, n - n_unk, p=props[k])
        yu = rng.choice(np.arange(C, C + U), n_unk)
        yy = np.concatenate([yk, yu])
        x = mu[yy] + offsets[k] + rng.normal(0, 1, (n, d))
        return x, np.where(yy >= C, UNK, yy)

    # Classifier trained on a pooled, balanced, known-class-only set (no client offsets).
    ytr = rng.integers(0, C, n_train)
    clf = LogisticRegression(max_iter=2000).fit(mu[ytr] + rng.normal(0, 1, (n_train, d)), ytr)

    out = {}
    for split, nfun in [("cal", lambda: rng.integers(*n_cal)), ("test", lambda: n_test)]:
        xs, ys, cs = [], [], []
        for k in range(K):
            x, yk = sample(k, nfun())
            xs.append(x); ys.append(yk); cs.append(np.full(len(yk), k))
        x, yy, cc = map(np.concatenate, (xs, ys, cs))
        out[split] = dict(probs=clf.predict_proba(x), y=yy, client=cc)
    return out, dict(K=K, C=C, pi=pi)
