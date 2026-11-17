"""Which novelty score separates unknown from known species? AUROC on tune (selection) and test
(report only), all sites, frozen features. Higher score = more novel.
  msp, energy               from the federated head
  proto                     1 - max cos to class means (federated aggregates)
  maha                      min_c Mahalanobis distance, tied covariance (class means + pooled
                            within-class covariance: federated aggregates; Lee et al. 2018)
  maha_rel                  maha minus Mahalanobis distance to the global mean (relative
                            Mahalanobis, Ren et al. 2021; global mean/cov also aggregates)
  knn_ref                   1 - max cos to any train feature (REFERENCE ONLY: needs raw features
                            at the server, not privacy-compliant)
Usage: python diag_novelty.py <feature tag> <n_seeds>
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import cp_baselines as cp
from openset import novelty_scores
from run_iwildcam import DEV, fedavg
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
tag, n = sys.argv[1], int(sys.argv[2])
for seed in range(n):
    s, info = load_split("time", seed)
    C = len(info["known_y"])
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    tr = part == "train"
    ytr = torch.from_numpy(y[tr]).to(DEV)
    glob = fedavg(X[tr], ytr, client[tr], C, seed)
    nov, _ = novelty_scores(glob, X, X[tr], ytr, C)
    with torch.no_grad():
        Xt = X[tr]
        cnt = torch.bincount(ytr, minlength=C).float().clamp(min=1)
        mu = torch.zeros(C, X.shape[1], device=DEV).index_add_(0, ytr, Xt) / cnt[:, None]
        R = Xt - mu[ytr]
        cov = R.T @ R / len(R) + 1e-4 * torch.eye(X.shape[1], device=DEV)
        Pm = torch.linalg.inv(cov)
        g_mu = Xt.mean(0)
        G = Xt - g_mu
        Pg = torch.linalg.inv(G.T @ G / len(G) + 1e-4 * torch.eye(X.shape[1], device=DEV))
        maha, maha0 = [], []
        for b in X.split(8192):
            d = b[:, None, :] - mu[None]                                   # (B, C, D)
            maha.append(torch.einsum("bcd,de,bce->bc", d, Pm, d).min(1).values)
            e = b - g_mu
            maha0.append(torch.einsum("bd,de,be->b", e, Pg, e))
        maha, maha0 = torch.cat(maha), torch.cat(maha0)
        nov["maha"] = maha.cpu().numpy()
        nov["maha_rel"] = (maha - maha0).cpu().numpy()
        Xn = torch.nn.functional.normalize(X, dim=1)
        Tn = torch.nn.functional.normalize(Xt, dim=1)
        nov["knn_ref"] = torch.cat([1 - (b @ Tn.T).max(1).values for b in Xn.split(4096)]).cpu().numpy()
    tm, te = part == "tune", part == "test"
    line = []
    for k, v in nov.items():
        line.append(f"{k}: tune {roc_auc_score(y[tm] == cp.UNK, v[tm]):.3f} test {roc_auc_score(y[te] == cp.UNK, v[te]):.3f}")
    print(f"seed {seed}: " + " | ".join(line), flush=True)
