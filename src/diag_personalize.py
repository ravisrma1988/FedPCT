"""Hypothesis: the residual drift is within-class (conditional) shift, which only the model can fix;
personalising the federated head with a site's delayed labels should raise accuracy on its later
data. Cheap upper-bound check on DEVELOPMENT sites: each site's test stream is cut in time into
H1 (labels available, used to adapt) and H2 (evaluation, never used for fitting).

  global      federated head, unchanged
  prior_fix   global logits + log(pi_H1 / pi_train)     (label-shift correction from H1 labels)
  ditto       head fine-tuned on H1 with L2 pull to the global weights (Li et al., ICML 2021);
              steps/lr/lambda fixed a priori
Reports known-class accuracy on H2 (mean over sites, and pooled), for sites with >= 30 known
images in each half.
Usage: python diag_personalize.py <feature tag> <n_seeds>
"""
import sys

import numpy as np
import torch

import cp_baselines as cp
from lockbox import dev_mask
from online import stream_order
from run_iwildcam import DEV, fedavg
from splits import load_split

ROOT, MIN = "${FEDPCT_ROOT}", 30
STEPS, LR, LAMBDA = 100, 0.1, 1.0


def ditto(glob, X, y):
    h = torch.nn.Linear(glob.in_features, glob.out_features).to(DEV)
    h.load_state_dict(glob.state_dict())
    w0 = [p.detach().clone() for p in glob.parameters()]
    opt = torch.optim.SGD(h.parameters(), lr=LR)
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(h(X * 10), y)
        loss = loss + LAMBDA / 2 * sum(((p - q) ** 2).sum() for p, q in zip(h.parameters(), w0))
        opt.zero_grad(); loss.backward(); opt.step()
    return h


tag, n = sys.argv[1], int(sys.argv[2])
for seed in range(n):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    pi_tr = torch.as_tensor((np.bincount(y[tr], minlength=C) + 1.0) / (tr.sum() + C), device=DEV)
    acc = {"global": [], "prior_fix": [], "ditto": []}
    pooled = {k: [0, 0] for k in acc}
    for k in np.flatnonzero(dev):
        idx = np.flatnonzero((part == "test") & (client == k))
        if len(idx) == 0:
            continue
        idx = idx[stream_order(s.iloc[idx])]
        half = len(idx) // 2
        h1, h2 = idx[:half], idx[half:]
        h1, h2 = h1[y[h1] != cp.UNK], h2[y[h2] != cp.UNK]
        if len(h1) < MIN or len(h2) < MIN:
            continue
        X1, y1 = X[h1], torch.from_numpy(y[h1]).to(DEV)
        X2, y2 = X[h2], torch.from_numpy(y[h2]).to(DEV)
        with torch.no_grad():
            lg = glob(X2 * 10)
            pi1 = (torch.bincount(y1, minlength=C).float() + 1) / (len(y1) + C)
            preds = {"global": lg.argmax(1), "prior_fix": (lg + torch.log(pi1 / pi_tr)).argmax(1)}
        preds["ditto"] = ditto(glob, X1, y1)(X2 * 10).argmax(1)
        for name, p in preds.items():
            c = int((p == y2).sum())
            acc[name].append(c / len(y2))
            pooled[name][0] += c
            pooled[name][1] += len(y2)
    print(f"seed {seed}: sites={len(acc['global'])}  " + "  ".join(
        f"{n_}: site-mean {np.mean(v):.3f} pooled {pooled[n_][0] / pooled[n_][1]:.3f}" for n_, v in acc.items()),
        flush=True)
