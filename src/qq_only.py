"""Re-run only the quantile-of-quantiles baseline of baselines_lit.py (fixed beta = 1 - alpha); the
other baselines are unchanged and their results stay in results/baselines_lit_*.json.
Usage: python qq_only.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from baselines_lit import ROOT, evaluate, qq_choose, qq_threshold
from lockbox import dev_mask
from online import stream_order
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

n, tag = int(sys.argv[1]), sys.argv[2]
allr = []
for seed in range(n):
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
    Sc = cp.true_scores(S_all[calk], y[calk])
    sizes = np.bincount(client[calk], minlength=K)
    beta, k, mc = qq_choose(sizes[sizes > 0], ALPHA, rng)
    tau = qq_threshold(Sc, client[calk], K, beta, k)
    rows = []
    for c in range(K):
        idx = np.flatnonzero((part == "test") & (client == c))
        if not dev[c] or len(idx) == 0:
            continue
        idx = idx[stream_order(s.iloc[idx])]
        rows.append(evaluate(S_all[idx], cp.true_scores(S_all[idx], y[idx]), np.full(len(idx), tau), seqs[idx], seen[c]))
    r = dict(cov=np.mean([x["cov"] for x in rows]), size=np.mean([x["size"] for x in rows]),
             sig=100 * np.mean([x["sig"] for x in rows]), sig_unseen=100 * np.mean([x["sig"] for x in rows if not x["seen"]]),
             beta=float(beta), k=int(k), n_sites_cal=int((sizes > 0).sum()), mc_cov=float(mc))
    allr.append(r)
    print(f"seed {seed}: qq beta={beta} k={k}/{r['n_sites_cal']} (MC cov {mc:.3f}) %sig {r['sig']:.0f} size {r['size']:.2f}", flush=True)
json.dump(allr, open(f"{ROOT}/results/baselines_qq_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
m = {k: np.mean([r[k] for r in allr]) for k in ("cov", "sig", "sig_unseen", "size")}
print(f"qq (beta=1-alpha)  cov {m['cov']:.3f}  %sig {m['sig']:.0f}%  %sig_unseen {m['sig_unseen']:.0f}%  size {m['size']:.2f}")
