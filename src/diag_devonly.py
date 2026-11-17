"""Re-check the backbone decision on DEVELOPMENT-site test images only. compare_backbones.py and the
novelty diagnostics reported test metrics over all sites, so the crop choice saw lockbox test
images; this repeats the deciding comparison with lockbox sites excluded.
Per tag: FedAvg known-class accuracy, pooled LAC (alpha 0.1, cal = tune+cal) mean set size and
coverage, novelty AUROC (MSP) - all on dev-site test images.
Usage: python diag_devonly.py <n_seeds> <tag> [<tag> ...]
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import cp_baselines as cp
import scores
from lockbox import dev_mask
from run_iwildcam import ALPHA, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
n, tags = int(sys.argv[1]), sys.argv[2:]
res = {t: [] for t in tags}
for seed in range(n):
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    devs = dev_mask(info)[client]
    calm, tem = np.isin(part, ["tune", "cal"]), (part == "test") & devs
    tr = part == "train"
    for t in tags:
        torch.manual_seed(seed)
        X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{t.format(seed=seed)}.npy")[s.row.to_numpy()]).float().cuda()
        head = fedavg(X[tr], torch.from_numpy(y[tr]).cuda(), client[tr], C, seed)
        Pc, Pt = probs_of(head, X[calm]), probs_of(head, X[tem])
        tau = cp.thresholds("pooled", dict(S=scores.lac(Pc), y=y[calm], client=client[calm]), ALPHA, C, K)[0]
        St, yt = scores.lac(Pt), y[tem]
        kn = yt != cp.UNK
        res[t].append(dict(acc=(Pt.argmax(1)[kn] == yt[kn]).mean(), size=(St[kn] <= tau).sum(1).mean(),
                           cov=(cp.true_scores(St, yt)[kn] <= tau).mean(),
                           auroc=roc_auc_score(~kn, 1 - Pt.max(1))))
    print(f"seed {seed}: " + " | ".join(f"{t} acc {res[t][-1]['acc']:.3f} size {res[t][-1]['size']:.2f} "
                                       f"auroc {res[t][-1]['auroc']:.3f}" for t in tags), flush=True)
print(f"\n== DEV-site test only, FedAvg head, pooled LAC alpha={ALPHA}, {n} seeds")
for t in tags:
    m = {k: np.mean([r[k] for r in res[t]]) for k in res[t][0]}
    print(f"{t:<24} acc {m['acc']:.3f}  cov {m['cov']:.3f}  size {m['size']:.2f}  msp auroc {m['auroc']:.3f}")
