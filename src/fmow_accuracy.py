"""Descriptive top-1 accuracy of the FedAvg classifier on FMoW development-site test images (5 seeds)."""
import numpy as np
import torch

import fmow_env

fmow_env.use_fmow()
from lockbox import dev_mask  # noqa: E402
from run_iwildcam import DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

acc = []
for sd in range(5):
    torch.manual_seed(sd)
    s, info = load_split("time", sd)
    X = torch.from_numpy(np.load("${FEDPCT_ROOT}/features/iwildcam_fmow_dinov2.npy")[s.row.to_numpy()]).float().to(DEV)
    y, cl, pt = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    tr = pt == "train"
    g = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), cl[tr], len(info["known_y"]), sd)
    P = probs_of(g, X)
    m = (pt == "test") & (y != -1) & dev_mask(info)[cl]
    acc.append(float((P[m].argmax(1) == y[m]).mean()))
    print(sd, acc[-1], flush=True)
print("FMOW_ACC per seed", np.round(acc, 3).tolist(), "mean", round(float(np.mean(acc)), 3))
