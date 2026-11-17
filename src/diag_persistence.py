"""Do site offsets persist from calibration into deployment? At the pooled RAPS threshold, compare
each site's excess miss rate on cal with its excess miss rate in successive test periods (fifths of
its own test stream). Reports the weighted correlation across sites (sites with >= MIN known images
in both periods) and the slope of test excess on cal excess (1 = fully persistent offset, 0 = none).
Also correlation between consecutive test fifths (persistence within deployment).
Usage: python diag_persistence.py <feature tag> <n_seeds>
"""
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT, MIN = "${FEDPCT_ROOT}", 30


def wcorr_slope(a, b, w):
    w = w / w.sum()
    ma, mb = np.sum(w * a), np.sum(w * b)
    cov = np.sum(w * (a - ma) * (b - mb))
    va, vb = np.sum(w * (a - ma) ** 2), np.sum(w * (b - mb) ** 2)
    return cov / np.sqrt(va * vb), cov / va


tag, n = sys.argv[1], int(sys.argv[2])
for seed in range(n):
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    head = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P = probs_of(head, X)
    tm = part == "tune"
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    S = scores.raps(P, lam, kreg, seed + 7)
    cm = part == "cal"
    tau = cp.thresholds("pooled", dict(S=S[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    miss = cp.true_scores(S, y) > tau
    kn = y != cp.UNK
    d = s.assign(miss=miss, kn=kn)
    start = d.groupby("seq_id").datetime.transform("min")
    d["fifth"] = -1
    te = (d.part == "test") & d.kn
    d.loc[te, "fifth"] = (d[te].assign(st=start[te]).groupby("client").st.rank(pct=True) * 5).clip(upper=4.999).astype(int)
    calx = d[(d.part.isin(["cal", "tune"])) & d.kn].groupby("client").miss.agg(["mean", "size"])
    lines = []
    prev = calx
    for f in range(5):
        tx = d[d.fifth == f].groupby("client").miss.agg(["mean", "size"])
        for name, ref in [("cal", calx), ("prev", prev)]:
            j = ref.join(tx, lsuffix="_a", rsuffix="_b", how="inner")
            j = j[(j.size_a >= MIN) & (j.size_b >= MIN)]
            if len(j) > 5:
                r, b = wcorr_slope(j.mean_a.to_numpy(), j.mean_b.to_numpy(),
                                   np.minimum(j.size_a, j.size_b).to_numpy().astype(float))
                lines.append(f"{name}->T{f}: r={r:+.2f} slope={b:+.2f} (n={len(j)})")
        prev = tx
    print(f"seed {seed}: " + " | ".join(lines), flush=True)
