"""Is there drift that sites share? Static pooled RAPS threshold (alpha=0.1) on test; per site the
miss indicator of known-class images. Variance of site x bin excess miss rate is decomposed into
  common  : bin means across sites  (what a shared component g can track)
  site    : site means              (what a per-site offset d_k can track)
  residual: the rest
for two time axes: (a) position in the site's own test stream (decile; "staleness since
calibration"), (b) calendar month (season). Weighted by images per cell; cells with < MIN images
dropped. A permutation null (bins shuffled within site) gives the common share expected by chance.
Usage: python diag_common_drift.py <feature tag> <n_seeds>
"""
import sys

import numpy as np
import pandas as pd
import torch

import cp_baselines as cp
import scores
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT, MIN, PERM = "${FEDPCT_ROOT}", 20, 200


def decompose(df, col, rng):
    d = df.groupby(["client", col]).agg(m=("miss", "mean"), n=("miss", "size")).reset_index()
    d = d[d.n >= MIN]
    d["e"] = d.m - ALPHA

    def shares(dd):
        w = dd.n / dd.n.sum()
        mu = np.average(dd.e, weights=w)
        tot = np.average((dd.e - mu) ** 2, weights=w)
        site = dd.groupby("client").apply(lambda g: np.average(g.e, weights=g.n), include_groups=False)
        dd = dd.assign(e1=dd.e - dd.client.map(site))
        binm = dd.groupby(col).apply(lambda g: np.average(g.e1, weights=g.n), include_groups=False)
        common = np.average(dd[col].map(binm) ** 2, weights=w)
        site_v = np.average((dd.client.map(site) - mu) ** 2, weights=w)
        return common / tot, site_v / tot

    c, sv = shares(d)
    null = [shares(d.assign(**{col: d.groupby("client")[col].transform(lambda x: rng.permutation(x.to_numpy()))}))[0]
            for _ in range(PERM)]
    return dict(common=c, site=sv, null_mean=float(np.mean(null)), null_p95=float(np.quantile(null, 0.95)),
                cells=len(d), sites=d.client.nunique())


if __name__ == "__main__":
    tag, n = sys.argv[1], int(sys.argv[2])
    rows = []
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
        te = (part == "test") & (y != cp.UNK)
        df = s[te].assign(miss=(cp.true_scores(S[te], y[te]) > tau).astype(float))
        start = df.groupby("seq_id").datetime.transform("min")
        df["decile"] = df.assign(st=start).groupby("client").st.rank(pct=True).mul(10).clip(upper=9.999).astype(int)
        df["month"] = pd.to_datetime(df.datetime).dt.month
        rng = np.random.default_rng(seed)
        for col in ["decile", "month"]:
            r = decompose(df, col, rng)
            r.update(seed=seed, axis=col)
            rows.append(r)
            print(f"seed {seed} {col:<7} common {r['common']:.3f} (null {r['null_mean']:.3f}, p95 {r['null_p95']:.3f})"
                  f"  site {r['site']:.3f}  cells {r['cells']} sites {r['sites']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{ROOT}/results/diag_common_drift_{tag.replace('{seed}', 'S')}.csv", index=False)
    print("\nmean over seeds:\n", out.groupby("axis")[["common", "null_mean", "null_p95", "site"]].mean().round(3).to_string())
