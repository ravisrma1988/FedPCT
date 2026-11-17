"""Sanity checks and failure-mode preview on synthetic data.

Scenario A (iid, no unknowns): every method should sit at ~1-alpha coverage.
Scenario B (label shift + covariate shift + unknowns): expected failures --
  pooled/labelshift lose per-client coverage, closed-set methods miss all unknowns,
  local is valid but has large/unstable sets at small clients.
"""
import json
import sys

import numpy as np

import cp_baselines as cp
import synthetic

ALPHA = 0.1
SCENARIOS = {
    "A_iid": dict(beta=0, shift=0.0, unk_clients=0.0),
    "B_shift_openset": dict(beta=0.3, shift=0.7, unk_clients=0.5),
}


def run(scen, seed):
    data, meta = synthetic.make(seed=seed, **SCENARIOS[scen])
    K, C = meta["K"], meta["C"]
    cal, te = data["cal"], data["test"]
    res = {}
    for m in ["local", "pooled", "labelshift"]:
        tau = cp.thresholds(m, cal, ALPHA, C, K)
        sets = cp.predict_sets(1 - te["probs"], te["client"], tau)
        res[m] = cp.summarize(cp.evaluate(sets, te["y"], te["client"], K), ALPHA)
    # Naive open-set: pooled threshold + max-softmax novelty flag with pooled FPR control.
    tau = cp.thresholds("pooled", cal, ALPHA, C, K)
    known = cal["y"] != cp.UNK
    nov_cal = 1 - cal["probs"][known].max(1)
    nov_tau = cp.conformal_quantile(nov_cal, ALPHA)
    sets = cp.predict_sets(1 - te["probs"], te["client"], tau, 1 - te["probs"].max(1), nov_tau)
    rows = cp.evaluate(sets, te["y"], te["client"], K)
    res["pooled+msp"] = cp.summarize(rows, ALPHA)
    rec = [r["unk_recall"] for r in rows if not np.isnan(r["unk_recall"])]
    res["pooled+msp"]["unk_recall"] = float(np.mean(rec)) if rec else float("nan")
    return res


if __name__ == "__main__":
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    for scen in SCENARIOS:
        allr = [run(scen, s) for s in seeds]
        print(f"\n== {scen}  (alpha={ALPHA}, target coverage {1-ALPHA:.2f}, {len(allr)} seeds)")
        print(f"{'method':<12}{'mean_cov':>9}{'worst_cov':>10}{'worst_known':>12}"
              f"{'%clients<0.88':>15}{'set_size':>9}{'unk_recall':>11}")
        for m in allr[0]:
            agg = {k: np.mean([r[m][k] for r in allr]) for k in allr[0][m]}
            sd = np.std([r[m]["worst_cov"] for r in allr])
            print(f"{m:<12}{agg['mean_cov']:9.3f}{agg['worst_cov']:7.3f}±{sd:.2f}"
                  f"{agg['worst_cov_known']:12.3f}"
                  f"{100*agg['frac_clients_below']:14.0f}%{agg['mean_size']:9.2f}"
                  f"{agg.get('unk_recall', float('nan')):11.2f}")
        json.dump(allr, open(f"results/synthetic_{scen}.json", "w"), indent=1)
