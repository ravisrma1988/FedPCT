"""Online (deployment-time) evaluation on the validated time splits.

Stream: at each site, the test partition is processed in time order (by sequence start).
Feedback: each sequence is labelled with probability R (same draw for every method); its labels
become available DELAY sequences later. Only known-class labelled images drive updates.
Score 1 - p_y; prediction set {y : 1 - p_y <= tau}.

Methods (per site):
  static_pooled / static_local / static_labelshift : thresholds fixed after calibration (tune+cal)
  qt_local       : quantile tracking (online gradient on tau), initialised at static_local
  qt_pooled_init : the same update, initialised at static_pooled (federation-informed start)
Quantile tracking: tau <- tau + eta * (err - alpha), err = 1[score > tau used at prediction time]
(the update behind ACI / conformal PID control). eta is NOT tuned on test: every value in ETAS is
reported as a sensitivity analysis.
Metrics on ALL known-class test images (labelled or not): long-run coverage, set size, share of
sites significantly below target (sequence-cluster bootstrap), share of 200-image windows < 0.85.
Usage: python online.py <n_seeds>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
RATES, DELAY, ETAS, WINDOW = (0.1, 0.3, 1.0), 5, (0.005, 0.02, 0.05), 200


def stream_order(s_te):
    """Time order by sequence start; seq_id breaks ties so each sequence stays contiguous."""
    start = s_te.groupby("seq_id").datetime.transform("min")
    _, seq_code = np.unique(s_te.seq_id.to_numpy(), return_inverse=True)
    return np.lexsort((s_te.datetime.to_numpy(), seq_code, start.to_numpy()))


def run_site(score_known, seq_codes, labelled_seq, tau0, eta):
    """score_known: (n,) score of the true label (nan for unknown-species rows), in stream order.
    Returns tau used for each image."""
    n = len(score_known)
    taus = np.empty(n)
    tau = tau0
    seq_bounds = np.flatnonzero(np.r_[True, seq_codes[1:] != seq_codes[:-1], True])
    pending = []  # (reveal_after_seq_index, start, end)
    for j in range(len(seq_bounds) - 1):
        a, b = seq_bounds[j], seq_bounds[j + 1]
        # apply feedback that has arrived before this sequence
        while pending and pending[0][0] <= j:
            _, pa, pb = pending.pop(0)
            for i in range(pa, pb):
                if not np.isnan(score_known[i]):
                    err = float(score_known[i] > taus[i])
                    tau = tau + eta * (err - ALPHA)
        taus[a:b] = tau
        if labelled_seq[seq_codes[a]]:
            pending.append((j + DELAY + 1, a, b))
    return taus


def site_metrics(hit, size, seq, alpha):
    w = [hit[i:i + WINDOW].mean() for i in range(0, len(hit) - WINDOW + 1, WINDOW)]
    return dict(cov=float(hit.mean()), size=float(size.mean()),
                sig_below=cp.cluster_upper_bound(hit, seq) < 1 - alpha,
                win_below=float(np.mean(np.array(w) < 0.85)) if w else np.nan)


FRONTIER_ALPHAS = (0.1, 0.07, 0.05, 0.03, 0.02)  # static pooled at stricter levels: size-coverage frontier


def run(seed, Xall, score_name):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    seqs = s.seq_id.to_numpy()
    tr = part == "train"
    head = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)

    if score_name == "lac":  # no hyper-parameters: calibrate on tune + cal
        score_fn = scores.lac
        calm = np.isin(part, ["tune", "cal"])
    else:  # RAPS: (lam, kreg) from tune only, calibrate on cal only
        tm = part == "tune"
        lam, kreg = scores.tune_raps(probs_of(head, X[tm]), y[tm], seqs[tm], ALPHA, seed)
        score_fn = lambda P: scores.raps(P, lam, kreg, seed + 7)  # noqa: E731
        calm = part == "cal"
    cal = dict(S=score_fn(probs_of(head, X[calm])), y=y[calm], client=client[calm])
    static = {m: cp.thresholds(m, cal, ALPHA, C, K) for m in ["pooled", "local", "labelshift"]}
    frontier = {a: cp.thresholds("pooled", cal, a, C, K) for a in FRONTIER_ALPHAS[1:]}

    rng = np.random.default_rng(10_000 + seed)
    out = []
    for k in range(K):
        idx = np.flatnonzero((part == "test") & (client == k))
        if len(idx) == 0:
            continue
        idx = idx[stream_order(s.iloc[idx])]
        S = score_fn(probs_of(head, X[idx]))
        yy = y[idx]
        kn = yy != cp.UNK
        score = cp.true_scores(S, yy)
        seq = seqs[idx]
        seq_codes = np.cumsum(np.r_[0, seq[1:] != seq[:-1]])  # contiguous, in stream order
        u = rng.random(seq_codes.max() + 1)

        def evaluate(taus):
            hit = (score[kn] <= taus[kn])
            size = (S <= taus[:, None]).sum(1)
            return site_metrics(hit, size[kn], seq[kn], ALPHA)

        row = dict(client=k, seen=bool(seen[k]), n=int(kn.sum()))
        for m, t in static.items():
            row[f"static_{m}"] = evaluate(np.full(len(idx), t[k]))
        for a, t in frontier.items():
            row[f"static_pooled|alpha={a}"] = evaluate(np.full(len(idx), t[k]))
        for r in RATES:
            lab = u < r
            for eta in ETAS:
                for name, t0 in [("qt_local", static["local"][k]), ("qt_pooled_init", static["pooled"][k])]:
                    t0 = min(t0, float(S.max()))  # local quantile can be +inf for tiny cal sets
                    row[f"{name}|r={r}|eta={eta}"] = evaluate(run_site(score, seq_codes, lab, t0, eta))
        out.append(row)
    return out


def summarise(all_rows):
    keys = [k for k in all_rows[0][0] if k not in ("client", "seen", "n")]
    print(f"{'method':<34}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'%win<0.85':>10}{'size':>7}")
    for key in keys:
        agg = []
        for rows in all_rows:
            v = [r[key] for r in rows]
            unseen = [r[key] for r in rows if not r["seen"]]
            agg.append([np.mean([x["cov"] for x in v]), 100 * np.mean([x["sig_below"] for x in v]),
                        100 * np.mean([x["sig_below"] for x in unseen]),
                        100 * np.nanmean([x["win_below"] for x in v]), np.mean([x["size"] for x in v])])
        a = np.mean(agg, 0)
        print(f"{key:<34}{a[0]:7.3f}{a[1]:6.0f}%{a[2]:11.0f}%{a[3]:9.0f}%{a[4]:7.2f}")


if __name__ == "__main__":
    Xall = np.load(f"{ROOT}/features/iwildcam_bioclip.npy")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    score_name = sys.argv[2] if len(sys.argv) > 2 else "lac"
    all_rows = []
    for sd in range(n):
        all_rows.append(run(sd, Xall, score_name))
        print(f"seed {sd} done", flush=True)
    json.dump(all_rows, open(f"{ROOT}/results/online_time_{score_name}.json", "w"), default=float)
    print(f"\n== online, time split, score={score_name}, alpha={ALPHA}, delay={DELAY} seqs, {n} seeds "
          f"(cov/size: mean over sites, known classes, all test images)")
    summarise(all_rows)
