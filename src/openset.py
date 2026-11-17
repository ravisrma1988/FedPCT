"""Online open-set component: an "unknown species" label with its own online error control.
DEVELOPMENT sites only; known classes use the static pooled RAPS threshold (isolates this part).

Novelty score n(x) (higher = more novel), chosen by AUROC on the TUNE partition only, among
  msp     1 - max softmax of the federated head
  energy  -logsumexp(logits)
  proto   1 - max_c cos(x, mu_c), mu_c = class means of train features (federated aggregates)
Unknown flag: "unknown" is added to the set iff n(x) > nu. Tracked quantity t = -n, threshold
theta = -nu; an unknown image is missed iff t > theta. Target: unknown recall >= 1 - ALPHA_U.
  start   theta_0 = pooled conformal quantile of t over cal unknowns of ALL sites (a single site
          has too few unknowns: federation needed)
  frozen  theta_0 fixed;  static frontier at alpha_u in FRONTIER_U
  local   per-site tracking from the site's own unknown feedback (eta_l)
  shared  one federated threshold: the server pools all sites' unknown feedback (eta_g)
Feedback: uniform labelling at rate r, delay DELAY; annotators report "not a known species".
eta for local/shared chosen on the tune replay (pinball loss on tune unknowns); grid also kept.
Metrics: pooled recall over all dev-site unknown test images; per-site recall for sites with
>= MIN_UNK unknowns (+ share significantly below 1 - ALPHA_U, sequence-cluster bootstrap);
false-unknown rate on known images (mean over sites); overall coverage (known via tau, unknown via flag).
Usage: python openset.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import cp_baselines as cp
import scores
from lockbox import dev_mask
from online import DELAY, stream_order
from online2 import labels_for, track2
from online_gpu import pack
from run_iwildcam import ALPHA, DEV, fedavg
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
ALPHA_U = 0.2
FRONTIER_U = (0.1, 0.2, 0.3)
RATES = (0.1, 0.3, 1.0)
ETAS = (0.005, 0.02, 0.05, 0.2)
MIN_UNK = 20


def novelty_scores(glob, X, tr_X, tr_y, C):
    with torch.no_grad():
        logits = glob(X * 10)
        P = torch.softmax(logits, 1)
        mu = torch.zeros(C, X.shape[1], device=DEV).index_add_(0, tr_y, tr_X)
        mu = torch.nn.functional.normalize(mu, dim=1)
        Xn = torch.nn.functional.normalize(X, dim=1)
        out = {"msp": 1 - P.max(1).values,
               "energy": -torch.logsumexp(logits, 1),
               "proto": 1 - (Xn @ mu.T).max(1).values}
    return {k: v.cpu().numpy() for k, v in out.items()}, P.cpu().numpy()


def unk_streams(s, keep, n, y, seqs):
    sites = []
    for k in keep:
        idx = np.flatnonzero((s.part.to_numpy() == "test") & (s.client.to_numpy() == k))
        idx = idx[stream_order(s.iloc[idx])]
        sq = seqs[idx]
        t = np.where(y[idx] == cp.UNK, -n[idx], np.nan)
        sites.append(dict(idx=idx, seq=sq, seq_codes=np.cumsum(np.r_[0, sq[1:] != sq[:-1]]), score=t,
                          t_all=-n[idx], y=y[idx]))
    return sites


def evaluate(thetas, sites, S_known, tau):
    """thetas (G,K,J). Returns per-config dict of metrics."""
    G = thetas.shape[0]
    res = []
    for g in range(G):
        rec_num = rec_den = 0
        site_rec, site_sig, ff, cov = [], [], [], []
        for k, st in enumerate(sites):
            th = thetas[g, k, torch.as_tensor(st["seq_codes"], device=DEV)].cpu().numpy()
            flag = st["t_all"] < th                       # unknown label added
            unk = st["y"] == cp.UNK
            kn = ~unk
            if unk.any():
                rec_num += flag[unk].sum(); rec_den += unk.sum()
                if unk.sum() >= MIN_UNK:
                    site_rec.append(flag[unk].mean())
                    site_sig.append(cp.cluster_upper_bound(flag[unk], st["seq"][unk]) < 1 - ALPHA_U)
            if kn.any():
                ff.append(flag[kn].mean())
            hit_known = S_known[k] <= tau if S_known[k] is not None else np.zeros(0, bool)
            cov.append((hit_known.sum() + flag[unk].sum()) / len(st["y"]))
        res.append(dict(recall=rec_num / max(rec_den, 1), site_recall=float(np.mean(site_rec)) if site_rec else np.nan,
                        pct_sites_sig=100 * float(np.mean(site_sig)) if site_sig else np.nan,
                        n_sites_unk=len(site_rec), false_flag=100 * float(np.mean(ff)), cov_all=float(np.mean(cov))))
    return res


def run(seed, tag):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    ytr = torch.from_numpy(y[tr]).to(DEV)
    glob = fedavg(X[tr], ytr, client[tr], C, seed)
    nov, P = novelty_scores(glob, X, X[tr], ytr, C)

    tm, cm = part == "tune", part == "cal"
    auroc = {k: float(roc_auc_score(y[tm] == cp.UNK, v[tm])) for k, v in nov.items()}
    best = max(auroc, key=auroc.get)
    n = nov[best]
    test_auroc = {k: float(roc_auc_score(y[part == "test"] == cp.UNK, v[part == "test"])) for k, v in nov.items()}

    # known-class part: static pooled RAPS threshold with the global head
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    S = scores.raps(P, lam, kreg, seed + 7)
    tau = cp.thresholds("pooled", dict(S=S[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]

    keep = [k for k in range(K) if dev[k] and ((part == "test") & (client == k)).any()]
    sites = unk_streams(s, keep, n, y, seqs)
    S_known = [cp.true_scores(S[st["idx"]], st["y"])[st["y"] != cp.UNK] for st in sites]
    pad, nk = pack(sites)
    J = pad.shape[1]
    cal_unk = cm & (y == cp.UNK)
    out = dict(auroc_tune=auroc, auroc_test=test_auroc, chosen=best, n_cal_unknown=int(cal_unk.sum()))

    # static frontier
    th0 = {a: cp.conformal_quantile(-n[cal_unk], a) for a in FRONTIER_U}
    static = torch.stack([torch.full((len(sites), J), th0[a], device=DEV) for a in FRONTIER_U])
    out["static"] = dict(zip(map(str, FRONTIER_U), evaluate(static, sites, S_known, tau)))

    # tune replay for eta: streams over the tune partition of all sites with tune unknowns
    tune_sites = []
    for k in range(K):
        idx = np.flatnonzero(tm & (client == k))
        if len(idx) == 0 or not (y[idx] == cp.UNK).any():
            continue
        idx = idx[stream_order(s.iloc[idx])]
        sq = seqs[idx]
        tune_sites.append(dict(seq_codes=np.cumsum(np.r_[0, sq[1:] != sq[:-1]]),
                               score=np.where(y[idx] == cp.UNK, -n[idx], np.nan)))
    tpad, tnk = pack(tune_sites)
    th_tune = cp.conformal_quantile(-n[tm & (y == cp.UNK)], ALPHA_U)
    E = torch.tensor(ETAS, device=DEV)
    Z = torch.zeros(len(ETAS), device=DEV)

    def pin(taus, sts):
        tot = cnt = 0.0
        for k, st in enumerate(sts):
            m = ~np.isnan(st["score"])
            t = taus[:, k, torch.as_tensor(st["seq_codes"][m], device=DEV)]
            d = torch.as_tensor(st["score"][m], dtype=torch.float32, device=DEV)[None] - t
            tot = tot + torch.where(d >= 0, (1 - ALPHA_U) * d, -ALPHA_U * d).sum(1)
            cnt += m.sum()
        return (tot / max(cnt, 1)).cpu().numpy()

    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        lab = labels_for(sites, rate, 10_000 + seed, J)
        for fam, (el, eg) in {"local": (E, Z), "shared": (Z, E)}.items():
            tt = track2(tpad, tnk, lab_t[None].expand(len(ETAS), -1, -1),
                        torch.full((len(tune_sites),), th_tune, device=DEV), el, eg, alpha=ALPHA_U)
            g_star = int(np.argmin(pin(tt, tune_sites)))
            ts = track2(pad, nk, lab[None].expand(len(ETAS), -1, -1),
                        torch.full((len(sites),), th0[ALPHA_U], device=DEV), el, eg, alpha=ALPHA_U)
            ev = evaluate(ts, sites, S_known, tau)
            out[f"{fam}|r={rate}"] = dict(eta=ETAS[g_star], selected=ev[g_star],
                                         grid={str(e): ev[g] for g, e in enumerate(ETAS)})
    return out


if __name__ == "__main__":
    n_seeds, tag = int(sys.argv[1]), sys.argv[2]
    allr = []
    for sd in range(n_seeds):
        r = run(sd, tag)
        allr.append(r)
        print(f"seed {sd}: chosen {r['chosen']}  AUROC tune {r['auroc_tune']}  test {r['auroc_test']}  "
              f"cal unknowns {r['n_cal_unknown']}", flush=True)
    json.dump(allr, open(f"{ROOT}/results/openset_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== open-set, DEV sites, features={tag}, target unknown recall {1 - ALPHA_U:.2f}, delay={DELAY}, {n_seeds} seeds")
    print(f"{'method':<22}{'recall':>8}{'site_rec':>9}{'%sites<':>9}{'false_flag%':>12}{'cov_all':>9}  eta")

    def line(name, d, eta=""):
        print(f"{name:<22}{d['recall']:8.3f}{d['site_recall']:9.3f}{d['pct_sites_sig']:8.0f}%{d['false_flag']:11.1f}%"
              f"{d['cov_all']:9.3f}  {eta}")

    for a in map(str, FRONTIER_U):
        line(f"static a_u={a}", {k: np.nanmean([r["static"][a][k] for r in allr]) for k in allr[0]["static"][a]})
    for key in [k for k in allr[0] if "|r=" in k]:
        etas = [r[key]["eta"] for r in allr]
        line(key, {k: np.nanmean([r[key]["selected"][k] for r in allr]) for k in allr[0][key]["selected"]},
             max(set(etas), key=etas.count))
