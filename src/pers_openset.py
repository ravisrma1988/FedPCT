"""Integrated method: federated personalisation (Ditto) + per-site online conformal tracking for the
known classes + a federated "unknown species" flag with its own online threshold. DEVELOPMENT sites.

One feedback stream per site (uniform labelling at rate r, delay DELAY; selective labelling gave no
gain): a labelled sequence gives the true species, or "not a known species". The same labels drive
  * Ditto refits (known-class labels only, every REFIT_EVERY),
  * tau  (known classes, RAPS score of the head in force): frozen | qt (eta* from tune replay),
  * theta (unknown flag, novelty n = 1 - max prob of the head in force; flag iff -n < theta):
      frozen | local | shared, start = pooled conformal quantile over cal unknowns of all sites
      (global head), eta_u per family from the tune replay (as in openset.py).
A personalised head changes the novelty scale, so a theta calibrated on the global head may drift;
the online theta is what should absorb that.
Metrics: known part as before (cov, %sites sig below 1-ALPHA, size); unknown recall (pooled,
%sites with >= MIN_UNK unknowns sig below 1-ALPHA_U), false-flag rate on knowns (mean over
sites), joint coverage (known: true label in set; unknown: flagged), novelty AUROC of the head.
Usage: python pers_openset.py <n_seeds> <feature tag>
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
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from openset import ALPHA_U, ETAS, MIN_UNK
from pers_select import simulate
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
RATES = (0.1, 0.3, 1.0)
MODES = ("global", "ditto")
TH_FAMS = ("frozen", "local", "shared")


def eta_u_choice(s, part, y, seqs, client, n, rate, seed, K):
    """Tune replay for the unknown threshold (global-head novelty), per family -> eta."""
    tm = part == "tune"
    tsites = []
    for k in range(K):
        idx = np.flatnonzero(tm & (client == k))
        if len(idx) == 0 or not (y[idx] == cp.UNK).any():
            continue
        idx = idx[stream_order(s.iloc[idx])]
        sq = seqs[idx]
        tsites.append(dict(seq_codes=np.cumsum(np.r_[0, sq[1:] != sq[:-1]]),
                           score=np.where(y[idx] == cp.UNK, -n[idx], np.nan)))
    tpad, tnk = pack(tsites)
    th = torch.full((len(tsites),), cp.conformal_quantile(-n[tm & (y == cp.UNK)], ALPHA_U), device=DEV)
    lab = labels_for(tsites, rate, 20_000 + seed, tpad.shape[1])[None].expand(len(ETAS), -1, -1)
    E, Z = torch.tensor(ETAS, device=DEV), torch.zeros(len(ETAS), device=DEV)
    out = {}
    for fam, (el, eg) in {"local": (E, Z), "shared": (Z, E)}.items():
        tt = track2(tpad, tnk, lab, th, el, eg, alpha=ALPHA_U)
        loss = np.zeros(len(ETAS))
        cnt = 0
        for k, st in enumerate(tsites):
            m = ~np.isnan(st["score"])
            t = tt[:, k, torch.as_tensor(st["seq_codes"][m], device=DEV)]
            d = torch.as_tensor(st["score"][m], dtype=torch.float32, device=DEV)[None] - t
            loss += torch.where(d >= 0, (1 - ALPHA_U) * d, -ALPHA_U * d).sum(1).cpu().numpy()
            cnt += m.sum()
        out[fam] = ETAS[int(np.argmin(loss / max(cnt, 1)))]
    return out


def run(seed, tag):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    n_glob = 1 - P_all.max(1)
    tm, cm = part == "tune", part == "cal"
    lam, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam, kreg, seed + 7)  # noqa: E731
    S_glob = score_fn(P_all)
    tau0 = cp.thresholds("pooled", dict(S=S_glob[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    th0 = cp.conformal_quantile(-n_glob[cm & (y == cp.UNK)], ALPHA_U)
    tune_sites = [st for st in build_streams(s, "tune", S_glob, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_glob[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]

    sites_all = build_streams(s, "test", S_glob, y, seqs)
    keep = [k for k, st in enumerate(sites_all) if st is not None and dev[k]]
    base, seen_t = [sites_all[k] for k in keep], seen[keep]
    out = {}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta_star = pos[int(np.argmin(pinball(tt, tune_sites)))]
        eta_u = eta_u_choice(s, part, y, seqs, client, n_glob, rate, seed, K)
        for mode in MODES:
            rng = np.random.default_rng(10_000 + seed)  # same label draws for both modes
            sites, usites, Ls = [], [], []
            for st in base:
                ys = y[st["idx"]]
                P, L, _ = simulate(st, X[st["idx"]], ys, glob, mode, rate, "uniform", rng)
                S = score_fn(P)
                nv = 1 - P.max(1)
                sites.append(dict(st, S=S, score=cp.true_scores(S, ys)))
                usites.append(dict(seq_codes=st["seq_codes"], seq=st["seq"], y=ys, t_all=-nv,
                                   score=np.where(ys == cp.UNK, -nv, np.nan)))
                Ls.append(L)
            J = max(len(L) for L in Ls)
            W = np.zeros((len(sites), J), np.float32)
            for k, L in enumerate(Ls):
                W[k, :len(L)] = L
            W = torch.as_tensor(W, device=DEV)
            # known classes: frozen / qt
            pad, nk = pack(sites)
            taus = track2(pad, nk, W[None].expand(2, -1, -1), torch.full((len(sites),), tau0, device=DEV),
                          torch.tensor([0.0, eta_star], device=DEV), torch.zeros(2, device=DEV))
            ev = site_eval(taus, sites, seen_t)
            # unknown flag: frozen / local / shared
            upad, unk_n = pack(usites)
            el = torch.tensor([0.0, eta_u["local"], 0.0], device=DEV)
            eg = torch.tensor([0.0, 0.0, eta_u["shared"]], device=DEV)
            ths = track2(upad, unk_n, W[None].expand(3, -1, -1), torch.full((len(usites),), th0, device=DEV),
                         el, eg, alpha=ALPHA_U)
            yy = np.concatenate([u["y"] for u in usites])
            nn = -np.concatenate([u["t_all"] for u in usites])
            auroc = float(roc_auc_score(yy == cp.UNK, nn))
            for gi, tname in enumerate(("frozen", "qt")):
                for hi, uname in enumerate(TH_FAMS):
                    rec_n = rec_d = 0
                    srec, ff, joint = [], [], []
                    for k, (st, u) in enumerate(zip(sites, usites)):
                        codes = torch.as_tensor(u["seq_codes"], device=DEV)
                        flag = u["t_all"] < ths[hi, k, codes].cpu().numpy()
                        unk = u["y"] == cp.UNK
                        hit_known = st["score"][~unk] <= taus[gi, k, codes].cpu().numpy()[~unk]
                        rec_n += flag[unk].sum(); rec_d += unk.sum()
                        if unk.sum() >= MIN_UNK:
                            srec.append(cp.cluster_upper_bound(flag[unk], u["seq"][unk]) < 1 - ALPHA_U)
                        if (~unk).any():
                            ff.append(flag[~unk].mean())
                        joint.append((hit_known.sum() + flag[unk].sum()) / len(unk))
                    out[f"{mode}|r={rate}|tau={tname}|theta={uname}"] = dict(
                        summary(ev[gi]), recall=rec_n / max(rec_d, 1), usig=100 * float(np.mean(srec)),
                        ff=100 * float(np.mean(ff)), joint=float(np.mean(joint)), auroc=auroc,
                        eta=eta_star, eta_u=eta_u.get(uname, 0.0))
            print(f"  seed {seed} r={rate} {mode} done (novelty AUROC {auroc:.3f})", flush=True)
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = []
    for sd in range(n):
        allr.append(run(sd, tag))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{ROOT}/results/pers_openset_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== integrated (Ditto + tracking + unknown flag), DEV sites, features={tag}, RAPS, delay={DELAY}, "
          f"alpha={ALPHA}, alpha_u={ALPHA_U}, {n} seeds")
    print(f"{'mode|rate|tau|theta':<36}{'cov':>6}{'%sig':>6}{'size':>6}{'recall':>8}{'%usig':>7}{'ff%':>7}"
          f"{'joint':>7}{'auroc':>7}")
    for key in allr[0]:
        m = {k: np.mean([r[key][k] for r in allr]) for k in allr[0][key]}
        print(f"{key:<36}{m['cov']:6.3f}{m['sig']:5.0f}%{m['size']:6.2f}{m['recall']:8.3f}{m['usig']:6.0f}%"
              f"{m['ff']:6.1f}%{m['joint']:7.3f}{m['auroc']:7.3f}")
