"""Final evaluation of the frozen method (PREREG.md) and all comparators, on ONE set of sites.

  python lockbox_eval.py dev       dry run on development sites (must reproduce development numbers)
  python lockbox_eval.py lockbox   the one-time lockbox evaluation. Refuses to run unless
                                   splits/prereg_freeze.json exists and every hashed file is
                                   unchanged, and refuses to run a second time.

Everything is computed exactly as in development (same functions, seeds and label draws):
  global head: FCP (pooled), label-shift, quantile-of-quantiles, density-ratio (static);
               per-site tracking at rate r (comparator 2)
  Ditto(lambda*(r, seed)): fixed-threshold frontier (comparator 3/4) and tracking (the method),
               plus the unknown flag (fixed nu) for global and Ditto heads.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import cp_baselines as cp
import scores
from baselines_lit import evaluate, qq_choose, qq_threshold, site_classifier, weighted_tau
from lockbox import dev_mask
from online import DELAY, stream_order
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from openset import ALPHA_U, MIN_UNK
from pers_frontier import FRONTIER, interp
from pers_select import simulate
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
TAG = "bioclip2crop"
SEEDS = range(5)
RATES = (0.1, 0.3, 1.0)
FREEZE = f"{ROOT}/splits/prereg_freeze.json"
DONE = f"{ROOT}/results/LOCKBOX_RUN.json"
HASHED = ["PREREG.md", "src/lockbox_eval.py", "src/baselines_lit.py", "src/pers_select.py", "src/pers_online.py",
          "src/pers_frontier.py", "src/online2.py", "src/online.py", "src/online_gpu.py", "src/openset.py",
          "src/cp_baselines.py", "src/scores.py", "src/run_iwildcam.py", "src/splits.py", "src/lockbox.py",
          "results/pers_lambda_bioclip2crop.json", "splits/lockbox_sites.json"]


def sha(path):
    return hashlib.sha256(open(f"{ROOT}/{path}", "rb").read()).hexdigest()


def unknown_metrics(sites, nov, ys_list, th0):
    rec_n = rec_d = 0
    usig, ff = [], []
    for st, n_, ys in zip(sites, nov, ys_list):
        flag, unk = -n_ < th0, ys == cp.UNK
        rec_n += flag[unk].sum(); rec_d += unk.sum()
        if unk.sum() >= MIN_UNK:
            usig.append(cp.cluster_upper_bound(flag[unk], st["seq"][unk]) < 1 - ALPHA_U)
        ff.append(flag[~unk].mean())
    ya, na = np.concatenate(ys_list), np.concatenate(nov)
    return dict(recall=rec_n / max(rec_d, 1), usig=100 * float(np.mean(usig)) if usig else np.nan,
                n_usites=len(usig), ff=100 * float(np.mean(ff)), auroc=float(roc_auc_score(ya == cp.UNK, na)))


def stream_run(base, X, y, glob, mode, rate, lam, seed, score_fn):
    rng = np.random.default_rng(10_000 + seed)  # same draws as development
    sites, Ls, nov, yy = [], [], [], []
    for st in base:
        ys = y[st["idx"]]
        P, L, _ = simulate(st, X[st["idx"]], ys, glob, mode, rate, "uniform", rng, lam)
        S = score_fn(P)
        sites.append(dict(st, S=S, score=cp.true_scores(S, ys)))
        Ls.append(L), nov.append(1 - P.max(1)), yy.append(ys)
    pad, nk = pack(sites)
    W = np.zeros((len(sites), pad.shape[1]), np.float32)
    for k, L in enumerate(Ls):
        W[k, :len(L)] = L
    return sites, pad, nk, torch.as_tensor(W, device=DEV), nov, yy


def run(seed, which, lam_star):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    mask = dev_mask(info) if which == "dev" else ~dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    cal = dict(S=S_all[cm], y=y[cm], client=client[cm])
    tau_fr = [cp.thresholds("pooled", cal, a, C, K)[0] for a in FRONTIER]
    tau0 = tau_fr[0]
    th0 = cp.conformal_quantile(-(1 - P_all.max(1))[cm & (y == cp.UNK)], ALPHA_U)
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and mask[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    out = dict(n_sites=len(keep), n_unseen=int((~seen_t).sum()))

    # --- static literature baselines (global head) ---
    rng = np.random.default_rng(40_000 + seed)
    tau_ls = cp.thresholds("labelshift", cal, ALPHA, C, K)
    calk = cm & (y != cp.UNK)
    Sc = cp.true_scores(S_all[calk], y[calk])
    sizes = np.bincount(client[calk], minlength=K)
    beta, kq, _ = qq_choose(sizes[sizes > 0], ALPHA, rng)
    tau_qq = qq_threshold(Sc, client[calk], K, beta, kq)
    Xc = X[torch.as_tensor(calk, device=DEV)]
    Sc_t = torch.as_tensor(Sc, dtype=torch.float32, device=DEV)
    folds = np.unique(seqs[calk], return_inverse=True)[1] % 2
    lit = {m: [] for m in ("fcp", "labelshift", "qq", "dr")}
    for c, st in zip(keep, base):
        S, sc, sq, n = st["S"], st["score"], st["seq"], len(st["idx"])
        lit["fcp"].append(evaluate(S, sc, np.full(n, tau0), sq, seen[c]))
        lit["labelshift"].append(evaluate(S, sc, np.full(n, tau_ls[c]), sq, seen[c]))
        lit["qq"].append(evaluate(S, sc, np.full(n, tau_qq), sq, seen[c]))
        is_site = torch.as_tensor(client[calk] == c, device=DEV)
        if is_site.sum() < 10:
            lit["dr"].append(evaluate(S, sc, np.full(n, tau0), sq, seen[c]))
            continue
        p_cal, model = site_classifier(Xc, is_site, folds)
        ratio = (1 - is_site.float().mean()) / is_site.float().mean()
        to_w = lambda p: (p.clamp(1e-4, 1 - 1e-4) / (1 - p.clamp(1e-4, 1 - 1e-4))) * ratio  # noqa: E731
        tau = weighted_tau(Sc_t, to_w(p_cal), to_w(model(X[torch.as_tensor(st["idx"], device=DEV)])), ALPHA).cpu().numpy()
        lit["dr"].append(evaluate(S, sc, tau, sq, seen[c]))
    for m, rows in lit.items():
        out[m] = dict(cov=np.mean([r["cov"] for r in rows]), size=np.mean([r["size"] for r in rows]),
                      sig=100 * np.mean([r["sig"] for r in rows]),
                      sig_unseen=100 * np.mean([r["sig"] for r in rows if not r["seen"]]) if (~seen_t).any() else np.nan)

    # --- online: global + tracking, Ditto frontier + tracking, unknown flag ---
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        r = dict(eta=eta, lam=lam_star[rate])
        sites, pad, nk, W, nov, yy = stream_run(base, X, y, glob, "global", rate, None, seed, score_fn)
        taus = track2(pad, nk, W[None], torch.full((len(sites),), tau0, device=DEV),
                      torch.tensor([eta], device=DEV), torch.zeros(1, device=DEV))
        r["global_qt"] = summary(site_eval(taus, sites, seen_t)[0])
        r["global_unknown"] = unknown_metrics(sites, nov, yy, th0)
        sites, pad, nk, W, nov, yy = stream_run(base, X, y, glob, "ditto", rate, lam_star[rate], seed, score_fn)
        G = len(FRONTIER) + 1
        starts = tau_fr + [tau0]
        etas = [0.0] * len(FRONTIER) + [eta]
        taus = torch.cat([track2(pad, nk, W[None], torch.full((len(sites),), starts[g], device=DEV),
                                 torch.tensor([etas[g]], device=DEV), torch.zeros(1, device=DEV)) for g in range(G)])
        ev = [summary(x) for x in site_eval(taus, sites, seen_t)]
        r["ditto_frontier"] = {str(a): e for a, e in zip(FRONTIER, ev[:-1])}
        r["ours"] = ev[-1]
        r["ditto_fixed"] = ev[0]
        r["frontier_sig_at_ours_size"] = interp(ev[:-1], ev[-1]["size"], "sig")
        r["ours_unknown"] = unknown_metrics(sites, nov, yy, th0)
        out[f"r={rate}"] = r
        print(f"  seed {seed} r={rate}: ours %sig {r['ours']['sig']:.0f} size {r['ours']['size']:.2f} | ditto fixed "
              f"{r['ditto_fixed']['sig']:.0f} | global qt {r['global_qt']['sig']:.0f} | fcp {out['fcp']['sig']:.0f}", flush=True)
    return out


def report(allr, which):
    print(f"\n== FINAL EVALUATION on {which.upper()} sites ({allr[0]['n_sites']} sites, {allr[0]['n_unseen']} unseen), "
          f"features={TAG}, RAPS, alpha={ALPHA}, alpha_u={ALPHA_U}, delay={DELAY}, seeds {list(SEEDS)}")
    print(f"{'method':<34}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'size':>7}   per-seed %sig")
    def line(name, getter):
        v = [getter(r) for r in allr]
        m = {k: np.nanmean([x[k] for x in v]) for k in ("cov", "sig", "sig_unseen", "size")}
        print(f"{name:<34}{m['cov']:7.3f}{m['sig']:6.0f}%{m['sig_unseen']:11.0f}%{m['size']:7.2f}   "
              f"{[round(x['sig']) for x in v]}")
    for m in ("fcp", "labelshift", "qq", "dr"):
        line(f"{m} (static, global)", lambda r, m=m: r[m])
    for rate in RATES:
        k = f"r={rate}"
        line(f"global + tracking   {k}", lambda r: r[k]["global_qt"])
        line(f"Ditto fixed tau     {k}", lambda r: r[k]["ditto_fixed"])
        line(f"OURS Ditto+tracking {k}", lambda r: r[k]["ours"])
        d = [r[k]["ours"]["sig"] - r[k]["frontier_sig_at_ours_size"] for r in allr]
        d1 = [r[k]["ours"]["sig"] - r["fcp"]["sig"] for r in allr]
        print(f"   ours - FCP: {np.mean(d1):+.1f} pts (per seed {np.round(d1, 1).tolist()}); ours - Ditto frontier at "
              f"matched size: {np.mean(d):+.1f} pts (per seed {np.round(d, 1).tolist()})")
        for name in ("global_unknown", "ours_unknown"):
            u = {q: np.nanmean([r[k][name][q] for r in allr]) for q in ("recall", "usig", "ff", "auroc")}
            print(f"   unknown flag {name.split('_')[0]:<7}: recall {u['recall']:.3f}  %sites<0.8 {u['usig']:.0f}%  "
                  f"false flags {u['ff']:.1f}%  AUROC {u['auroc']:.3f}")


if __name__ == "__main__":
    which = sys.argv[1]
    assert which in ("dev", "lockbox")
    if which == "lockbox":
        assert os.path.exists(FREEZE), "pre-registration not frozen"
        frozen = json.load(open(FREEZE))["sha256"]
        bad = [p for p in HASHED if frozen.get(p) != sha(p)]
        assert not bad, f"files changed since freeze: {bad}"
        assert not os.path.exists(DONE), "lockbox already evaluated; it is run once"
        json.dump(dict(started=str(np.datetime64("now")), freeze=frozen), open(DONE, "w"), indent=1)
    sel = json.load(open(f"{ROOT}/results/pers_lambda_{TAG}.json"))
    allr = []
    for sd in SEEDS:
        lam_star = {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in RATES}
        allr.append(run(sd, which, lam_star))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{ROOT}/results/final_eval_{which}.json", "w"), indent=1, default=float)
    report(allr, which)
    print(f"FINAL_EVAL_DONE {which}")
