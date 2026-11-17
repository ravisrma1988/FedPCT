"""Post-lockbox ablations and sensitivity analyses. DEVELOPMENT sites only (the lockbox is spent).
Frozen files are not edited: settings are swapped in at run time (module globals of pers_select /
online2), so the code path is otherwise identical to lockbox_eval.py.

Configs (each vs the base config; lambda*(r, seed) from results/pers_lambda_bioclip2crop.json, chosen
under the base config, except where the config defines lambda):
  base            alpha 0.1, delay 5, refit every 50, Ditto, FedAvg global head (must match final_eval_dev)
  finetune_l0     personalised head starts at global, no proximal pull (lambda = 0)
  local_only      zero-initialised head on the site's own feedback only (no federation)
  central_head    global head trained centrally (non-federated upper bound), then Ditto
  alpha=0.05/0.2  target miss rate (frontier grid scaled with alpha)
  delay=0/2/10    feedback delay in sequences
  refit=20/100    Ditto refit period (labelled known images)
Per config, rates 0.3 and 1.0: FCP static, global + tracking, personalised fixed tau, method
(personalised + tracking), method minus personalised frontier at matched size; unknown flag at
alpha_u in {0.1, 0.2, 0.3} for the method's head.
Usage: python sens_ablate.py <n_seeds> [config ...]
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import online2
import pers_select
import scores
from lockbox import dev_mask
from lockbox_eval import TAG, stream_run, unknown_metrics
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from pers_frontier import interp
from pers_online import ditto
from run_iwildcam import DEV, central, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
RATES = (0.3, 1.0)
REL_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.3)  # frontier alphas = alpha * REL_GRID
AU = (0.1, 0.2, 0.3)
CONFIGS = {
    "base": {}, "finetune_l0": dict(mode="finetune_l0"), "local_only": dict(mode="local_only"),
    "central_head": dict(head="central"), "alpha=0.05": dict(alpha=0.05), "alpha=0.2": dict(alpha=0.2),
    "delay=0": dict(delay=0), "delay=2": dict(delay=2), "delay=10": dict(delay=10),
    "refit=20": dict(refit=20), "refit=100": dict(refit=100),
}


def configure(cfg):
    pers_select.DELAY = online2.DELAY = cfg.get("delay", 5)
    pers_select.REFIT_EVERY = cfg.get("refit", 50)
    online2.ALPHA = cfg.get("alpha", 0.1)                     # used by pinball() and site_eval()
    mode = cfg.get("mode", "ditto")
    if mode == "finetune_l0":
        pers_select.ditto = lambda g, X, y, lam: ditto(g, X, y, 0.0, True)
    elif mode == "local_only":
        pers_select.ditto = lambda g, X, y, lam: ditto(g, X, y, 0.0, False)
    else:
        pers_select.ditto = ditto


def run(seed, cfg, lam_sel):
    configure(cfg)
    a = cfg.get("alpha", 0.1)
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    dev = dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    ytr = torch.from_numpy(y[tr]).to(DEV)
    glob = central(X[tr], ytr, C, seed) if cfg.get("head") == "central" else fedavg(X[tr], ytr, client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], a, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    cal = dict(S=S_all[cm], y=y[cm], client=client[cm])
    fr_alphas = [a * g for g in REL_GRID]
    tau_fr = [cp.thresholds("pooled", cal, x, C, K)[0] for x in fr_alphas]
    tau0 = tau_fr[0]
    nu_neg = {au: cp.conformal_quantile(-(1 - P_all.max(1))[cm & (y == cp.UNK)], au) for au in AU}
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), a, C, K)[0]
    pos = [e for e in ETA_L if e > 0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and dev[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    one = torch.zeros(1, device=DEV)
    out = {}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1), torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV), alpha=a)
        eta = pos[int(np.argmin(pinball(tt, tune_sites)))]
        r = dict(eta=eta)
        sites, pad, nk, W, _, _ = stream_run(base, X, y, glob, "global", rate, None, seed, score_fn)
        taus = torch.cat([track2(pad, nk, W[None], torch.full((len(sites),), tau0, device=DEV), e, one, alpha=a)
                          for e in (torch.zeros(1, device=DEV), torch.tensor([eta], device=DEV))])
        ev = [summary(x) for x in site_eval(taus, sites, seen_t)]
        r["fcp"], r["global_qt"] = ev
        sites, pad, nk, W, nov, yy = stream_run(base, X, y, glob, "ditto", rate, lam_sel[rate], seed, score_fn)
        starts, etas = tau_fr + [tau0], [0.0] * len(tau_fr) + [eta]
        taus = torch.cat([track2(pad, nk, W[None], torch.full((len(sites),), starts[g], device=DEV),
                                 torch.tensor([etas[g]], device=DEV), one, alpha=a) for g in range(len(starts))])
        ev = [summary(x) for x in site_eval(taus, sites, seen_t)]
        r["pers_fixed"], r["ours"] = ev[0], ev[-1]
        r["front_sig_at_ours"] = interp(ev[:-1], ev[-1]["size"], "sig")
        for au in AU:
            r[f"unk_{au}"] = unknown_metrics(sites, nov, yy, nu_neg[au])
        out[f"r={rate}"] = r
    return out


if __name__ == "__main__":
    n = int(sys.argv[1])
    names = sys.argv[2:] or list(CONFIGS)
    sel = json.load(open(f"{ROOT}/results/pers_lambda_{TAG}.json"))
    for name in names:
        cfg = CONFIGS[name]
        allr = []
        for sd in range(n):
            lam_sel = {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in RATES}
            allr.append(run(sd, cfg, lam_sel))
        json.dump(allr, open(f"{ROOT}/results/sens_{name}.json", "w"), indent=1, default=float)
        for rate in RATES:
            k = f"r={rate}"
            m = lambda key, q: np.nanmean([r[k][key][q] for r in allr])  # noqa: E731
            d = [r[k]["ours"]["sig"] - r[k]["front_sig_at_ours"] for r in allr]
            u = {au: (m(f"unk_{au}", "recall"), m(f"unk_{au}", "ff")) for au in AU}
            print(f"{name:<13} {k:<6} FCP {m('fcp', 'sig'):3.0f}%/{m('fcp', 'size'):5.2f} | global+qt "
                  f"{m('global_qt', 'sig'):3.0f}%/{m('global_qt', 'size'):5.2f} | pers fixed {m('pers_fixed', 'sig'):3.0f}%/"
                  f"{m('pers_fixed', 'size'):5.2f} | OURS {m('ours', 'sig'):3.0f}%/{m('ours', 'size'):5.2f} "
                  f"(unseen {m('ours', 'sig_unseen'):3.0f}%) | ours-frontier {np.mean(d):+.1f} | unk recall/ff "
                  + " ".join(f"a_u={au}:{u[au][0]:.2f}/{u[au][1]:.0f}%" for au in AU), flush=True)
    print("SENS_DONE")
