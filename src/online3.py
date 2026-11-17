"""Reformulated federation: federated empirical-Bayes site offsets (eb.py) as the static threshold
and as the start of per-site online tracking. Validated time splits, RAPS, any feature file.

Compared (all calibrated on cal; RAPS params and eta chosen on tune only):
  static frontiers at alpha' in FRONTIER: pooled, local, eb
  online per-site tracking (qt_site) started at: pooled, eb   -- eta chosen on tune replay
Usage: python online3.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import eb
import scores
from online import DELAY, RATES
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
FRONTIER = (0.1, 0.07, 0.05, 0.03, 0.02)
ETAS = [e for e in ETA_L if e > 0]


def run(seed, tag):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    Xall = np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=seed)}.npy")
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    head = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P = probs_of(head, X)
    tm = part == "tune"
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    S_all = scores.raps(P, lam, kreg, seed + 7)
    cm = part == "cal"
    start = s.groupby("seq_id").datetime.transform("min").to_numpy().astype("datetime64[s]").astype(np.int64)
    cal = dict(S=S_all[cm], y=y[cm], client=client[cm], seq=seqs[cm], time=start[cm])

    test_sites = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_sites) if st is not None]
    test_sites = [test_sites[k] for k in keep]
    seen_t = seen[keep]
    pad, nk = pack(test_sites)
    J = pad.shape[1]
    out = {"acc": float((P[part == "test"].argmax(1) == y[part == "test"])[y[part == "test"] != cp.UNK].mean())}

    def const(taus_k):  # (G, K_all) -> (G, K_test, J)
        t = torch.as_tensor(np.asarray(taus_k)[:, keep], dtype=torch.float32, device=DEV)
        return t[:, :, None].expand(-1, -1, J).contiguous()

    fr = {"pooled": [], "local": [], "eb": [], "persist": []}
    eb_info, ps_info = {}, {}
    for a in FRONTIER:
        fr["pooled"].append(cp.thresholds("pooled", cal, a, C, K))
        loc = cp.thresholds("local", cal, a, C, K)
        fr["local"].append(np.minimum(loc, S_all.max()))  # +inf (tiny cal) -> full set
        t_eb, eb_info[a] = eb.eb_thresholds(cal, a, K, seed)
        fr["eb"].append(t_eb)
        t_ps, ps_info[a] = eb.persistence_thresholds(cal, a, K)
        fr["persist"].append(t_ps)
    for name, taus in fr.items():
        ev = site_eval(const(taus), test_sites, seen_t)
        out[f"static_{name}"] = {a: summary(e) for a, e in zip(FRONTIER, ev)}
    out["eb_info"] = dict(eb=eb_info[ALPHA], persist=ps_info[ALPHA])

    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    el = torch.tensor(ETAS, device=DEV)
    eg = torch.zeros(len(ETAS), device=DEV)
    inits = {"pooled": np.full(K, fr["pooled"][0][0]), "eb": fr["eb"][0], "persist": fr["persist"][0]}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(ETAS), -1, -1),
                    torch.full((len(tune_sites),), tau_tune, device=DEV), el, eg)
        choice = int(np.argmin(pinball(tt, tune_sites)))
        lab_s = labels_for(test_sites, rate, 10_000 + seed, J)
        for iname, t0 in inits.items():
            ts = track2(pad, nk, lab_s[None].expand(len(ETAS), -1, -1),
                        torch.as_tensor(t0[keep], dtype=torch.float32, device=DEV), el, eg)
            ev = site_eval(ts, test_sites, seen_t)
            out[f"qt_from_{iname}|r={rate}"] = dict(chosen=ETAS[choice], selected=summary(ev[choice]),
                                                    grid={str(e): summary(ev[g]) for g, e in enumerate(ETAS)})
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = []
    for sd in range(n):
        allr.append(run(sd, tag))
        print(f"seed {sd} done  eb: {allr[-1]['eb_info']}", flush=True)
    json.dump(allr, open(f"{ROOT}/results/online3_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== online v3 (EB offsets), features={tag}, RAPS, time split, delay={DELAY}, {n} seeds; "
          f"acc={np.mean([r['acc'] for r in allr]):.3f}")
    print(f"{'method':<28}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'%win<.85':>9}{'size':>7}")

    def line(name, d):
        print(f"{name:<28}{d['cov']:7.3f}{d['sig']:6.0f}%{d['sig_unseen']:11.0f}%{d['win']:8.0f}%{d['size']:7.2f}")

    for name in ["pooled", "persist", "eb", "local"]:
        for a in FRONTIER:
            line(f"static_{name} a={a}", {k: np.mean([r[f"static_{name}"][a][k] for r in allr])
                                         for k in allr[0][f"static_{name}"][a]})
    for key in [k for k in allr[0] if k.startswith("qt_from_")]:
        line(f"{key} eta*", {k: np.mean([r[key]["selected"][k] for r in allr]) for k in allr[0][key]["selected"]})
