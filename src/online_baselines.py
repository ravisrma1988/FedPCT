"""Stronger single-stream online conformal trackers as per-site comparators, DEVELOPMENT SITES ONLY.

Every tracker works in score space on the same padded streams as FedPCT (online_gpu.pack), receives
the same delayed (DELAY sequences) and partial (labelling draws W) feedback, and judges a sequence
against the threshold that was used when it was predicted. Hyper-parameters are chosen per method and
labelling rate on the tune replay by mean pinball loss, exactly as FedPCT's step size.

  qt      FedPCT's tracker (adaptive conformal inference in score space), step eta
  decay   decaying step size eta_n = eta0 * n^(-1/2 - 0.1), n = labelled sequences received so far
          (Angelopoulos, Barber and Bates, 2024)
  dtaci   dynamically tuned ACI (Gibbs and Candes, 2024): E experts with steps in ETA_EXP, mixed by
          exponential weights on their pinball losses with learning rate eta_w and mixing sigma;
          the output is the weighted mean threshold
  pid     quantile tracking plus the saturating integrator of conformal PID control (Angelopoulos,
          Candes and Tibshirani, 2023), r_t(x) = K_I tan(x log t / (t C_sat)); no scorecaster

Usage: python online_baselines.py iwildcam|cct|fmow      (development sites only)
The FMoW lockbox run calls run(..., which="lockbox") from lockbox_eval_fmow.py, under its freeze guard.
"""
import itertools
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "${FEDPCT_ROOT}/src")
DS = sys.argv[1] if __name__ == "__main__" else None
if DS == "cct":
    import cct_env
    cct_env.use_cct()
elif DS == "fmow":
    import fmow_env
    fmow_env.use_fmow()
import lockbox_eval as le  # noqa: E402

if DS == "cct":
    le.TAG = cct_env.TAG
elif DS == "fmow":
    le.TAG = fmow_env.TAG

import cp_baselines as cp  # noqa: E402
import scores  # noqa: E402
from lockbox import dev_mask  # noqa: E402
from online import DELAY  # noqa: E402
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2  # noqa: E402
from online_gpu import pack  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

ROOT = le.ROOT
POS = [e for e in ETA_L if e > 0]
ETA_EXP = (0.001, 0.005, 0.02, 0.05, 0.2)
GRID = {
    "qt": [dict(eta=e) for e in POS],
    "decay": [dict(eta0=e) for e in (0.02, 0.05, 0.2, 0.5, 2.0)],
    "dtaci": [dict(eta_w=w, sigma=s) for w, s in itertools.product((0.01, 0.1, 1.0), (0.01, 0.1))],
    "pid": [dict(eta=e, ki=ki, csat=cs) for e, ki, cs in itertools.product(POS, (0.1, 0.5, 2.0), (1.0, 5.0))],
}


def _excess_and_loss(pad, nk, lab_m, taus_m, alpha):
    """pad_m (K,I), taus_m (..., K): labelled excess misses and pinball loss of one sequence step."""
    sc = pad[None]                                                      # (1,K,I)
    t = taus_m[..., None]                                               # (G,K,1)
    miss = (sc > t).sum(-1).float()                                     # nan > t is False
    exc = lab_m * (miss - alpha * nk)
    diff = sc - t
    pin = torch.where(diff >= 0, (1 - alpha) * diff, -alpha * diff)
    loss = lab_m * torch.nan_to_num(pin, nan=0.0).sum(-1)
    return exc, loss


def track_any(method, cfgs, pad, nk, W, tau0, alpha=ALPHA, delay=DELAY):
    """pad (K,J,I), nk (K,J), W (K,J) labelling flags, tau0 (K,). Returns taus (G,K,J), G = len(cfgs)."""
    K, J, _ = pad.shape
    G = len(cfgs)
    taus = torch.empty(G, K, J, device=DEV)
    col = lambda key: torch.tensor([c[key] for c in cfgs], device=DEV)[:, None]  # noqa: E731
    if method in ("qt", "decay", "pid"):
        cum = torch.zeros(G, K, device=DEV)        # labelled excess received so far
        off = torch.zeros(G, K, device=DEV)        # tau = tau0 + off, the same arithmetic as track2
        zero = torch.zeros(G, 1, device=DEV)
        nlab = torch.zeros(G, K, device=DEV)       # labelled sequences received
        nimg = torch.zeros(G, K, device=DEV)       # labelled images received
        for j in range(J):
            taus[:, :, j] = tau0[None, :] + zero + off
            m = j - delay
            if m < 0:
                continue
            exc, _ = _excess_and_loss(pad[:, m], nk[:, m], W[None, :, m], taus[:, :, m], alpha)
            if method == "qt":
                off = off + col("eta") * exc
            elif method == "decay":
                nlab = nlab + W[None, :, m]
                off = off + col("eta0") * nlab.clamp(min=1) ** (-0.6) * exc
            else:
                cum = cum + exc
                nimg = nimg + W[None, :, m] * nk[None, :, m]
                t = nimg + 1
                arg = (cum * torch.log(t) / (t * col("csat"))).clamp(-1.5, 1.5)
                off = col("eta") * cum + col("ki") * torch.tan(arg)
        return taus
    # dtaci
    E = len(ETA_EXP)
    gam = torch.tensor(ETA_EXP, device=DEV)[None, None, :]
    th = tau0[None, :, None].expand(G, K, E).clone()
    hist = torch.empty(G, K, E, J, device=DEV)
    w = torch.full((G, K, E), 1.0 / E, device=DEV)
    eta_w, sig = col("eta_w")[..., None], col("sigma")[..., None]
    for j in range(J):
        hist[..., j] = th
        taus[:, :, j] = (w * th).sum(-1)
        m = j - delay
        if m < 0:
            continue
        lab_m = W[None, :, m, None]
        sc = pad[None, :, m, None, :]                                     # (1,K,1,I)
        t = hist[..., m][..., None]                                       # (G,K,E,1)
        miss = (sc > t).sum(-1).float()
        exc = lab_m * (miss - alpha * nk[None, :, m, None])
        diff = sc - t
        loss = lab_m * torch.nan_to_num(torch.where(diff >= 0, (1 - alpha) * diff, -alpha * diff), nan=0.0).sum(-1)
        th = th + gam * exc
        w = w * torch.exp(-eta_w * (loss - loss.min(-1, keepdim=True).values))
        w = w / w.sum(-1, keepdim=True)
        w = (1 - sig) * w + sig / E
    return taus


def choose(method, tune_sites, tpad, tnk, Wt, tau_tune):
    cfgs = GRID[method]
    tt = track_any(method, cfgs, tpad, tnk, Wt, torch.full((len(tune_sites),), tau_tune, device=DEV))
    return cfgs[int(np.argmin(pinball(tt, tune_sites)))]


def run(seed, lam_star, which="dev"):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    mask = dev_mask(info) if which == "dev" else ~dev_mask(info)
    X = torch.from_numpy(np.load(f"{ROOT}/features/iwildcam_{le.TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm, cm = part == "tune", part == "cal"
    lam_r, kreg = scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda q: scores.raps(q, lam_r, kreg, seed + 7)  # noqa: E731
    S_all = score_fn(P_all)
    tau0 = cp.thresholds("pooled", dict(S=S_all[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]
    tune_sites = [st for st in build_streams(s, "tune", S_all, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_all[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    test_all = build_streams(s, "test", S_all, y, seqs)
    keep = [k for k, st in enumerate(test_all) if st is not None and mask[k]]
    base, seen_t = [test_all[k] for k in keep], seen[keep]
    out = dict(n_sites=len(keep), n_unseen=int((~seen_t).sum()))
    for rate in le.RATES:
        Wt = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        chosen = {m: choose(m, tune_sites, tpad, tnk, Wt, tau_tune) for m in GRID}
        r = dict(chosen=chosen)
        for head in ("global", "ditto"):
            sites, pad, nk, W, _, _ = le.stream_run(base, X, y, glob, head, rate,
                                                    lam_star[rate] if head == "ditto" else None, seed, score_fn)
            t0 = torch.full((len(sites),), tau0, device=DEV)
            ref = track2(pad, nk, W[None], t0, torch.tensor([chosen["qt"]["eta"]], device=DEV), torch.zeros(1, device=DEV))
            got = track_any("qt", [chosen["qt"]], pad, nk, W, t0)
            assert torch.allclose(ref, got, atol=1e-4), "qt tracker does not reproduce FedPCT's tracker"
            for m in GRID:
                taus = got if m == "qt" else track_any(m, [chosen[m]], pad, nk, W, t0)
                r[f"{head}_{m}"] = summary(site_eval(taus, sites, seen_t)[0])
        out[f"r={rate}"] = r
        print(f"  seed {seed} r={rate}: " + "  ".join(f"{k} {v['sig']:.0f}%/{v['size']:.2f}"
                                                     for k, v in r.items() if k != "chosen"), flush=True)
    return out


def report(allr, title):
    print(f"\n== online trackers, {title} ({allr[0]['n_sites']} sites), 5 seeds")
    print(f"{'head_method':<16}{'rate':>5}{'%sig':>8}{'unseen':>8}{'size':>7}{'cov':>7}   per-seed %sig")
    for rate in le.RATES:
        for key in [k for k in allr[0][f"r={rate}"] if k != "chosen"]:
            g = lambda q: np.nanmean([a[f"r={rate}"][key][q] for a in allr])  # noqa: E731
            print(f"{key:<16}{rate:>5}{g('sig'):7.1f}%{g('sig_unseen'):7.0f}%{g('size'):7.2f}{g('cov'):7.3f}   "
                  f"{[round(a[f'r={rate}'][key]['sig']) for a in allr]}")


if __name__ == "__main__" and DS in ("iwildcam", "cct", "fmow"):
    sel = json.load(open(f"{ROOT}/results/pers_lambda_{le.TAG}.json"))
    allr = []
    for sd in le.SEEDS:
        allr.append(run(sd, {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in le.RATES}))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{ROOT}/results/online_baselines_dev_{DS}.json", "w"), indent=1, default=float)
    report(allr, f"{DS} DEVELOPMENT sites")
    print("ONLINE_BASELINES_DONE")
