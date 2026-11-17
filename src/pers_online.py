"""Federated online personalisation + online conformal tracking (development sites only).

Each site starts from the federated (FedAvg) head. Delayed labelled feedback (rate r, delay DELAY
sequences) accumulates; every REFIT_EVERY new labelled known-class images the site refits its head
on ALL feedback received so far:
  global     no personalisation
  prior_fix  global logits + log(pi_fb / pi_train)  (species-mix correction from feedback labels)
  ditto      head fine-tuned on feedback with an L2 pull to the global head (Li et al., ICML 2021)
Scores: RAPS (lam, kreg chosen on tune with the global head). Thresholds: start at the pooled cal
quantile (global head) and are either frozen (eta=0) or tracked per site (eta chosen on the tune
replay with the global head, online2's rule; full eta grid also reported). Refit times depend only
on the labelling schedule, so each site's score stream is computed in segments, then the verified
GPU tracker runs over all sites x etas. ACI's long-run guarantee holds for any model sequence.
Hyper-parameters fixed a priori: REFIT_EVERY, DITTO_*.
Usage: python pers_online.py <n_seeds> <feature tag>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from lockbox import dev_mask
from online import DELAY, RATES
from online2 import ETA_L, build_streams, labels_for, pinball, site_eval, summary, track2
from online_gpu import pack
from run_iwildcam import ALPHA, DEV, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
REFIT_EVERY = 50
DITTO_STEPS, DITTO_LR, DITTO_LAMBDA = 100, 0.1, 1.0
ETAS = [0.0] + [e for e in ETA_L if e > 0]
MODES = ("global", "prior_fix", "ditto", "finetune_l0", "local_only")
# ablations: finetune_l0 = start from the global head, no pull (lambda = 0);
#            local_only  = zero-initialised head trained on the site's feedback only (no federation)


def ditto(glob, X, y, lam=DITTO_LAMBDA, from_global=True):
    h = torch.nn.Linear(glob.in_features, glob.out_features).to(DEV)
    if from_global:
        h.load_state_dict(glob.state_dict())
    else:
        torch.nn.init.zeros_(h.weight), torch.nn.init.zeros_(h.bias)
    w0 = [p.detach().clone() for p in glob.parameters()]
    opt = torch.optim.SGD(h.parameters(), lr=DITTO_LR)
    for _ in range(DITTO_STEPS):
        loss = torch.nn.functional.cross_entropy(h(X * 10), y)
        if lam > 0:
            loss = loss + lam / 2 * sum(((p - q) ** 2).sum() for p, q in zip(h.parameters(), w0))
        opt.zero_grad(); loss.backward(); opt.step()
    return h


def personalised_stream(st, Xs, ys, lab_seq, glob, pi_tr, mode, score_fn):
    """Scores of one site's stream when the head is refitted on arrived feedback.
    st: site stream dict (idx, seq_codes); Xs/ys: features/labels of the stream (stream order);
    lab_seq: (J,) labelled flag per sequence. Returns S (n, C) under the model in force."""
    codes = st["seq_codes"]
    J = codes.max() + 1
    known = ys != cp.UNK
    # feedback of sequence m arrives before sequence m + DELAY + 1
    n_lab = np.array([(known & (codes == m)).sum() if lab_seq[m] else 0 for m in range(J)])
    arrived = np.zeros(J + DELAY + 1, dtype=int)
    arrived[DELAY + 1:DELAY + 1 + J] = np.cumsum(n_lab)
    arrived = np.maximum.accumulate(arrived)[:J]             # labelled images available at step j
    refit_at = [j for j in range(J) if j > 0 and arrived[j] // REFIT_EVERY > arrived[j - 1] // REFIT_EVERY]
    bounds = [0] + refit_at + [J]
    S = np.empty((len(ys), glob.out_features), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = np.flatnonzero((codes >= a) & (codes < b))
        if len(rows) == 0:
            continue
        fb = np.flatnonzero(known & np.isin(codes, np.flatnonzero(lab_seq[:max(a - DELAY, 0)])))
        with torch.no_grad():
            if mode == "global" or len(fb) == 0:
                P = torch.softmax(glob(Xs[rows] * 10), 1)
            elif mode == "prior_fix":
                C = glob.out_features
                yfb = torch.as_tensor(ys[fb], device=DEV)
                pi = (torch.bincount(yfb, minlength=C).float() + 1) / (len(fb) + C)
                P = torch.softmax(glob(Xs[rows] * 10) + torch.log(pi / pi_tr), 1)
        if mode in ("ditto", "finetune_l0", "local_only") and len(fb) > 0:
            h = ditto(glob, Xs[fb], torch.as_tensor(ys[fb], device=DEV),
                      lam=0.0 if mode != "ditto" else DITTO_LAMBDA, from_global=mode != "local_only")
            with torch.no_grad():
                P = torch.softmax(h(Xs[rows] * 10), 1)
        S[rows] = score_fn(P.cpu().numpy())
    return S


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
    pi_tr = torch.as_tensor((np.bincount(y[tr], minlength=C) + 1.0) / (tr.sum() + C), device=DEV, dtype=torch.float32)
    P = probs_of(glob, X)
    tm = part == "tune"
    lam, kreg = scores.tune_raps(P[tm], y[tm], seqs[tm], ALPHA, seed)
    score_fn = lambda p: scores.raps(p, lam, kreg, seed + 7)  # noqa: E731
    S_glob = score_fn(P)
    cm = part == "cal"
    tau0 = cp.thresholds("pooled", dict(S=S_glob[cm], y=y[cm], client=client[cm]), ALPHA, C, K)[0]

    # eta choice on the tune replay (global head), as in online2
    tune_sites = [st for st in build_streams(s, "tune", S_glob, y, seqs) if st is not None]
    tpad, tnk = pack(tune_sites)
    tau_tune = cp.thresholds("pooled", dict(S=S_glob[tm], y=y[tm], client=client[tm]), ALPHA, C, K)[0]
    pos = [e for e in ETAS if e > 0]

    sites_all = build_streams(s, "test", S_glob, y, seqs)
    keep = [k for k, st in enumerate(sites_all) if st is not None and dev[k]]
    base = [sites_all[k] for k in keep]
    seen_t = seen[keep]
    J = max(st["seq_codes"].max() + 1 for st in base)
    el = torch.tensor(ETAS, device=DEV)
    eg = torch.zeros(len(ETAS), device=DEV)
    out = {}
    for rate in RATES:
        lab_t = labels_for(tune_sites, rate, 20_000 + seed, tpad.shape[1])
        tt = track2(tpad, tnk, lab_t[None].expand(len(pos), -1, -1),
                    torch.full((len(tune_sites),), tau_tune, device=DEV),
                    torch.tensor(pos, device=DEV), torch.zeros(len(pos), device=DEV))
        eta_star = pos[int(np.argmin(pinball(tt, tune_sites)))]
        lab = labels_for(base, rate, 10_000 + seed, J).cpu().numpy() > 0.5
        for mode in MODES:
            sites = []
            for st, L in zip(base, lab):
                ys = y[st["idx"]]
                S = personalised_stream(st, X[st["idx"]], ys, L, glob, pi_tr, mode, score_fn)
                sites.append(dict(st, S=S, score=cp.true_scores(S, ys)))
            pad, nk = pack(sites)
            taus = track2(pad, nk, torch.as_tensor(lab, dtype=torch.float32, device=DEV)[None].expand(len(ETAS), -1, -1),
                          torch.full((len(sites),), tau0, device=DEV), el, eg)
            ev = site_eval(taus, sites, seen_t)
            g_star = ETAS.index(eta_star)
            out[f"{mode}|r={rate}"] = dict(eta_star=eta_star, frozen=summary(ev[0]), tracked=summary(ev[g_star]),
                                           grid={str(e): summary(ev[g]) for g, e in enumerate(ETAS)})
        print(f"  seed {seed} r={rate} done (eta*={eta_star})", flush=True)
    # static pooled frontier with the global head, same dev sites
    fr = {}
    for a in (0.1, 0.07, 0.05, 0.03, 0.02):
        t = cp.thresholds("pooled", dict(S=S_glob[cm], y=y[cm], client=client[cm]), a, C, K)[0]
        fr[a] = summary(site_eval(torch.full((1, len(base), J), t, device=DEV), base, seen_t)[0])
    out["static_global"] = fr
    return out


if __name__ == "__main__":
    n, tag = int(sys.argv[1]), sys.argv[2]
    allr = []
    for sd in range(n):
        allr.append(run(sd, tag))
        print(f"seed {sd} done", flush=True)
    json.dump(allr, open(f"{ROOT}/results/pers_online_{tag.replace('{seed}', 'S')}.json", "w"), indent=1, default=float)
    print(f"\n== online personalisation + conformal, DEV sites, features={tag}, RAPS, delay={DELAY}, {n} seeds")
    print(f"{'method':<34}{'cov':>7}{'%sig':>7}{'%sig_unseen':>12}{'%win<.85':>9}{'size':>7}")

    def line(name, d):
        print(f"{name:<34}{d['cov']:7.3f}{d['sig']:6.0f}%{d['sig_unseen']:11.0f}%{d['win']:8.0f}%{d['size']:7.2f}")

    for a in allr[0]["static_global"]:
        line(f"static_global a={a}", {k: np.mean([r["static_global"][a][k] for r in allr]) for k in allr[0]["static_global"][a]})
    for key in [k for k in allr[0] if "|r=" in k]:
        for v in ["frozen", "tracked"]:
            line(f"{key} {v}", {k: np.mean([r[key][v][k] for r in allr]) for k in allr[0][key][v]})
