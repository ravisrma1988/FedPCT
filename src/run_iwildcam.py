"""Protocol v2 baselines on iWildCam BioCLIP features, using ONLY the validated splits
from splits.py (no partitioning happens here).

Per seed and split mode:
  * head: linear layer on frozen features, trained with FedAvg over the train partition of seen
    clients (a centrally trained head is reported alongside as a reference)
  * calibration: tune + cal partitions (baselines have no hyper-parameters, so they use both)
  * test: test partition; per-client significance via sequence-cluster bootstrap
Usage:  python run_iwildcam.py <n_seeds> <time|seq>
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import binom

import cp_baselines as cp
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
ALPHA = 0.1
ROUNDS, PER_ROUND, LOCAL_EPOCHS, LR, BS = 150, 20, 2, 0.5, 64
DEV = "cuda"


def make_head(d, C, init=None):
    """Linear head on x*10. init=None: zeros. init=(C, d) unit-norm text embeddings: W = 10*T,
    so initial logits equal BioCLIP zero-shot logits (logit scale 100 x cosine)."""
    h = torch.nn.Linear(d, C).to(DEV)
    torch.nn.init.zeros_(h.bias)
    with torch.no_grad():
        h.weight.copy_(torch.zeros(C, d) if init is None else 10 * torch.as_tensor(init, dtype=torch.float32))
    return h


def local_sgd(head, X, y, epochs, gen):
    opt = torch.optim.SGD(head.parameters(), lr=LR)
    for _ in range(epochs):
        for idx in torch.randperm(len(y), generator=gen, device="cpu").split(BS):
            loss = torch.nn.functional.cross_entropy(head(X[idx] * 10), y[idx])  # x10: features are unit-norm
            opt.zero_grad(); loss.backward(); opt.step()


def fedavg(X, y, client, C, seed, init=None):
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    clients = np.unique(client)
    data = {k: (X[client == k], y[client == k]) for k in clients}
    glob = make_head(X.shape[1], C, init)
    for _ in range(ROUNDS):
        sel = rng.choice(clients, min(PER_ROUND, len(clients)), replace=False)
        states, ws = [], []
        for k in sel:
            h = make_head(X.shape[1], C); h.load_state_dict(glob.state_dict())
            local_sgd(h, *data[k], LOCAL_EPOCHS, gen)
            states.append(h.state_dict()); ws.append(len(data[k][1]))
        ws = torch.tensor(ws, dtype=torch.float32, device=DEV) / sum(ws)
        glob.load_state_dict({n: sum(w * s[n] for w, s in zip(ws, states)) for n in states[0]})
    return glob


def central(X, y, C, seed, epochs=10, init=None):
    gen = torch.Generator().manual_seed(seed)
    h = make_head(X.shape[1], C, init)
    local_sgd(h, X, y, epochs, gen)
    return h


@torch.no_grad()
def probs_of(head, X):
    return torch.softmax(head(X * 10), 1).cpu().numpy()


def calibrate_and_eval(cal, te, C, K, seen):
    res, per = {}, {}

    def finish(name, sets):
        rows = cp.evaluate(sets, te["y"], te["client"], K)
        clus = cp.sig_below_clustered(sets, te["y"], te["client"], te["seq"], K, ALPHA)
        for r in rows:
            nk = round(r["n"] * (1 - r["unk_frac"]))
            r["seen"] = bool(seen[r["client"]])
            # image-level binomial test kept only for comparison; it ignores sequence clustering
            r["sig_below_binom"] = bool(binom.cdf(round(r["cov_known"] * nk), nk, 1 - ALPHA) < 0.05)
            r["sig_below"] = bool(clus.get(r["client"], False))
        cov = np.array([r["cov_known"] for r in rows])
        s = cp.summarize(rows, ALPHA)
        s.update(p10_cov_known=float(np.quantile(cov, 0.1)),
                 pct_sig_below=100 * float(np.mean([r["sig_below"] for r in rows])),
                 pct_sig_below_unseen=100 * float(np.mean([r["sig_below"] for r in rows if not r["seen"]])),
                 pct_sig_below_binom=100 * float(np.mean([r["sig_below_binom"] for r in rows])))
        rec = [r["unk_recall"] for r in rows if not np.isnan(r["unk_recall"])]
        s["unk_recall"] = float(np.mean(rec)) if rec else float("nan")
        res[name], per[name] = s, rows

    for m in ["local", "pooled", "labelshift"]:
        tau = cp.thresholds(m, cal, ALPHA, C, K)
        finish(m, cp.predict_sets(te["S"], te["client"], tau))
    tau = cp.thresholds("pooled", cal, ALPHA, C, K)
    kn = cal["y"] != cp.UNK
    nov_tau = cp.conformal_quantile(1 - cal["probs"][kn].max(1), ALPHA)
    finish("pooled+msp", cp.predict_sets(te["S"], te["client"], tau, 1 - te["probs"].max(1), nov_tau))
    return res, per


def run(seed, Xall, mode):
    torch.manual_seed(seed)
    s, info = load_split(mode, seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()

    tr = part == "train"
    assert (y[tr] != cp.UNK).all() and seen[client[tr]].all()  # guaranteed by splits.validate
    Xt, yt = X[tr], torch.from_numpy(y[tr]).to(DEV)
    heads = {"fedavg": fedavg(Xt, yt, client[tr], C, seed), "central": central(Xt, yt, C, seed)}

    out, per_all = {"unknown": info["unknown_y"], "n_clients": K, "n_unseen": int((~seen).sum()),
                    "split_sha256": info["sha256"]}, {}
    masks = {"cal": np.isin(part, ["tune", "cal"]), "test": part == "test"}
    for hn, head in heads.items():
        parts = {p: dict(probs=probs_of(head, X[msk]), y=y[msk], client=client[msk],
                         seq=s.seq_id.to_numpy()[msk]) for p, msk in masks.items()}
        for d in parts.values():
            d["S"] = 1.0 - d["probs"]  # LAC
        te = parts["test"]
        kn = te["y"] != cp.UNK
        out[f"{hn}_acc_known"] = float((te["probs"].argmax(1)[kn] == te["y"][kn]).mean())
        pi = np.array([np.mean(te["y"][te["client"] == k] == cp.UNK) for k in range(K)])
        out["unk_frac_quantiles"] = np.quantile(pi, [0.5, 0.9, 1]).round(3).tolist()
        out[hn], per_all[hn] = calibrate_and_eval(parts["cal"], te, C, K, seen)
    return out, per_all


if __name__ == "__main__":
    Xall = np.load(f"{ROOT}/features/iwildcam_bioclip.npy")
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    mode = sys.argv[2] if len(sys.argv) > 2 else "time"
    allr = []
    for sd in seeds:
        r, per = run(sd, Xall, mode)
        allr.append(r)
        json.dump(per, open(f"{ROOT}/results/iwildcam_v2_{mode}_perclient_seed{sd}.json", "w"))
        print(f"seed {sd}: clients={r['n_clients']} (unseen {r['n_unseen']}), unknown={r['unknown']}, "
              f"acc fedavg={r['fedavg_acc_known']:.3f} central={r['central_acc_known']:.3f}, "
              f"unk_frac median/p90/max={r['unk_frac_quantiles']}", flush=True)
    json.dump(allr, open(f"{ROOT}/results/iwildcam_v2_{mode}_summary.json", "w"), indent=1)
    for hn in ["fedavg", "central"]:
        print(f"\n== iWildCam protocol v2, split={mode}, head={hn} (alpha={ALPHA}, {len(allr)} seeds)")
        print(f"{'method':<12}{'mean_cov':>9}{'p10_known':>10}{'%sig_clus':>10}{'%sig_unseen':>12}"
              f"{'%sig_binom':>11}{'set_size':>9}{'unk_recall':>11}")
        for mth in allr[0][hn]:
            a = {k: np.mean([r[hn][mth][k] for r in allr]) for k in allr[0][hn][mth]}
            print(f"{mth:<12}{a['mean_cov']:9.3f}{a['p10_cov_known']:10.3f}{a['pct_sig_below']:9.0f}%"
                  f"{a['pct_sig_below_unseen']:11.0f}%{a['pct_sig_below_binom']:10.0f}%"
                  f"{a['mean_size']:9.2f}{a['unk_recall']:11.2f}")
