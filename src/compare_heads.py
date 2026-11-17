"""Which head x score gives usable set sizes? Static calibration, validated time splits.

heads : zeroshot (BioCLIP text, no training), fedavg_zero (protocol-v2 head), fedavg_text
        (FedAvg initialised at the text embeddings), central_text (reference)
scores: lac  -> calibrated on tune+cal (no hyper-parameters)
        raps -> (lam, kreg) chosen on TUNE only (scores.tune_raps); calibrated on cal only
Head hyper-parameters are the protocol-v2 defaults, fixed a priori (not tuned on test).
Usage: python compare_heads.py <n_seeds>
"""
import json
import sys

import numpy as np
import torch

import cp_baselines as cp
import scores
from run_iwildcam import ALPHA, DEV, calibrate_and_eval, central, fedavg, probs_of
from splits import load_split

ROOT = "${FEDPCT_ROOT}"


def run(seed, Xall, Tall):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    T = Tall[info["known_y"]]
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    seq = s.seq_id.to_numpy()
    tr = part == "train"
    Xt, yt = X[tr], torch.from_numpy(y[tr]).to(DEV)

    zs = torch.nn.Linear(X.shape[1], C).to(DEV)
    with torch.no_grad():
        zs.weight.copy_(10 * torch.as_tensor(T)), zs.bias.zero_()
    heads = {"zeroshot": zs,
             "fedavg_zero": fedavg(Xt, yt, client[tr], C, seed),
             "fedavg_text": fedavg(Xt, yt, client[tr], C, seed, init=T),
             "central_text": central(Xt, yt, C, seed, init=T)}

    out = {}
    for hn, head in heads.items():
        P = {p: probs_of(head, X[part == p]) for p in ["tune", "cal", "test"]}
        d = {p: dict(probs=P[p], y=y[part == p], client=client[part == p], seq=seq[part == p])
             for p in P}
        te = d["test"]
        kn = te["y"] != cp.UNK
        acc = float((te["probs"].argmax(1)[kn] == te["y"][kn]).mean())

        lac_cal = {k: np.concatenate([d["tune"][k], d["cal"][k]]) for k in d["cal"]}
        lac_cal["S"] = scores.lac(lac_cal["probs"])
        te_lac = dict(te, S=scores.lac(te["probs"]))
        res_lac, _ = calibrate_and_eval(lac_cal, te_lac, C, K, seen)

        lam, kreg = scores.tune_raps(d["tune"]["probs"], d["tune"]["y"], d["tune"]["seq"], ALPHA, seed)
        raps_cal = dict(d["cal"], S=scores.raps(d["cal"]["probs"], lam, kreg, seed + 1))
        te_raps = dict(te, S=scores.raps(te["probs"], lam, kreg, seed + 2))
        res_raps, _ = calibrate_and_eval(raps_cal, te_raps, C, K, seen)
        out[hn] = dict(acc=acc, raps_params=[lam, kreg], lac=res_lac, raps=res_raps)
        print(f"  seed {seed} {hn}: acc={acc:.3f} raps(lam,k)=({lam},{kreg})", flush=True)
    return out


if __name__ == "__main__":
    Xall = np.load(f"{ROOT}/features/iwildcam_bioclip.npy")
    Tall = np.load(f"{ROOT}/features/iwildcam_text_bioclip.npy")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    allr = [run(sd, Xall, Tall) for sd in range(n)]
    json.dump(allr, open(f"{ROOT}/results/compare_heads_time.json", "w"), indent=1, default=float)
    print(f"\n== heads x scores, time split, static calibration, alpha={ALPHA}, {n} seeds")
    print(f"{'head':<14}{'acc':>6} {'score':<6}{'method':<12}{'mean_cov':>9}{'p10_known':>10}"
          f"{'%sig':>6}{'%sig_unseen':>12}{'size':>7}")
    for hn in allr[0]:
        acc = np.mean([r[hn]["acc"] for r in allr])
        for sc in ["lac", "raps"]:
            for m in ["pooled", "labelshift", "local"]:
                a = {k: np.mean([r[hn][sc][m][k] for r in allr]) for k in allr[0][hn][sc][m]}
                print(f"{hn:<14}{acc:6.3f} {sc:<6}{m:<12}{a['mean_cov']:9.3f}{a['p10_cov_known']:10.3f}"
                      f"{a['pct_sig_below']:5.0f}%{a['pct_sig_below_unseen']:11.0f}%{a['mean_size']:7.2f}")
