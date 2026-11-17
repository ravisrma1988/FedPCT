"""Frozen-backbone comparison on the validated time splits. Same heads (FedAvg, central; zero init,
protocol-v2 hyper-parameters fixed a priori) and LAC calibration on tune+cal; only the feature
file changes. Features are row-aligned with iwildcam_meta.csv, which splits index via `row`.
Usage: python compare_backbones.py <n_seeds> <tag> [<tag> ...]   (features/iwildcam_<tag>.npy)
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


def run(seed, Xall):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C, K = len(info["known_y"]), info["n_clients"]
    seen = s.groupby("client").seen.first().reindex(range(K)).to_numpy()
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, client, part = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy()
    seq = s.seq_id.to_numpy()
    tr = part == "train"
    Xt, yt = X[tr], torch.from_numpy(y[tr]).to(DEV)
    out = {}
    for hn, head in {"fedavg": fedavg(Xt, yt, client[tr], C, seed), "central": central(Xt, yt, C, seed)}.items():
        calm, tem = np.isin(part, ["tune", "cal"]), part == "test"
        cal = dict(probs=probs_of(head, X[calm]), y=y[calm], client=client[calm], seq=seq[calm])
        te = dict(probs=probs_of(head, X[tem]), y=y[tem], client=client[tem], seq=seq[tem])
        cal["S"], te["S"] = scores.lac(cal["probs"]), scores.lac(te["probs"])
        kn = te["y"] != cp.UNK
        res, _ = calibrate_and_eval(cal, te, C, K, seen)
        res["acc"] = float((te["probs"].argmax(1)[kn] == te["y"][kn]).mean())
        out[hn] = res
    return out


if __name__ == "__main__":
    n, tags = int(sys.argv[1]), sys.argv[2:]
    allr = {}
    for tag in tags:  # a tag may contain {seed} (per-seed features, e.g. fine-tuned backbones)
        allr[tag] = [run(sd, np.load(f"{ROOT}/features/iwildcam_{tag.format(seed=sd)}.npy")) for sd in range(n)]
        print(f"{tag} done", flush=True)
    name = "_".join(t.replace("{seed}", "S") for t in tags)
    json.dump(allr, open(f"{ROOT}/results/compare_backbones_time_{name}.json", "w"), indent=1, default=float)
    print(f"\n== frozen backbones, time split, LAC, alpha={ALPHA}, {n} seeds")
    print(f"{'features':<14}{'head':<9}{'acc':>6} {'method':<12}{'mean_cov':>9}{'p10_known':>10}"
          f"{'%sig':>6}{'%sig_unseen':>12}{'size':>7}")
    for tag, rs in allr.items():
        for hn in ["fedavg", "central"]:
            acc = np.mean([r[hn]["acc"] for r in rs])
            for m in ["pooled", "labelshift", "local"]:
                a = {k: np.mean([r[hn][m][k] for r in rs]) for k in rs[0][hn][m]}
                print(f"{tag:<14}{hn:<9}{acc:6.3f} {m:<12}{a['mean_cov']:9.3f}{a['p10_cov_known']:10.3f}"
                      f"{a['pct_sig_below']:5.0f}%{a['pct_sig_below_unseen']:11.0f}%{a['mean_size']:7.2f}")
