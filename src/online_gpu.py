"""GPU-vectorised online threshold tracking: same protocol and update as online.py, run for all
sites x configurations at once. Only the loop over sequence steps is sequential.

Exactness: a sequence's errors are judged against the threshold used when it was predicted, so
applying its feedback as one summed step equals applying it image by image. tau for sequence J
therefore is tau0 + eta * sum_{m <= J-DELAY-1} labelled_m * (misses_m - alpha * n_known_m).

Use: from online_gpu import track; or run `python online_gpu.py check` to compare with online.py's
loop on seed 0 (coverage and size must match to float precision).
"""
import sys

import numpy as np
import torch

DEV = "cuda"


def pack(sites):
    """sites: list of dicts with 'score' (n,) true-label score in stream order (nan = unknown),
    'seq_codes' (n,) contiguous 0..J-1. Returns padded per-sequence tensors on the GPU."""
    K = len(sites)
    J = max(int(s["seq_codes"].max()) + 1 for s in sites)
    I = max(int(np.bincount(s["seq_codes"]).max()) for s in sites)
    pad = torch.full((K, J, I), float("nan"), device=DEV)
    nk = torch.zeros(K, J, device=DEV)
    for k, s in enumerate(sites):
        codes = s["seq_codes"]
        pos = np.arange(len(codes)) - np.searchsorted(codes, codes)  # position within its sequence
        pad[k, codes, pos] = torch.as_tensor(s["score"], dtype=torch.float32, device=DEV)
        nk[k] = torch.bincount(torch.as_tensor(codes[~np.isnan(s["score"])], device=DEV), minlength=J)[:J]
    return pad, nk


def track(pad, nk, labelled, tau0, eta, alpha, delay):
    """pad (K,J,I), nk (K,J), labelled (G,K,J) bool, tau0 (G,K), eta (G,). Returns tau (G,K,J)."""
    G, K, J = labelled.shape
    taus = torch.empty(G, K, J, device=DEV)
    tau = tau0.clone()
    lab = labelled.float()
    for j in range(J):
        taus[:, :, j] = tau
        m = j - delay
        if m >= 0:
            miss = (pad[None, :, m, :] > taus[:, :, m, None]).sum(-1).float()  # nan > x is False
            tau = tau + eta[:, None] * lab[:, :, m] * (miss - alpha * nk[None, :, m])
    return taus


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    import online  # the CPU reference implementation

    rng = np.random.default_rng(0)
    sites, refs = [], []
    for _ in range(6):
        n_seq = rng.integers(20, 200)
        codes = np.repeat(np.arange(n_seq), rng.integers(1, 6, n_seq))
        score = rng.random(len(codes))
        score[rng.random(len(codes)) < 0.1] = np.nan
        sites.append(dict(score=score, seq_codes=codes))
    pad, nk = pack(sites)
    J = pad.shape[1]
    etas = [0.005, 0.05]
    lab = torch.zeros(len(etas), len(sites), J, dtype=torch.bool, device=DEV)
    worst = 0.0
    for k, s in enumerate(sites):
        L = rng.random(s["seq_codes"].max() + 1) < 0.4
        lab[:, k, :len(L)] = torch.as_tensor(L, device=DEV)
        for g, eta in enumerate(etas):
            refs.append((g, k, online.run_site(s["score"], s["seq_codes"], L, 0.3, eta)))
    taus = track(pad, nk, lab, torch.full((len(etas), len(sites)), 0.3, device=DEV),
                 torch.tensor(etas, device=DEV), online.ALPHA, online.DELAY)
    for g, k, ref in refs:
        got = taus[g, k, torch.as_tensor(sites[k]["seq_codes"], device=DEV)].cpu().numpy()
        worst = max(worst, float(np.abs(got - ref).max()))
    print(f"max |tau_gpu - tau_cpu| = {worst:.2e}  ->", "OK" if worst < 1e-4 else "MISMATCH")
