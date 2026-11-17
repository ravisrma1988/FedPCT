"""RAPS parameters selected on FMoW-WILDS per seed (same calls as lockbox_eval.run, nothing else)."""
import fmow_env

fmow_env.use_fmow()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lockbox_eval as le  # noqa: E402
import scores  # noqa: E402
from run_iwildcam import ALPHA, DEV, fedavg, probs_of  # noqa: E402
from splits import load_split  # noqa: E402

for seed in range(5):
    torch.manual_seed(seed)
    s, info = load_split("time", seed)
    C = len(info["known_y"])
    X = torch.from_numpy(np.load(f"{le.ROOT}/features/iwildcam_{fmow_env.TAG}.npy")[s.row.to_numpy()]).float().to(DEV)
    y, client, part, seqs = s.label.to_numpy(), s.client.to_numpy(), s.part.to_numpy(), s.seq_id.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), client[tr], C, seed)
    P_all = probs_of(glob, X)
    tm = part == "tune"
    print("FMOW_RAPS seed", seed, scores.tune_raps(P_all[tm], y[tm], seqs[tm], ALPHA, seed), flush=True)
