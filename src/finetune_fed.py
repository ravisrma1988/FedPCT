"""Federated partial fine-tuning of BioCLIP-2 (blocks 22-23 + ln_post + proj + linear head), LP-FT
style: the head starts from the FedAvg linear probe on frozen BioCLIP-2 features (Kumar et al.,
ICLR 2022), then FedAvg trains top + head from cached block-22 tokens (train partition only).
Afterwards every split row (tune/cal/test) is re-encoded from JPEG with the tuned top.

Hyper-parameters are fixed a priori (not tuned on test): ROUNDS, PER_ROUND, LOCAL_EPOCHS, BS, LRs.
Usage: python finetune_fed.py <seed>
Outputs: features/iwildcam_bioclip2ft_seed<s>.npy (203029 x 768 float16; rows outside the split = 0)
         features/bioclip2ft_seed<s>.json (settings, train loss per round, tune accuracy)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import vit_split as vs
from run_iwildcam import fedavg
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
IMG_DIR = f"{ROOT}/data/iwildcam_v2.0/train"
ROUNDS, PER_ROUND, LOCAL_EPOCHS, BS = 60, 20, 1, 64
LR_TOP, LR_HEAD, WD = 1e-5, 1e-4, 0.05
SEED = int(sys.argv[1])
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

s, info = load_split("time", SEED)
C = len(info["known_y"])
tr = s[s.part == "train"]
idx = pd.read_csv(f"{ROOT}/features/tokens_b22_index.csv")
pos_of_row = pd.Series(idx.pos.to_numpy(), index=idx.row.to_numpy())
tr = tr.assign(pos=pos_of_row.loc[tr.row.to_numpy()].to_numpy())
tokens = np.load(f"{ROOT}/features/tokens_b22.npy", mmap_mode="r")
clients = {k: (np.sort(g.pos.to_numpy()), g.sort_values("pos").label.to_numpy())
           for k, g in tr.groupby("client")}
print(f"seed {SEED}: {len(tr)} train rows, {len(clients)} train clients, C={C}", flush=True)

# 1) Linear-probe initialisation on frozen BioCLIP-2 features (same FedAvg as protocol v2).
F = np.load(f"{ROOT}/features/iwildcam_bioclip2.npy")
Xtr = torch.from_numpy(F[tr.row.to_numpy()]).float().cuda()
lp = fedavg(Xtr, torch.from_numpy(tr.label.to_numpy()).cuda(), tr.client.to_numpy(), C, SEED)
del Xtr

# 2) Federated fine-tuning of the top + head.
v, size, mean, std = vs.load()
for p in v.parameters():
    p.requires_grad_(False)
glob_top = vs.Top(v).cuda()
glob_head = torch.nn.Linear(F.shape[1], C).cuda()
glob_head.load_state_dict(lp.state_dict())
del F


def state():
    return {**{f"top.{k}": t for k, t in glob_top.state_dict().items()},
            **{f"head.{k}": t for k, t in glob_head.state_dict().items()}}


log, t0 = [], time.time()
for rnd in range(ROUNDS):
    sel = rng.choice(list(clients), min(PER_ROUND, len(clients)), replace=False)
    acc_state, tot, losses = None, 0, []
    for k in sel:
        pos, lab = clients[k]
        # The cache is sorted by client; one contiguous read per client, then drop rows that are
        # not in this seed's train set (other seeds' unknown species).
        X = torch.from_numpy(np.asarray(tokens[pos[0]:pos[-1] + 1])[pos - pos[0]]).cuda(non_blocking=True)
        y = torch.from_numpy(lab).cuda()
        top, head = vs.Top(v).cuda(), torch.nn.Linear(glob_head.in_features, C).cuda()
        top.load_state_dict(glob_top.state_dict()), head.load_state_dict(glob_head.state_dict())
        opt = torch.optim.AdamW([{"params": top.parameters(), "lr": LR_TOP},
                                 {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
        top.train()
        for _ in range(LOCAL_EPOCHS):
            for b in torch.randperm(len(y), device="cuda").split(BS):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(head(top(X[b]) * 10), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        st = {**{f"top.{n}": t for n, t in top.state_dict().items()},
              **{f"head.{n}": t for n, t in head.state_dict().items()}}
        w = len(y)
        acc_state = {n: t.float() * w for n, t in st.items()} if acc_state is None else \
            {n: acc_state[n] + st[n].float() * w for n in st}
        tot += w
        del X, top, head, opt
    new = {n: (t / tot) for n, t in acc_state.items()}
    glob_top.load_state_dict({n[4:]: t.to(glob_top.state_dict()[n[4:]].dtype) for n, t in new.items() if n.startswith("top.")})
    glob_head.load_state_dict({n[5:]: t for n, t in new.items() if n.startswith("head.")})
    log.append(float(np.mean(losses)))
    if rnd % 10 == 0 or rnd == ROUNDS - 1:
        print(f"round {rnd}: mean local loss {log[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

# 3) Re-encode every split row (tune/cal/test and train) from JPEG with the tuned top.
meta = pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv")
rows = np.sort(s.row.to_numpy())
it = vs.dali_iter([f"{IMG_DIR}/{f}" for f in meta.filename.to_numpy()[rows]], size, mean, std, batch=256)
out = np.zeros((len(meta), glob_head.in_features), dtype=np.float16)
glob_top.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for b in it:
        x, i = b[0]["img"], b[0]["idx"].squeeze(-1).long().cpu().numpy()
        out[rows[i]] = glob_top(vs.trunk(v, x)).float().cpu().numpy().astype(np.float16)
np.save(f"{ROOT}/features/iwildcam_bioclip2ft_seed{SEED}.npy", out)

tune = s[s.part == "tune"]
with torch.no_grad():
    Xt = torch.from_numpy(out[tune.row.to_numpy()]).float().cuda()
    pred = glob_head(Xt * 10).argmax(1).cpu().numpy()
kn = tune.label.to_numpy() >= 0
tune_acc = float((pred[kn] == tune.label.to_numpy()[kn]).mean())
json.dump(dict(seed=SEED, rounds=ROUNDS, per_round=PER_ROUND, local_epochs=LOCAL_EPOCHS, bs=BS,
               lr_top=LR_TOP, lr_head=LR_HEAD, wd=WD, cut=vs.CUT, loss=log, tune_acc_known=tune_acc,
               split_sha256=info["sha256"], seconds=round(time.time() - t0)),
          open(f"{ROOT}/features/bioclip2ft_seed{SEED}.json", "w"), indent=1)
print(f"FT_DONE seed {SEED}: tune known-class acc (FT head) {tune_acc:.3f}  {time.time() - t0:.0f}s", flush=True)
