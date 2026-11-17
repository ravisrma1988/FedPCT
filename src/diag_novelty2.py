"""Why are unknown species missed, and do detector / text signals help? Crop features, all sites.
AUROC on tune (selection) and test (report only). Higher score = more novel. Leakage rules: only
KNOWN-class names / train data are used; unknown species' names and labels are never touched.
  msp          1 - max softmax of the federated head (current choice)
  msp_animal   if MegaDetector found an animal (conf >= CONF), drop the "empty" class and
               renormalise before taking 1 - max (an animal the head calls "empty" is suspicious)
  energy_an    -logsumexp of logits, "empty" excluded when an animal was detected
  text         1 - max cos(image, BioCLIP-2 text embedding of a KNOWN class prompt)
  fuse         mean of tune-standardised ranks of msp_animal and text (fixed a priori, not tuned)
Breakdown (test): unknowns by detector outcome, and the head's top-1 for unknowns ("empty" share).
Usage: python diag_novelty2.py <n_seeds>
"""
import sys

import numpy as np
import open_clip
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import cp_baselines as cp
import vit_split as vs
from run_iwildcam import DEV, fedavg
from splits import load_split

ROOT = "${FEDPCT_ROOT}"
TAG = "bioclip2crop"
CONF = 0.2  # same detection rule as the crops (fixed before results)

cats = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
cats = cats[cats.y != 99999].set_index("y")
EMPTY_Y = int(cats.index[cats.name == "empty"][0])
md = pd.read_csv(f"{ROOT}/features/md_best.csv").set_index("row")
model, _, _ = open_clip.create_model_and_transforms(vs.MODEL)
tok = open_clip.get_tokenizer(vs.MODEL)
model = model.to(DEV).eval()
Xall = np.load(f"{ROOT}/features/iwildcam_{TAG}.npy")


def text_emb(ys):
    prompts = ["a camera trap photo with no animal." if c == EMPTY_Y else f"a photo of {cats.name[c]}." for c in ys]
    with torch.no_grad():
        return torch.nn.functional.normalize(model.encode_text(tok(prompts).to(DEV)).float(), dim=-1)


for seed in range(int(sys.argv[1])):
    s, info = load_split("time", seed)
    known = info["known_y"]
    assert not set(known) & set(info["unknown_y"])
    C, e = len(known), known.index(EMPTY_Y)
    X = torch.from_numpy(Xall[s.row.to_numpy()]).float().to(DEV)
    y, part = s.label.to_numpy(), s.part.to_numpy()
    tr = part == "train"
    glob = fedavg(X[tr], torch.from_numpy(y[tr]).to(DEV), s.client.to_numpy()[tr], C, seed)
    an = torch.as_tensor(md.has_animal.reindex(s.row).to_numpy(bool), device=DEV)
    with torch.no_grad():
        logit = glob(X * 10)
        P = torch.softmax(logit, 1)
        la = logit.clone()
        la[an, e] = -torch.inf
        Pa = torch.softmax(la, 1)
        T = text_emb(known)                                   # known classes only
        txt = 1 - (torch.nn.functional.normalize(X, dim=1) @ T.T).max(1).values
    nov = {"msp": 1 - P.max(1).values, "msp_animal": 1 - Pa.max(1).values,
           "energy_an": -torch.logsumexp(la, 1), "text": txt}
    nov = {k: v.cpu().numpy() for k, v in nov.items()}
    tm, te = part == "tune", part == "test"
    unk = y == cp.UNK
    rk = lambda v: rankdata(v) / len(v)  # noqa: E731
    nov["fuse"] = (rk(nov["msp_animal"]) + rk(nov["text"])) / 2
    line = [f"{k}: tune {roc_auc_score(unk[tm], v[tm]):.3f} test {roc_auc_score(unk[te], v[te]):.3f}"
            for k, v in nov.items()]
    print(f"seed {seed}: " + " | ".join(line), flush=True)
    a = an.cpu().numpy()
    top = P.argmax(1).cpu().numpy()
    for name, m in [("detected", a), ("not detected", ~a)]:
        sel = te & m
        au = roc_auc_score(unk[sel], nov["msp"][sel]) if 0 < unk[sel].sum() < sel.sum() else np.nan
        print(f"   test unknowns {name:<13}: {unk[sel].sum():5d} of {unk[te].sum()} "
              f"(knowns {(~unk[sel]).sum():6d}) | msp AUROC within stratum {au:.3f} | "
              f"top-1 'empty' {np.mean(top[sel & unk] == e):.2f}", flush=True)
    for t in ("designed", "natural"):
        sel = te & ((s.unk_type.to_numpy() == t) | ~unk)
        if (sel & unk).sum():
            print(f"   {t:<9} unknowns: n={(sel & unk).sum():5d}  msp {roc_auc_score(unk[sel], nov['msp'][sel]):.3f}  "
                  f"msp_animal {roc_auc_score(unk[sel], nov['msp_animal'][sel]):.3f}  "
                  f"text {roc_auc_score(unk[sel], nov['text'][sel]):.3f}", flush=True)
