"""POST HOC sensitivity of the CCT lockbox evaluation to the pre-specified near-duplicate rule
(requested by the pre-submission audit of 2026-11-16; the CCT lockbox has already been used, so this is
not confirmatory).

1. Rebuild the five CCT time splits with the frozen split code and the same seeds, but WITHOUT dropping the
   test sequences that have a near-duplicate (cos > 0.98) of a training image of the same site. Everything
   before the drop (cleaning, sites, held-out classes, train/cal/tune partitions) is identical, which is
   asserted against the published splits. Written to splits_cct_keepdup/ (the published splits are untouched).
2. Rerun the frozen lockbox_eval.run on the CCT lockbox sites with these splits: same features, models,
   selection rules, proximal weights, step-size rule, label draws and thresholds.
Usage: python cct_keepdup.py
"""
import json
import os
import shutil

import numpy as np

import cct_env

cct_env.use_cct()
import splits  # noqa: E402
import lockbox  # noqa: E402

ROOT = cct_env.ROOT
PUB = cct_env.SPLIT_DIR
NEW = f"{ROOT}/splits_cct_keepdup"
meta, X = cct_env.load_meta()


def build(seed):
    try:
        return splits.build(meta, seed, "time", X), splits.UNK_POOL_TRAIN_RANGE
    except ValueError:
        old = splits.UNK_POOL_TRAIN_RANGE
        splits.UNK_POOL_TRAIN_RANGE = (old[0], cct_env.POOL_FALLBACK_MAX)
        try:
            return splits.build(meta, seed, "time", X), splits.UNK_POOL_TRAIN_RANGE
        finally:
            splits.UNK_POOL_TRAIN_RANGE = old


restored = {}
if not os.path.exists(f"{NEW}/time_seed4.json"):
    os.makedirs(NEW, exist_ok=True)
    for seed in range(5):
        splits.SPLIT_DIR = PUB
        pub, pinfo = splits.load_split("time", seed)
        (s, info), rng_used = build(seed)
        info["dataset"] = "cct"
        info["unk_pool_train_range"] = list(rng_used)
        info["note"] = "post hoc: near-duplicate test sequences retained"
        diag = splits.validate(s, info, X)
        # identical apart from the restored test sequences
        assert info["unknown_y"] == pinfo["unknown_y"] and info["locations"] == pinfo["locations"]
        for part in ("train", "cal", "tune"):
            a = set(pub.row[pub.part == part]); b = set(s.row[s.part == part])
            assert a == b, (seed, part, len(a ^ b))
        a, b = set(pub.row[pub.part == "test"]), set(s.row[s.part == "test"])
        assert a <= b
        splits.SPLIT_DIR = NEW
        splits.save(s, info, diag)
        restored[seed] = dict(test_rows_restored=len(b - a), test_rows=len(b),
                              seqs_restored=int(s[s.part == "test"].seq_id.nunique() - pub[pub.part == "test"].seq_id.nunique()))
        print(f"seed {seed}: restored {restored[seed]}", flush=True)
    shutil.copy2(f"{PUB}/lockbox_sites.json", f"{NEW}/lockbox_sites.json")
    json.dump(restored, open(f"{NEW}/restored.json", "w"), indent=1)
restored = json.load(open(f"{NEW}/restored.json"))

# evaluate with the frozen lockbox_eval functions on the retained-test splits
splits.SPLIT_DIR = NEW
lockbox.PATH = f"{NEW}/lockbox_sites.json"
import lockbox_eval as le  # noqa: E402
le.TAG = cct_env.TAG
sel = json.load(open(f"{ROOT}/results/pers_lambda_{le.TAG}.json"))
allr = []
for sd in le.SEEDS:
    lam_star = {rate: sel[sd][f"r={rate}"]["lam_star"] for rate in le.RATES}
    allr.append(le.run(sd, "lockbox", lam_star))
    print(f"seed {sd} done", flush=True)
json.dump(dict(restored=restored, runs=allr), open(f"{ROOT}/results/cct_keepdup_lockbox.json", "w"), indent=1, default=float)
le.report(allr, "CCT lockbox, near-duplicate test sequences retained (post hoc)")


def m(get, k):
    return float(np.mean([get(r)[k] for r in allr]))


print("SUMMARY")
for name, get in [("fcp", lambda r: r["fcp"]), ("labelshift", lambda r: r["labelshift"]), ("qq", lambda r: r["qq"]), ("dr", lambda r: r["dr"])]:
    print(name, {k: round(m(get, k), 3) for k in ("cov", "sig", "sig_unseen", "size")})
for q in le.RATES:
    k = f"r={q}"
    for name in ("global_qt", "ditto_fixed", "ours"):
        print(k, name, {kk: round(m(lambda r: r[k][name], kk), 3) for kk in ("cov", "sig", "sig_unseen", "size")})
    d = [r[k]["ours"]["sig"] - r["fcp"]["sig"] for r in allr]
    f = [r[k]["ours"]["sig"] - r[k]["frontier_sig_at_ours_size"] for r in allr]
    print(k, "ours-fcp", round(float(np.mean(d)), 1), np.round(d, 1).tolist(), "ours-frontier", round(float(np.mean(f)), 1), np.round(f, 1).tolist())
print("KEEPDUP_DONE")
