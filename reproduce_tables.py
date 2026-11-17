"""Rebuild the result tables of the manuscript and the main supplementary tables from the stored results.

No model is trained and no lockbox inference is rerun. The script reads only results/*.json in this repository
and needs only the Python 3 standard library. Usage, from the repository root or anywhere else:
    python reproduce_tables.py
Values are means over the five seeds, as in the manuscript. Table numbers follow the IEEE Access version:
Tables 3 to 5 (lockbox evaluations), 6 (post hoc lockbox analysis with the mean shortfall), 7 (ablations),
8 (backbone), 9 (online trackers) and 10 (theory check), followed by the post hoc review analyses (static
frontier of the global classifier, CCT near-duplicate retention) and supplementary tables S7, S10 and S11.
"""
import json
import os
import statistics

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf8") as f:
        return json.load(f)


def mean(rows, *path):
    vals = []
    for r in rows:
        for p in path:
            r = r[p]
        vals.append(r)
    return float(statistics.fmean(vals))


def lockbox_table(d, title):
    print(f"\n{title}\nMethod                      r     Cov.   Below(%)  Unseen(%)  Size")
    for key, lab in (("fcp", "FCP"), ("labelshift", "Label-shift"), ("qq", "Quantile of quantiles"), ("dr", "Density ratio")):
        print(f"{lab:<27} none  {mean(d, key, 'cov'):.3f}  {mean(d, key, 'sig'):>6.0f}  {mean(d, key, 'sig_unseen'):>8.0f}   {mean(d, key, 'size'):.2f}")
    for key, lab in (("global_qt", "Global and tracking"), ("ditto_fixed", "Personalized, fixed"), ("ours", "FedPCT")):
        for r in ("0.1", "0.3", "1.0"):
            k = f"r={r}"
            print(f"{lab:<27} {r}   {mean(d, k, key, 'cov'):.3f}  {mean(d, k, key, 'sig'):>6.0f}  "
                  f"{mean(d, k, key, 'sig_unseen'):>8.0f}   {mean(d, k, key, 'size'):.2f}")
            lab = ""
    for r in ("0.1", "0.3", "1.0"):
        k = f"r={r}"
        dif = [x[k]["ours"]["sig"] - x[k]["frontier_sig_at_ours_size"] for x in d]
        print(f"   tracking minus personalized static frontier at matched size, r={r}: {statistics.fmean(dif):+.1f} points "
              f"(seeds {[round(v, 1) for v in dif]})")


def posthoc_table():
    print("\nTable 6. Post hoc analysis of the lockbox sites, all classes and animal classes")
    print("Dataset   Classes  Method            Flagged(%)  Obs<0.9(%)  Shortfall  Mean   Median  Size")
    for ds, name in (("iwildcam", "iWildCam"), ("cct", "CCT")):
        d, rv = load(f"posthoc_animal_{ds}.json"), load(f"posthoc_review_{ds}.json")
        for cls, clab in (("all", "All"), ("animal", "Animal")):
            for key, lab in ((f"fcp_{cls}", "FCP"), (f"ours_{cls}|0.3", "FedPCT, r=0.3"), (f"ours_{cls}|1.0", "FedPCT, r=1.0")):
                v = d[key]
                print(f"{name:<9} {clab:<8} {lab:<17} {v['sig']:>6.0f}  {v['below_obs']:>10.0f}  {rv[key]['shortfall']:>9.3f}  "
                      f"{v['cov_mean']:.3f}  {v['cov_median']:.3f}  {v['size']:.2f}")


def ablation_table():
    print("\nTable 7. Ablations and sensitivity on iWildCam development sites (flagged % / mean set size)")
    print("Configuration                              rate 0.3        rate 1.0")
    fmt = lambda d, r, m: f"{mean(d, 'r=' + r, m, 'sig'):.0f}% / {mean(d, 'r=' + r, m, 'size'):.2f}"  # noqa: E731
    base = load("sens_base.json")
    print(f"{'FCP (reference)':<42} {fmt(base, '0.3', 'fcp'):<15} {fmt(base, '1.0', 'fcp')}")
    rows = [("base", "FedPCT (delay 5, refit every 50 images)"), ("local_only", "Local classifiers without federation"),
            ("finetune_l0", "Fine-tuning without proximal term"), ("central_head", "Centrally trained global classifier"),
            ("delay=0", "Delay 0 sequences"), ("delay=2", "Delay 2 sequences"), ("delay=10", "Delay 10 sequences"),
            ("refit=20", "Refit every 20 labelled images"), ("refit=100", "Refit every 100 labelled images")]
    for key, lab in rows:
        d = load(f"sens_{key}.json")
        print(f"{lab:<42} {fmt(d, '0.3', 'ours'):<15} {fmt(d, '1.0', 'ours')}")
    for a in ("0.05", "0.2"):
        d = load(f"sens_alpha={a}.json")
        lab = f"Target alpha={a} (FCP {fmt(d, '1.0', 'fcp')})"
        print(f"{lab:<42} {fmt(d, '0.3', 'ours'):<15} {fmt(d, '1.0', 'ours')}")


def backbone_table():
    print("\nTable 8. Backbone robustness with DINOv2 features (flagged % / mean set size)")
    print("Sites                    FCP             FedPCT, rate 0.3  FedPCT, rate 1.0")
    for f, lab in (("backbone_dinov2crop_dev.json", "iWildCam development"), ("backbone_dinov2crop_lockbox.json", "iWildCam lockbox"),
                   ("backbone_cct_dinov2crop_dev.json", "CCT development"), ("backbone_cct_dinov2crop_lockbox.json", "CCT lockbox")):
        d = load(f)
        cells = [f"{mean(d, 'fcp', 'sig'):.0f}% / {mean(d, 'fcp', 'size'):.2f}"]
        cells += [f"{mean(d, 'r=' + r, 'ours', 'sig'):.0f}% / {mean(d, 'r=' + r, 'ours', 'size'):.2f}" for r in ("0.3", "1.0")]
        print(f"{lab:<24} {cells[0]:<15} {cells[1]:<17} {cells[2]}")


def tracker_table():
    print("\nTable 9. Per-site trackers with the personalized classifier (flagged % / mean set size)")
    for f, lab in (("online_baselines_dev_iwildcam.json", "iWildCam development"), ("online_baselines_dev_cct.json", "CCT development"),
                   ("online_baselines_fmow_lockbox.json", "FMoW-WILDS lockbox")):
        d = load(f)
        for key, name in (("ditto_qt", "FedPCT"), ("ditto_decay", "Decaying step size"), ("ditto_dtaci", "DtACI"), ("ditto_pid", "PID integrator")):
            cells = [f"{mean(d, 'r=' + r, key, 'sig'):.2f}% / {mean(d, 'r=' + r, key, 'size'):.2f}" for r in ("0.3", "1.0")]
            print(f"{lab:<22} {name:<20} {cells[0]:<17} {cells[1]}")
            lab = ""


def theory_table():
    print("\nTable 10. Numerical check of the theory on development sites")
    print("Dataset   r    Streams  Violations  Bound   Floor<0.10  Observed")
    for ds, name in (("iwildcam", "iWildCam"), ("cct", "CCT")):
        a, b = load(f"theory_check_{ds}.json"), load(f"theory_check2_{ds}.json")
        for r in (0.1, 0.3, 1.0):
            ra = [o for o in a if abs(o["r"] - r) < 1e-9]
            rb = [o for o in b if abs(o["r"] - r) < 1e-9]
            viol = sum((not o["lemma_ok"]) or o["dev"] > o["bound"] for o in ra)
            viol += sum(o["dev_frozen"] > o["bound_frozen"] for o in rb)
            bound = float(statistics.median([o["bound_frozen"] for o in rb]))
            floor = 100 * float(statistics.fmean([o["floor"] < 0.10 for o in rb]))
            obs = float(statistics.median([o["dev"] for o in ra]))
            print(f"{name:<9} {r:<4} {len(ra):>7}  {viol:>10}  {(f'{bound:.1f}' if bound >= 1 else f'{bound:.2f}'):>6}  {floor:>9.0f}%  {obs:.3f}")


def review_analyses():
    print("\nPost hoc: FedPCT against the static frontier of the global classifier (FCP at alpha' < 0.1) at matched set size")
    for ds, name in (("iwildcam", "iWildCam"), ("cct", "CCT"), ("fmow", "FMoW-WILDS")):
        d = load(f"posthoc_review_{ds}.json")
        for r in ("0.3", "1.0"):
            m = d[f"matched_global|{r}"]
            print(f"{name:<11} r={r}: frontier {m['frontier_sig']:.1f}%, FedPCT minus frontier {m['diff']:+.1f} points "
                  f"(seeds {m['diff_seeds']}, clipped seeds {m['clipped_seeds']})")
    d = load("cct_keepdup_lockbox.json")
    runs = d["runs"]
    print("\nPost hoc: CCT lockbox with the near-duplicate test sequences retained (flagged % / cov. / size)")
    print("restored per seed:", {k: (v["seqs_restored"], v["test_rows_restored"]) for k, v in d["restored"].items()})
    print(f"FCP                          {mean(runs, 'fcp', 'sig'):.1f}% / {mean(runs, 'fcp', 'cov'):.3f} / {mean(runs, 'fcp', 'size'):.2f}")
    for r in ("0.3", "1.0"):
        k = f"r={r}"
        print(f"FedPCT, r={r}                {mean(runs, k, 'ours', 'sig'):.1f}% / {mean(runs, k, 'ours', 'cov'):.3f} / {mean(runs, k, 'ours', 'size'):.2f}")
    k = "r=1.0"
    print(f"Personalized, fixed, r=1.0   {mean(runs, k, 'ditto_fixed', 'sig'):.1f}% / {mean(runs, k, 'ditto_fixed', 'cov'):.3f} / {mean(runs, k, 'ditto_fixed', 'size'):.2f}")
    dif = [x[k]["ours"]["sig"] - x[k]["frontier_sig_at_ours_size"] for x in runs]
    print(f"tracking minus personalized frontier, r=1.0: {statistics.fmean(dif):+.1f} points")


def sensitivity_tables():
    meth = (("fcp", "FCP"), ("ours|0.3", "FedPCT, r=0.3"), ("ours|1.0", "FedPCT, r=1.0"))
    print("\nSupplement, screening sensitivity (flagged %; BH-based screening sensitivity, not formal FDR control)")
    print("Dataset   Method          Sequence  BH screen  Block 5  Block 10  Block 20")
    for ds, name in (("iwildcam", "iWildCam"), ("cct", "CCT")):
        d = load(f"posthoc_sensitivity_{ds}.json")
        for k, lab in meth:
            v = d[k]
            print(f"{name:<9} {lab:<15} {v['sig']:>8.1f}  {v['bh']:>9.1f}  {v['block5']:>7.1f}  {v['block10']:>8.1f}  {v['block20']:>8.1f}")
    print("\nSupplement, coverage of individual classes")
    print("Dataset   Method          Classes   Median  Below(%)  Rare   Common  Non-animal")
    for ds, name in (("iwildcam", "iWildCam"), ("cct", "CCT")):
        d = load(f"posthoc_sensitivity_{ds}.json")
        for k, lab in meth:
            v = d[f"classes|{k}"]
            cl = f"{v['n_classes']:.0f} ({v['n_animal_classes']:.0f})"
            print(f"{name:<9} {lab:<15} {cl:<9} {v['animal_median_cov']:.3f}  {v['animal_share_below']:>7.0f}  "
                  f"{v['rare_animal_median_cov']:.3f}  {v['common_animal_median_cov']:.3f}  {v['nonanimal_cov']:.3f}")


if __name__ == "__main__":
    lockbox_table(load("final_eval_lockbox.json"), "Table 3. Confirmatory lockbox, iWildCam (43 sites, 10 unseen, five seeds)")
    lockbox_table(load("final_eval_cct_lockbox.json"), "Table 4. External validation lockbox, CCT (25 sites, 7 unseen, five seeds)")
    lockbox_table(load("final_eval_fmow_lockbox.json"), "Table 5. Confirmatory lockbox, FMoW-WILDS (32 sites, 8 unseen, five seeds)")
    posthoc_table()
    ablation_table()
    backbone_table()
    tracker_table()
    theory_table()
    review_analyses()
    sensitivity_tables()
