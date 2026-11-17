# Pre-registration: frozen method and lockbox protocol

Status: FROZEN 2026-11-02, after all development-site experiments and before any lockbox
evaluation. SHA-256 of this file and of every script/input the evaluation uses is recorded in
`splits/prereg_freeze.json`; `src/lockbox_eval.py lockbox` refuses to run if any of them changed,
and refuses to run twice. The lockbox is run once and reported whatever the outcome.

## 1. Data and splits (unchanged from development)
- iWildCam v2.0 (WILDS), cleaned by `splits.clean()` (multi-species photos and exact duplicates removed).
- Clients: camera sites with >= 250 images (143). Time split per site: train 40% earliest /
  cal 20% / test 40% latest, by sequence start; tune = 1/3 of cal sequences. Splits for seeds 0-4,
  SHA-256 verified on load.
- 8 held-out "designed" unknown species per seed + natural unknowns (classes absent from train).
- **Lockbox:** the 43 sites in `splits/lockbox_sites.json` (33 seen + 10 unseen; drawn with rng
  20261101). Lockbox test images were seen by some earlier experiments (see the disclosure below);
  no method component after 2026-11-01 used them. Lockbox sites took part
  in federated training and calibration exactly as in deployment (their train/cal data are not
  test data); only their TEST partitions are held out.
- **Disclosure (lockbox not pristine):** the lockbox was drawn on 2026-11-01, after the static
  baseline, online-tracking, empirical-Bayes and persistence experiments, which reported test
  metrics over all sites. Later, `compare_backbones.py` (crop vs whole-image vs fine-tuned
  features) and `diag_novelty*.py` also reported test metrics over all sites, so the backbone
  choice saw lockbox test images. The deciding comparison is repeated on development sites only
  (`diag_devonly.py`, section 5). Method components chosen after 2026-11-01 (Ditto, tracking
  configuration, unknown flag, lambda rule) used development sites only.

## 2. Frozen method
| Component | Specification |
|---|---|
| Features | BioCLIP-2 (ViT-L/14) image embedding of the MegaDetector v5a animal crop: best animal box with conf >= 0.2, +10% margin, square in pixels, clipped; otherwise the full image's centre square. Frozen backbone. |
| Global head | FedAvg linear head on features x10; 150 rounds, 20 clients/round, 2 local epochs, LR 0.5, batch 64; zero init. |
| Personalisation | Ditto per site from its own delayed feedback: refit every 50 newly arrived labelled known-class images, 100 SGD steps, LR 0.1, proximal weight lambda. |
| lambda | Selected federatedly per labelling rate r: grid {0.1, 0.3, 1, 3, 10}; criterion = prequential log-loss of the head in force on each development site's held-out cal+tune block, replayed as a stream at rate r; image-weighted mean over sites; argmin. (Selected values: see section 5.) |
| Known-class score | RAPS; (lambda_RAPS, k_reg) tuned on tune by the existing grid rule. |
| Known-class threshold | tau_0 = pooled conformal quantile on cal at alpha = 0.1 (global head scores). Per-site online quantile tracking from delayed feedback, eta* chosen by tune replay (pinball loss) from {0.005, 0.02, 0.05}. |
| Unknown flag | Novelty n = 1 - max prob of the head in force; flag "unknown" iff n > nu, nu fixed = pooled conformal quantile over cal unknowns of all sites (global head), alpha_u = 0.2. No online update. |
| Feedback | Uniform labelling of sequences at rate r in {0.1, 0.3, 1.0}; labels arrive after a delay of 5 sequences; an unknown image is labelled "not a known species". |

## 3. Comparators (all on the same features, splits, feedback draws)
1. Global FedAvg head + static pooled threshold (and label-shift weighted threshold).
2. Global head + per-site tracking (no personalisation).
3. Ditto + fixed tau_0 (personalisation, no tracking).
4. Static frontier of 3 (fixed thresholds at alpha in {0.1, 0.095, ..., 0.07, 0.06, 0.05, 0.03}) for the matched-size comparison.
5. (Development-only ablations, NOT run on the lockbox: local-only heads; fine-tune-from-global
   without proximal term; reported from development runs with whole-image features.)
6. Literature baselines (`baselines_lit.py`, `qq_only.py`; global head, static, known classes only):
   - FCP: pooled conformal quantile over all sites (Lu et al., ICML 2023); identical to comparator 1.
   - Label-shift weighted quantile (Plassier et al., ICML 2023).
   - Quantile-of-quantiles (FedCP-QQ idea, Humbert et al., ICML 2023): sites send local 0.9-quantiles,
     server takes the k-th smallest, k by Monte-Carlo coverage under exchangeability with the actual
     cal sizes (our implementation of the idea; beta fixed at 1 - alpha after a beta search proved
     unstable on development data, k = 4..126 across seeds).
   - Density-ratio weighted CP for the target site (Tibshirani et al. 2019), the shared core of PFCP
     (Min et al., NeurIPS 2025) and PFWCP (Vejling et al. 2026); OUR ADAPTATION, not their algorithms;
     logistic site-vs-rest ratio, 2-fold cross-fitted by sequence, non-private (server sees features).
   - Online literature methods are covered by comparator 2 (per-site quantile tracking = ACI-style,
     with delayed feedback as in delayed ACI; importance-weighted variant = IM-OCP-style, tested in
     development with no gain under selective labelling).
   Development results (5 seeds, %sites sig below / size): FCP 30% / 3.87; label-shift 30% / 5.39;
   QQ 37% / 3.29; density-ratio 28% / 4.20.

## 4. Endpoints and analysis (lockbox test partitions only)
- **Primary:** share of lockbox sites whose known-class coverage is significantly below 0.9
  (one-sided, sequence-cluster bootstrap upper bound, level 0.95), with mean set size; full method
  vs comparator 1 (headline) and vs comparator 3 at matched size (frontier interpolation) for the
  tracking contribution. Reported per r, seen and unseen sites separately, mean over seeds 0-4
  with per-seed values.
- **Secondary:** mean coverage; share of 200-image windows below 0.85; unknown recall (pooled and
  share of sites with >= 20 unknowns significantly below 0.8); false-flag rate on known images;
  novelty AUROC; joint coverage.
- **Expected from development (stated in advance):** tracking gains at r = 1.0, marginal at
  r = 0.3, none at r = 0.1; no gain for unseen sites; personalisation lowers false flags.
- With 43 lockbox sites (10 unseen), one site = 2.3 points (10 for unseen): differences under
  about 5 points will be reported as not distinguishable.

## 5. Values fixed by the rules above (filled in from development runs)
- Backbone re-check on development-site test only (`logs/diag_devonly.log`, FedAvg, pooled LAC,
  5 seeds): whole image acc 0.665 / size 4.11 / MSP AUROC 0.746; crop 0.722 / 2.90 / 0.799;
  fine-tuned 0.669 / 3.91 / 0.746. Crop best in every seed: decision unchanged without the lockbox.
- lambda*(r), applied per seed exactly as selected (`results/pers_lambda_bioclip2crop.json`):
  r = 0.1: [1, 3, 10, 3, 3]; r = 0.3: [1, 3, 3, 3, 3]; r = 1.0: [1, 1, 1, 1, 1] (seeds 0-4).
  Development test sensitivity is small for lambda in [0.3, 3] (known-class %sig within ~2 points);
  lambda = 0.1 drops unknown recall below 0.8 at r = 1.0 (0.782), since nu is fixed on the global head.

- Dry run of `src/lockbox_eval.py dev` (`logs/final_eval_dev.log`, `results/final_eval_dev.json`)
  reproduces every development number: FCP 30% / 3.87; label-shift 30% / 5.39; QQ 37% / 3.29;
  density-ratio 28% / 4.20; method at r = 0.1 / 0.3 / 1.0: 27% / 4.43, 19% / 4.25, 11% / 4.00;
  method minus Ditto frontier at matched size: +1.9, -2.4, -4.3 points.
- The lockbox evaluation runs exactly `python src/lockbox_eval.py lockbox` (same code path as the
  dry run, site mask inverted). If it crashes, the cause is fixed without changing any method
  setting, the crash and fix are disclosed, and the run is repeated.

## 6. Development findings that motivated the design (reported, not re-tested)
Crops +5 pts accuracy and ~25% smaller sets vs whole image; partial fine-tuning +0.4 pts (dropped);
selective (uncertainty) labelling no gain; empirical-Bayes / persistence thresholds dominated by
pooled; shared drift component negligible; online tracking of the unknown threshold no gain;
detector-conditioned and text zero-shot novelty scores worse than MSP.

File hashes at freeze: `splits/prereg_freeze.json` (a file cannot contain its own hash).
