# Pre-registration addendum: second-domain validation on FMoW-WILDS (satellite images)

Written 2026-11-15, after the FMoW metadata table was built and BEFORE any FMoW split, model, method
or baseline result existed. Section 6 is completed from the split build (label-free and count-only
facts) before any model is trained on FMoW. Frozen together with the evaluation code (hashes in
`splits_fmow/prereg_freeze.json`) before the FMoW lockbox is evaluated, once.

## 1. Purpose
Confirmatory test of the method frozen in `PREREG.md` in a second domain with a fresh, never-used
lockbox. The camera-trap lockboxes of iWildCam and CCT were spent on 2026-11-02. **No method
component, setting, grid or hyper-parameter rule is changed for FMoW**, with one exception forced by
the domain: the image encoder (section 3). The pre-registered selection rules run on FMoW data
exactly as before: RAPS parameters and the tracking step on the tune slice, the Ditto weight by the
federated prequential rule on development sites. FMoW development results are a pipeline check and
are not used to change anything.

## 2. Data (`src/fmow_meta.py`, `reports/fmow_meta_report.json`)
- FMoW-WILDS v1.1 (Koh et al. 2021; Christie et al. 2018), 523,846 RGB satellite images, 224x224,
  62 building and land-use categories, 2002-2017, 207 countries. Downloaded 2026-11-15 from the
  Hugging Face mirror named in the WILDS code; archive size matched the published 53,893,501,163 bytes
  and passed the gzip integrity check.
- Licence: FMoW Challenge Public License (non-commercial, attribution, share-alike). Results may be
  published; any shared derived data (for example features) carries the same terms. The code and
  data archive will therefore contain only split manifests (image identifiers), not images or features.
- **Site** = country (`country_code`). Sites with >= 250 images are kept (the frozen rule): 105
  countries, 98.6% of the images.
- **Sequence** = object instance (the `img_path` folder). FMoW revisits the same object over years.
  Metadata inspection (no model) showed that with image-level partitions 67% of test images would
  have another view of the same object in the same country's train partition. Whole objects are
  therefore partitioned, ordered by their first view, exactly as camera-trap sequences are ordered by
  their start time. The sequence bootstrap of the endpoint resamples whole objects, and the feedback
  delay of 5 counts objects.
- **Seen / unseen sites:** FMoW has no site-level split. 26 of the 105 countries (25%, stratified by
  UN region, rng 20261115) are unseen and contribute no training data.

## 3. Features (the one domain-forced change)
BioCLIP 2 is a biology encoder and MegaDetector crops do not apply to satellite images. The encoder
is DINOv2 ViT-L/14 (`facebook/dinov2-large`, pooled CLS output, 1024-d, L2-normalised), the second
backbone already used in the camera-trap robustness analysis (`PREREG_BACKBONE.md`), applied to the
whole 224x224 image with ImageNet normalisation (`src/fmow_features.py`). Label-free.

## 4. Splits (`src/splits_fmow.py`), frozen split code unchanged
- Time split per site by whole objects (train 40% / cal 20% / test 40% of objects by first view);
  tune = 1/3 of cal objects; exact feature duplicates removed by `splits.clean()`; hard assertions.
- Held-out classes: N_UNK = 4 per seed. Pool = the iWildCam rule (20-200 train images, >= 3 train
  sites, seen sites only), with the upper bound raised to 2,000, then 20,000, then removed, in that
  order, until at least 4 classes qualify (`src/fmow_env.py`).
- Near-duplicates: the CCT rule. If more than 2% of test images have a cos > 0.98 match among the same
  site's train images, those test objects are dropped (label-free) and the split is re-validated.
- Lockbox: 30% of sites per seen/unseen stratum, rng 20261115, drawn in `splits_fmow.py` before any
  FMoW model result. Only their test partitions are held out, as for iWildCam and CCT.

## 5. Hypotheses, endpoints, comparators and analysis
Endpoints, comparators and statistics are those of `PREREG.md` sections 3-4, computed by the same
functions (`src/lockbox_eval_fmow.py` runs the frozen `lockbox_eval.py` on FMoW).
- **H1 (primary):** with full feedback (r = 1.0), the share of lockbox sites significantly below 0.9
  is lower for FedPCT than for FCP (mean over seeds 0-4), and lower in every seed.
- **H2:** at r = 0.3, FedPCT's share is lower than that of each of the four static federated
  baselines (FCP, label-shift, quantile-of-quantiles, density-ratio).
- **H3:** at sites never seen in training, the share is lower for FedPCT than for FCP at r = 1.0.
- **Secondary, pre-specified:** the stronger online trackers of `src/online_baselines.py`
  (decaying-step ACI, DtACI, conformal PID integrator), each with the global head and with the Ditto
  head, hyper-parameters chosen on the tune replay by the same pinball rule, on the same sites, seeds
  and labelling draws; reported with set sizes, no hypothesis attached.
- Every hypothesis is reported whatever the outcome. A failure of H1 is reported as a failure of the
  method in this domain. With about 31 lockbox sites one site moves the endpoint by about 3 points
  (more for unseen sites): differences under 6 points are reported as not distinguishable.
- **Expected (stated in advance, from the camera-trap results):** gains at r = 1.0, smaller at
  r = 0.3, none at r = 0.1; a smaller tracking contribution if drift between calibration and test is
  mild.

## 6. Facts fixed by the split build (filled in before any model is trained on FMoW)
From `logs/splits_fmow.log` and `splits_fmow/time_seed*.json`, recorded 2026-11-15 before any model
was trained on FMoW.
- Cleaning (`splits.clean()`): 523,846 rows in; 28 rows of 12 images with conflicting labels on
  identical features dropped; 254 exact duplicate rows dropped; 523,564 rows out.
- 105 sites (79 seen, 26 unseen); 58 known classes per seed.
- Held-out classes: the iWildCam pool range (20-200) had fewer than 4 classes, so the first fallback
  (20-2,000) applied in every seed, pool of 13 classes. Held out per seed (category indices):
  seed 0 [4, 7, 45, 61]; seed 1 [0, 27, 31, 61]; seed 2 [0, 5, 18, 44]; seed 3 [4, 5, 44, 50];
  seed 4 [25, 27, 44, 45].
- Near-duplicates: 0.05% of test images (seed 0) have a cos > 0.98 match in the same site's train
  partition, below the 2% rule, so no test object was dropped (iWildCam 0.51%, CCT 7.3%).
- Test images per seed: about 78,000 (seed 0: 77,110 known-class and 1,334 held-out-class images),
  about 17% of the retained images, because objects with many views tend to be first imaged early.
- Lockbox (`splits_fmow/lockbox_sites.json`): 32 of 105 sites (24 seen, 8 unseen). One lockbox site
  moves the endpoint by 3.1 points (12.5 for the unseen sites). Known-class test images per lockbox
  site: median 158 (range 25 to 12,737); development sites: 73 (18 unseen), median 190.
- Implementation note: FMoW timestamps are stored with and without milliseconds, so
  `fmow_env.use_fmow()` parses the loaded `datetime` column with the ISO 8601 parser. The split files
  and their hashes are unchanged.

## 7. Known limitations disclosed in advance
- Countries are a coarse notion of site, and FMoW images were collected for a benchmark, not by a
  monitoring programme; delayed partial feedback is simulated as for the camera traps.
- DINOv2's training data (LVD-142M) are curated web images; overlap with FMoW cannot be verified.
- Site sizes are very unequal (253 to about 102,000 images per country).
