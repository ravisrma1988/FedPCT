# Pre-registration addendum: external validation on Caltech Camera Traps (CCT)

Written 2026-11-02, after the CCT splits and lockbox were built and BEFORE any model, method or
baseline was run on CCT. Frozen together with the evaluation code (hashes in
`splits_cct/prereg_freeze.json`) before the CCT lockbox is evaluated, once.

## Purpose
External validation of the method frozen in `PREREG.md` (iWildCam). **No method component, setting
or hyper-parameter is changed for CCT.** Only the pre-registered selection rules run on CCT data:
RAPS parameters and tracking step eta on the tune slice, Ditto lambda by the federated prequential
rule on development sites' cal+tune blocks. CCT development results are a pipeline check (dry run)
and are not used to change anything.

## Data (`src/cct_meta.py`, `reports/cct_meta_report.json`)
- CCT (LILA BC, CDLA-Permissive 1.0): 243,100 images, 140 locations, 2010-2015.
- Label-free integrity rules: 77 multi-species photos dropped; 108 byte-identical copies with
  conflicting labels dropped, 19 extra exact copies dropped; 1 image with an out-of-range date
  dropped; no sequence spans two locations or more than one day. 242,895 images remain.
- Seen / unseen sites: CCT's official location-level train/val split.
- Clients: 84 locations with >= 250 images (60 seen, 24 unseen).

## Features
BioCLIP-2 on MegaDetector v5a crops, the same rule as iWildCam (conf >= 0.2, +10% margin, square);
boxes from LILA's published MDv5a results (`src/crop_features_cct.py`); 45.3% of images cropped.

## Splits (`src/splits_cct.py`, `logs/splits_cct.log`), frozen split code unchanged
- Time split per site by whole sequences (train 40% / cal 20% / test 40%), tune = 1/3 of cal.
- Unknown species: the iWildCam pool rule (20-200 train images, >= 3 train sites) had < 2 species,
  so the pre-set fallback (upper bound 2000) applied: pool of 9; N_UNK = 2 per seed.
- Near-duplicates: 7.3% of test images (seed 0) have a cos > 0.98 crop match in the same site's
  train partition (iWildCam: 0.51%). By the pre-set rule (> 2%), those test SEQUENCES are dropped
  (4,660-4,838 per seed). Outcome-blind diagnosis: 95% are animal images, 99% match the same class,
  median time gap 235 days; full-frame perceptual hashes are only slightly closer than random
  same-site same-class pairs (dHash <= 4: 16% vs 11%), i.e. mostly recurrence at a fixed camera, not
  re-saved copies. Consequence: the CCT test set is somewhat harder than iWildCam's.

## Lockbox (`splits_cct/lockbox_sites.json`)
25 of 84 sites (18 seen, 7 unseen), rng 20261102, CCT20 pilot locations excluded (they took part in
an early pilot). Drawn before any CCT result.

## Endpoints, comparators, analysis
Exactly as `PREREG.md` sections 3-4 (same script path: `src/lockbox_eval_cct.py`, which runs the
frozen `lockbox_eval.py` functions on CCT). With 25 lockbox sites one site = 4 points (14 for the 7
unseen sites): differences under about 8 points are reported as not distinguishable.

## Known limitations disclosed in advance
- MegaDetector v5 was trained on data including CCT (and WCS camera traps, the source of iWildCam):
  detection on both datasets is likely better than on new cameras (shared by all methods).
- BioCLIP-2's training data (TreeOfLife-200M, 617.8k GBIF camera-trap images) may overlap with
  either dataset; not verifiable from the paper.
- CCT classes are coarse (e.g. "bird", "rodent") and 52% of images are empty.
