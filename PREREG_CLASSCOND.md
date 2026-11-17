# Pre-specification: class-conditional threshold tracking (FedPCT-CC), development sites

Written 2026-11-03, after the confirmatory lockbox evaluations, the post hoc animal-only analysis and the
negative detector-conditional analysis (PREREG_MONDRIAN.md), and BEFORE any class-conditional threshold
has been computed. The hash of this file and of the analysis script is recorded in
`splits/prereg_classcond.json` before the run. The outcome is reported whatever it is.

## Question
Does tracking one threshold for animal classes and one for non-animal classes (empty, car) at each site
raise coverage for animal classes towards 0.9, at an acceptable cost in set size, compared with the
single tracked threshold of FedPCT?

## Rationale
FedPCT controls the long-run miscoverage over all known-class images of a site, which empty frames
dominate. FedPCT-D grouped images by a detector output, which did not help. FedPCT-CC instead groups the
candidate classes by their label. Class c enters the set of image x when s(x, c) <= tau_g(c), the
threshold of the group of c. The image is covered when s(x, y) <= tau_g(y), so the miss of an image
depends only on the threshold of the group of its true class. Each group threshold is updated with Eq. 3
using the misses and counts of labelled images whose true class is in that group. Theorem 1 then
applies to the substream of animal images of each site separately (Mondrian, label-conditional
form), so long-run animal coverage at each site is controlled, not only coverage over all images.

## Method (FedPCT-CC)
- Groups: animal = every known class except `empty` and `car`; non-animal = `empty`, `car`.
- Start of each group threshold: pooled conformal quantile at alpha = 0.1 of the true-class scores of
  the calibration images (not the tune slice) whose true class is in that group, global classifier.
- Update: Eq. 3 per group, same step size eta* as FedPCT (no new tuning), same delay D = 5, same
  feedback draws, same personalized classifier (Ditto, selected lambda), same RAPS scores and seeds.
- Prediction set: {c : s(x, c) <= tau_g(c)}. Unknown-class flag unchanged.
- No other change, and no further variant will be tried.

## Evaluation
- Development sites only (iWildCam 100, CCT 59), five seeds, labelling rates 0.3 and 1.0.
  FedPCT is recomputed on the same streams as the reference.
- Primary: share of sites flagged as significantly below 0.9 for animal classes at r = 1.0
  (sequence bootstrap, 2,000 replicates, as in the manuscript).
- Secondary: mean animal-class site coverage, observed share below 0.9, all-class flagged share and
  coverage, coverage for non-animal classes, mean set size over all known-class images and over animal
  images, all at r = 0.3 and 1.0.

## Decision rule (fixed now)
FedPCT-CC is called SUPPORTED only if, at r = 1.0, on BOTH datasets:
1. the animal-class flagged share falls by at least 5 points relative to FedPCT, with a reduction in at
   least 4 of the 5 seeds, and
2. the mean set size over all known-class images rises by no more than 25% relative to FedPCT.
Otherwise it is NOT SUPPORTED, FedPCT stays the proposed method, and FedPCT-CC is reported in Online
Resource 1 as a negative result with one sentence in the Discussion.

## Lockbox
Both lockboxes were already used once for the confirmatory evaluation. If and only if FedPCT-CC is
SUPPORTED, it is applied once to both lockboxes with these settings and reported as a post hoc reuse,
not as a confirmatory result. If it is NOT SUPPORTED, no lockbox run is made.
