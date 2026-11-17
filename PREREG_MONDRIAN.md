# Pre-specification: detector-conditional thresholds (exploratory, development sites only)

Written 2026-11-03, after the post hoc animal-only analysis showed that coverage for animal classes
stays below target, and BEFORE any detector-conditional threshold has been computed. Hash recorded in
`splits/prereg_mondrian.json` before the run. Reported in the manuscript whatever the outcome.

## Question
Does tracking separate thresholds for frames with and without a detected animal raise coverage for
animal classes, compared with the single tracked threshold of FedPCT?

## Variant (FedPCT-D)
- Group of an image: MegaDetector v5a found an animal box with confidence >= 0.2 (the crop rule), or not.
  Known at prediction time; uses no labels.
- Each site keeps two thresholds, one per group. Start: pooled conformal quantile at alpha = 0.1 of the
  true-class scores of calibration images of that group (global classifier). Each threshold is updated
  with Eq. 3 using only the misses and image counts of its own group, with the same step size eta*
  as FedPCT, the same feedback draws, delay and personalized classifier.
- Everything else identical to FedPCT (features, Ditto with the selected lambda, RAPS, seeds).

## Evaluation
- Development sites only (iWildCam 100, CCT 59), five seeds, labelling rates 0.3 and 1.0.
- Primary for this exploratory analysis: share of sites flagged as significantly below 0.9 for animal
  classes and mean animal-class site coverage at r = 1.0, FedPCT-D versus FedPCT.
- Also reported: the same for all classes, observed share below 0.9, mean set size.
- Not confirmatory: no lockbox is left, so any gain is a development-site result.
