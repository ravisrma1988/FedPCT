# Pre-registration addendum: second-backbone robustness analysis

Written 2026-11-02, after both lockbox evaluations and BEFORE any DINOv2 feature, model or result
exists. Its SHA-256 is recorded in `splits/prereg_backbone.json` before the weights are downloaded.
**All results of this analysis are reported in the manuscript whatever they show.**

## Question
Do the method's gains depend on the BioCLIP-2 backbone? (BioCLIP-2 is biology-specific and its
training data may overlap with the evaluation datasets.)

## Backbone
DINOv2 ViT-L/14 (`facebook/dinov2-large`, Apache-2.0; self-supervised, no species labels or text).
Feature = the model's pooled CLS output (`pooler_output`, 1024-d), L2-normalised; input 224x224,
ImageNet mean/std. Same MegaDetector crops as BioCLIP-2 (`features/md_best.csv`,
`features/md_best_cct.csv`), same crop geometry. Frozen backbone.

## Fixed in advance (nothing tuned for DINOv2)
- Splits, lockbox site lists, unknown species: unchanged (built from BioCLIP-2 features; not rebuilt).
- Method and every hyper-parameter exactly as `PREREG.md` (head x10 input scaling, FedAvg, Ditto
  refit rule, RAPS, tracking, unknown flag). The pre-registered selection rules run on DINOv2
  features: RAPS and eta on tune, Ditto lambda by the federated prequential rule on development
  sites (`pers_lambda.py`).
- Evaluation: the frozen `lockbox_eval.py` functions, both datasets, seeds 0-4, all comparators.

## Reported
1. **Development sites** of iWildCam and CCT (primary for this analysis).
2. **Lockbox sites** of both datasets, labelled **post hoc, not confirmatory**: these sites were
   already used once for the confirmatory BioCLIP-2 evaluation.
For each: the full comparator table, method - FCP, method - Ditto frontier at matched size, unknown
flag metrics, and accuracy of the global head. The robustness claim holds if the method's
reduction of %sites-below versus FCP at r = 1.0 has the same sign in every seed on both datasets'
development sites; its size relative to BioCLIP-2 is reported, not judged.
