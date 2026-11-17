# FedPCT: Online per-site coverage tracking for personalized federated conformal prediction

Code, split manifests, frozen protocols, logs and per-site results for the manuscript

> R. Sharma, R. R. Upadhyay, V. Gupta and G. Rathi, "Online Per-Site Coverage Tracking for Personalized Federated
> Conformal Prediction Under Drift and Delayed Partial Feedback", submitted to IEEE Access, 2026.

FedPCT gives per-site conformal prediction sets in a federated network whose data drift and whose expert labels
arrive late and only for part of the data. Each site personalizes a federated linear classifier on frozen image
features with a proximal penalty, tracks its own conformal (RAPS) threshold online from the delayed and partial
labels, and flags images of classes held out from training. The evaluation uses iWildCam v2.0 (WILDS), Caltech
Camera Traps and FMoW-WILDS with single-use lockbox test periods.

## Quick check without a GPU

```bash
python verify_bundle.py      # hashes of the distributed files and of the frozen protocol records
python reproduce_tables.py   # rebuilds the result tables of the manuscript from results/*.json
```

Both scripts need only the Python 3 standard library.

## Contents

| Folder | Contents |
|---|---|
| `src/` | all experiment code (data preparation, features, splits, federated training, conformal methods, FedPCT, lockbox evaluations, post hoc and exploratory analyses) |
| `PREREG*.md`, `THEORY.md` | frozen protocol documents and theory notes |
| `splits/`, `splits_cct/`, `splits_fmow/` | validated split manifests with hashes, lockbox site lists and freeze records |
| `splits_cct_keepdup/` | post hoc CCT splits with the near-duplicate test sequences retained |
| `results/` | per-seed and per-site results behind every table and figure |
| `logs/` | run logs, including the single lockbox evaluations |
| `reports/` | data-cleaning and integrity reports |

`SHA256_ORIGINAL.txt` lists the hashes of the files as they were run. Before release, local directory paths were
replaced by `${FEDPCT_ROOT}` or `${WORKDIR}`, and `SHA256_BUNDLE.txt` lists the hashes of the files as distributed.
The freeze records hash the original files, so `verify_bundle.py` compares them with `SHA256_ORIGINAL.txt`. Set
the root variables (or edit the `ROOT` constants) to the project directory before running the code.

Raw images, image features and model weights are not redistributed. They can be regenerated with the code from
the public releases of iWildCam v2.0 (WILDS), Caltech Camera Traps (LILA, labels of 13 January 2021) and
FMoW-WILDS. The split manifests contain record identifiers, labels, locations and timestamps from these datasets
and remain subject to their licences (CDLA-Permissive for Caltech Camera Traps, the WILDS terms for iWildCam and
the Functional Map of the World Challenge Public License, non-commercial with share-alike terms, for FMoW-WILDS).

## Licence

The code is released under the MIT licence (`LICENSE`). Split manifests and other data derived from the source
datasets remain under the licences of those datasets, as stated above.

## Software

See `ENVIRONMENT.md` and `requirements.txt`. The experiments ran in the NVIDIA NGC PyTorch container 25.03 on one
partition of an NVIDIA B200 GPU. MegaDetector v5a runs in its own environment (`src/setup_md.sh`).

## Order of the main scripts (each script's docstring gives its usage)

iWildCam
1. `get_iwildcam.sh`, `verify_iwildcam.py`, `check_mirror.py`: download and checks against WILDS v2.0
2. `extract_iwildcam.py`, `run_extract_dali.sh`: metadata table and whole-image features
3. `setup_md.sh`, `run_md.sh` (`md_run.py`): MegaDetector v5a
4. `crop_features.py`: BioCLIP 2 crop features (imageomics/bioclip-2, revision 2957b32, OpenCLIP 3.3.0)
5. `splits.py --seeds 5 --modes time seq`, `lockbox.py`: validated splits and the lockbox draw
6. `pers_lambda.py 5 bioclip2crop`: federated selection of the proximal weight
7. `freeze.py`, `lockbox_eval.py dev`, `lockbox_eval.py lockbox`: frozen protocol, dry run, single lockbox evaluation
8. `baselines_lit.py`, `sens_ablate.py`, `backbone_eval.py`, `online_baselines.py`, `online2.py`, `site_prior_dev.py`,
   `theory_check.py`, `theory_check2.py`, `mondrian_dev.py`, `classcond_dev.py`: baselines, ablations and secondary analyses

Caltech Camera Traps
1. `cct_fetch_meta.sh`, `cct_fetch_images.sh`, `cct_meta.py`: labels, detections, images and metadata table
2. `crop_features_cct.py`, `splits_cct.py`: crop features and splits with the frozen split code
3. `cct_run.py <script>` runs an unmodified iWildCam script on CCT, for example `cct_run.py pers_lambda.py 5 cct_bioclip2crop`
4. `freeze_cct.py`, `lockbox_eval_cct.py dev`, `lockbox_eval_cct.py lockbox`

FMoW-WILDS
1. `fmow_fetch.sh`, `fmow_meta.py`, `fmow_features.py`: data, metadata table and DINOv2 features (facebook/dinov2-large)
2. `splits_fmow.py`, `pers_lambda.py` (through `fmow_env.py`), `freeze_fmow.py`
3. `lockbox_eval_fmow.py dev`, `lockbox_eval_fmow.py lockbox`: single lockbox evaluation with the secondary trackers

Post hoc analyses of the used lockboxes (not confirmatory)
- `posthoc_animal.py`, `posthoc_sensitivity.py`: animal-only analysis, screening and per-class sensitivity
- `posthoc_review.py`: mean coverage shortfall and the static frontier of the global classifier at matched set size
- `cct_keepdup.py`: CCT lockbox with the near-duplicate test sequences retained
- `fmow_raps.py`: RAPS parameters selected on FMoW-WILDS
