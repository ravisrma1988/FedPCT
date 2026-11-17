# Software environment

The experiments ran on one NVIDIA GPU server inside the NVIDIA NGC PyTorch container, release 25.03.
The versions below are those of that environment as used for the reported runs.

## Main pipeline (container 25.03)

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.7.0a0 (NGC build nv25.03) |
| torchvision | 0.22.0a0 (NGC build) |
| CUDA / cuDNN | 12.8 / 9.8.0 |
| NVIDIA DALI | 1.47.0 (whole-image feature extraction only) |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.2 |
| Pillow | 11.1.0 |
| OpenCLIP (open_clip_torch) | 3.3.0, model hf-hub:imageomics/bioclip-2, revision 2957b32 |

requirements.txt pins the packages that are installed on top of the container. The WILDS package is used only by
verify_iwildcam.py to compare the downloaded data with the WILDS v2.0 statistics, and its version was not recorded.

## MegaDetector environment

setup_md.sh creates a separate virtual environment that reuses the container's torch, torchvision and numpy and
installs megadetector 10.0.28 and ultralytics-yolov5 0.1.1 without dependencies, plus opencv-python-headless
4.11.0.86 and the light runtime packages listed in the script. The detector weights are md_v5a.0.0.pt from the
MegaDetector v5.0 release, and the script prints their SHA-256 hash.

## What runs without a GPU

reproduce_tables.py and verify_bundle.py need only the Python 3 standard library and the files of this archive.
Rebuilding features, training the federated heads and running the conformal methods needs the container above,
the public datasets and a CUDA GPU.
