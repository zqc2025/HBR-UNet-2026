# HBR-UNet

Official research code for **HBR-UNet: High-Level Context Enhancement and
Boundary--Region Refinement for Lightweight Skin Lesion Segmentation**.

HBR-UNet is built on the EGE-UNet backbone and introduces two complementary
components:

- **High-Level Large-Kernel (HLLK) enhancement** at encoder stages E4--E6. It
  combines channel expansion, a 7x7 depthwise convolution, channel projection,
  and residual fusion to enlarge high-level contextual coverage.
- **Boundary-Region Refinement (BRR)** after GAB3, GAB2, and GAB1. Its region
  and boundary branches generate signed residual corrections controlled by
  zero-initialized learnable scaling factors.

The default configuration is the full HBR-UNet model reported in the
manuscript: 0.099 M parameters and 0.087 GFLOPs for a 1x3x256x256 input.

This repository contains no dataset images, masks, model checkpoints,
prediction images, full training logs, local paths, manuscript figures, or
exploratory experiment records.

## Repository structure

```text
HBR-UNet-2026/
├── configs/                  # Training configuration and module switches
├── data/                     # Dataset layout instructions only
├── datasets/                 # Image-mask loader with pairing validation
├── models/
│   ├── hbr_unet.py           # Public HBR-UNet entry point
│   └── egeunet.py            # Backbone plus HLLK and BRR implementation
├── scripts/                  # Data preparation and integrity checks
├── splits/                   # Prepared public-dataset split identifiers
├── engine.py                 # Training/evaluation loops and metrics
├── evaluate_ph2_external.py  # Direct PH2 external evaluation
├── train.py                  # Training entry point
├── utils.py                  # Losses, transforms, logging, and utilities
├── CITATION.cff
├── requirements.txt
├── NOTICE
└── LICENSE
```

The GitHub repository retains its original URL slug for stable links, while
the model and manuscript name are **HBR-UNet**.

## Environment

The reference environment follows the original EGE-UNet release:

- Python 3.8
- PyTorch 1.8.0
- torchvision 0.9.0
- NVIDIA GeForce RTX 3080 for the reported experiments

Install a PyTorch build compatible with the local CUDA toolkit, then install
the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset preparation

Download ISIC 2017, ISIC 2018, and PH2 from their official sources and comply
with their respective terms. Dataset files are not redistributed here.

- ISIC Challenge Archive: https://challenge.isic-archive.com/data/
- PH2 database: https://www.fc.up.pt/addi/ph2%20database.html

The manuscript follows the EGE-UNet 70/30 partitions:

| Dataset | Training | Evaluation |
| --- | ---: | ---: |
| ISIC 2017 | 1,500 | 650 |
| ISIC 2018 | 1,886 | 808 |
| PH2 | 0 | 200 external-test images |

Arrange preprocessed 256x256 images and masks as described in
[`data/README.md`](data/README.md). The ISIC 2018 manifests use the numeric
filenames in the prepared EGE-UNet data package and are not a mapping to
official ISIC Archive identifiers.

Validate the local layout before training:

```bash
python scripts/check_data_layout.py --dataset isic17
python scripts/check_data_layout.py --dataset isic18
python scripts/check_data_layout.py --dataset ph2
```

## Training

The default configuration trains the full HBR-UNet on ISIC 2017 for 300
epochs, using AdamW, an initial learning rate of 0.001, weight decay of 0.01,
CosineAnnealingLR, batch size 8, input size 256x256, and inference threshold
0.55.

```bash
python train.py
```

Linux/macOS examples:

```bash
HBR_DATASET=isic18 HBR_SEED=42 python train.py
HBR_DATASET=isic17 HBR_USE_HLLK=0 HBR_USE_BRR=0 python train.py
HBR_DATASET=isic17 HBR_HLLK_KERNEL_SIZE=5 python train.py
HBR_DATASET=isic17 HBR_BRR_USE_REGION=1 HBR_BRR_USE_BOUNDARY=0 python train.py
```

PowerShell example:

```powershell
$env:HBR_DATASET = "isic17"
$env:HBR_SEED = "42"
$env:HBR_USE_HLLK = "1"
$env:HBR_USE_BRR = "1"
python train.py
```

The legacy `EGE_*` environment-variable names remain accepted for backward
compatibility, but new runs should use `HBR_*`. Generated checkpoints,
TensorBoard files, logs, predictions, and result tables are written under
`results/` and excluded from Git.

## Loss function

The implementation matches the manuscript:

- final and deep-supervision predictions use `0.8 * BCE + 1.2 * Dice`;
- deep-supervision weights are `(0.1, 0.2, 0.3, 0.4, 0.5)`;
- BRR region and automatically generated 3x3 morphological-boundary targets
  use scale weights `(0.01, 0.02, 0.03)`.

No manually annotated boundary labels are required.

## PH2 external evaluation

Prepare PH2 locally:

```bash
python scripts/prepare_ph2_256.py --raw-root "path/to/PH2Dataset/PH2 Dataset images"
```

Evaluate a trusted HBR-UNet state-dict checkpoint without training,
fine-tuning, threshold selection, or model selection on PH2:

```bash
python evaluate_ph2_external.py \
  --training-dataset isic17 \
  --checkpoint path/to/best.pth
```

The output contains aggregate metrics only and does not store checkpoint paths
or source image names.

## Metric definitions

For binary lesion segmentation, the manuscript uses the label `mIoU` for the
foreground intersection over union
`TP / (TP + FP + FN)`. DSC, sensitivity, specificity, and accuracy are computed
from the same dataset-level pixel confusion matrix. HD95 is computed per image
in pixel units and then averaged. These exact definitions should be considered
when comparing values produced by other implementations.

## Reproducibility and release notes

- The main comparisons in the manuscript use three independent runs under the
  same settings; set `HBR_SEED` explicitly for each run.
- The fixed inference threshold is 0.55.
- PH2 is used only for direct external testing.
- HLLK and BRR can be disabled independently for ablation experiments.
- Only load checkpoints obtained from trusted sources.

## Acknowledgment and license

This project is derived from the public
[EGE-UNet](https://github.com/JCruan519/EGE-UNet) implementation. The upstream
source and this derivative distribution retain the Apache License 2.0.
Modified files carry notices and the principal changes are summarized in
[`NOTICE`](NOTICE).

Please cite the original EGE-UNet paper when using this derivative code. The
`CITATION.cff` file contains the manuscript citation metadata for HBR-UNet and
can be updated with the journal DOI after publication.
