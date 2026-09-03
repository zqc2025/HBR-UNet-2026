# HBR-EGE-UNet

Minimal research code for **High-Level Boundary-Region Refinement for Lightweight Skin Lesion Segmentation**. The repository follows the public EGE-UNet layout and adds the high-level large-kernel (HLLK) block and decoder-stage boundary-region refinement (BRR).

This release intentionally contains no dataset images, masks, prediction images, model checkpoints, full training logs, local file paths, manuscript figures, or exploratory experiment ledgers.

## Repository structure

```text
HBR-EGE-UNet-open-source/
├── configs/                  # Training configuration and module switches
├── data/                     # Dataset instructions only
├── datasets/                 # Validated image-mask loader
├── models/                   # EGE-UNet with HLLK and BRR
├── scripts/                  # PH2 preparation and data-layout checks
├── splits/                   # Exact public-dataset split identifiers
├── engine.py                 # Training/evaluation loops and metrics
├── evaluate_ph2_external.py  # Proposed-model evaluation on PH2
├── train.py                  # Training entry point
├── utils.py                  # Losses, transforms, logging, and utilities
├── requirements.txt
├── NOTICE
└── LICENSE
```

## Environment

The reference environment follows the original EGE-UNet release:

- Python 3.8
- PyTorch 1.8.0
- torchvision 0.9.0

Install a PyTorch build compatible with the local CUDA toolkit, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Dataset preparation

Download ISIC 2017, ISIC 2018, and PH2 from their official sources and comply with their respective terms. No dataset files are redistributed here.

- ISIC Challenge Archive: https://challenge.isic-archive.com/data/
- PH2: https://www.fc.up.pt/addi/ph2%20database.html

Arrange preprocessed 256x256 images and masks as described in [`data/README.md`](data/README.md). Integrity manifests for the prepared splits used by the experiments are provided in [`splits/`](splits/README.md). The ISIC 2018 files follow the numeric naming used by the prepared EGE-UNet dataset package; they are not a mapping back to official ISIC Archive identifiers.

Validate the local layout before training:

```bash
python scripts/check_data_layout.py --dataset isic17
python scripts/check_data_layout.py --dataset isic18
python scripts/check_data_layout.py --dataset ph2
```

## Training

The default configuration trains HBR-EGE-UNet on ISIC 2017 with HLLK and BRR enabled and a fixed segmentation threshold of 0.55.

```bash
python train.py
```

Linux/macOS module-switch examples:

```bash
EGE_DATASET=isic18 EGE_USE_HLLK=1 EGE_USE_BRR=1 python train.py
EGE_DATASET=isic17 EGE_USE_HLLK=0 EGE_USE_BRR=0 python train.py
```

PowerShell example:

```powershell
$env:EGE_DATASET = "isic17"
$env:EGE_USE_HLLK = "1"
$env:EGE_USE_BRR = "1"
python train.py
```

Generated checkpoints, TensorBoard files, logs, predictions, and result tables are written under `results/` and are excluded from Git.

## PH2 external evaluation

Prepare PH2 locally:

```bash
python scripts/prepare_ph2_256.py --raw-root "path/to/PH2Dataset/PH2 Dataset images"
```

Evaluate a trusted HBR-EGE-UNet state-dict checkpoint without fine-tuning:

```bash
python evaluate_ph2_external.py \
  --training-dataset isic17 \
  --checkpoint path/to/best.pth
```

The evaluation output contains aggregate metrics only. It does not store the checkpoint path or source image names.

## Metric definition

`mIoU` in this implementation is the foreground IoU computed from the dataset-level pixel confusion matrix. `DSC`, sensitivity, specificity, and accuracy are calculated from the same aggregate confusion matrix. `HD95` is calculated per image and then averaged. These definitions should be used when comparing the reported values with other implementations.

## Reproducibility notes

- The internal ISIC experiments use the fixed prepared-file manifests in `splits/`.
- PH2 is used only for direct external evaluation and is not used for training, model selection, threshold selection, or fine-tuning.
- The fixed inference threshold is 0.55.
- Only checkpoints from trusted sources should be loaded.

## Acknowledgment and license

This project is derived from the public [EGE-UNet](https://github.com/JCruan519/EGE-UNet) implementation. The upstream source and this derivative distribution retain the Apache License 2.0. Modified files carry notices and the principal changes are summarized in [`NOTICE`](NOTICE).

Please cite the original EGE-UNet paper when using this code. Citation details for the HBR-EGE-UNet manuscript can be added after publication.
