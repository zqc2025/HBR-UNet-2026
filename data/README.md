# Dataset layout

Dataset images and masks are not included in this repository. Create the following directories locally:

```text
data/
├── isic2017/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── isic2018/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
└── ph2/
    └── val/
        ├── images/
        └── masks/
```

Supported extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`. Image and mask IDs must match. For ISIC 2017, a mask suffix such as `_segmentation` is accepted automatically.

Use the identifier lists under `../splits/` to reconstruct the exact train/validation partitions. Do not commit downloaded or preprocessed dataset files; `.gitignore` excludes everything under `data/` except this document.

