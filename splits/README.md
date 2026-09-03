# Dataset split manifests

Each text file contains one prepared image stem per line. The manifests record the exact local partitions used in the working experiments without redistributing images or masks.

- `isic2017_train.txt`: 1,500 training samples
- `isic2017_val.txt`: 650 validation samples
- `isic2018_train.txt`: 1,886 training samples
- `isic2018_val.txt`: 808 validation samples
- `ph2_val.txt`: 200 external-evaluation samples

The PH2 list is an external-test manifest, not a training split.

ISIC 2017 and PH2 retain their dataset case identifiers. ISIC 2018 was distributed in the prepared EGE-UNet layout with split-local numeric filenames; consequently, its two manifests are integrity checks for that prepared layout and do not provide a mapping to the original ISIC Archive identifiers.
