# PH2 preparation utility for HBR-EGE-UNet. It does not redistribute PH2 data.

import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare PH2 images and lesion masks at a fixed resolution."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/ph2/raw/PH2Dataset/PH2 Dataset images"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ph2/val"),
    )
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def find_single(case_dir, folder_suffix, filename):
    matches = list(case_dir.glob(f"*{folder_suffix}/{filename}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one match for {case_dir.name}/{filename}, found {len(matches)}"
        )
    return matches[0]


def main():
    args = parse_args()
    image_dir = args.output_root / "images"
    mask_dir = args.output_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(path for path in args.raw_root.iterdir() if path.is_dir())
    if len(case_dirs) != 200:
        raise RuntimeError(f"Expected 200 PH2 cases, found {len(case_dirs)}")

    for case_dir in case_dirs:
        case_id = case_dir.name
        image_path = find_single(
            case_dir,
            "_Dermoscopic_Image",
            f"{case_id}.bmp",
        )
        mask_path = find_single(
            case_dir,
            "_lesion",
            f"{case_id}_lesion.bmp",
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(
                (args.size, args.size),
                Image.Resampling.BILINEAR,
            )
            image.save(image_dir / f"{case_id}.png", format="PNG")

        with Image.open(mask_path) as mask:
            mask = mask.convert("L").resize(
                (args.size, args.size),
                Image.Resampling.NEAREST,
            )
            mask = mask.point(lambda value: 255 if value > 0 else 0, mode="1")
            mask.convert("L").save(mask_dir / f"{case_id}.png", format="PNG")

    print(f"Prepared {len(case_dirs)} image-mask pairs in {args.output_root}")


if __name__ == "__main__":
    main()
