import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SPECS = {
    "isic2017_train.txt": ("isic2017", "train"),
    "isic2017_val.txt": ("isic2017", "val"),
    "isic2018_train.txt": ("isic2018", "train"),
    "isic2018_val.txt": ("isic2018", "val"),
    "ph2_val.txt": ("ph2", "val"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Export sample-ID manifests from an already prepared local dataset tree."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, (dataset, split) in SPECS.items():
        image_dir = args.data_root / dataset / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        identifiers = sorted(
            path.stem
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if not identifiers:
            raise RuntimeError(f"No supported images found in {image_dir}")
        output_path = args.output_dir / filename
        output_path.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
        print(f"Wrote {len(identifiers)} IDs: {output_path}")


if __name__ == "__main__":
    main()

