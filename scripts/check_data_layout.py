import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_SUFFIXES = ("_segmentation", "_mask")
EXPECTED_SPLITS = {
    "isic17": {"train": "isic2017_train.txt", "val": "isic2017_val.txt"},
    "isic18": {"train": "isic2018_train.txt", "val": "isic2018_val.txt"},
    "ph2": {"val": "ph2_val.txt"},
}
DATA_DIRECTORIES = {"isic17": "isic2017", "isic18": "isic2018", "ph2": "ph2"}


def sample_id(path, is_mask=False):
    value = path.stem
    if is_mask:
        lowered = value.casefold()
        for suffix in MASK_SUFFIXES:
            if lowered.endswith(suffix):
                value = value[: -len(suffix)]
                break
    return value.casefold()


def collect(directory, is_mask=False):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    ids = [
        sample_id(path, is_mask=is_mask)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate sample IDs in {directory}")
    return set(ids)


def read_manifest(path):
    return {
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def main():
    parser = argparse.ArgumentParser(description="Validate data pairs against release manifests.")
    parser.add_argument("--dataset", required=True, choices=sorted(EXPECTED_SPLITS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-root", type=Path, default=Path("splits"))
    args = parser.parse_args()

    dataset_root = args.data_root / DATA_DIRECTORIES[args.dataset]
    for split, manifest_name in EXPECTED_SPLITS[args.dataset].items():
        split_root = dataset_root / split
        images = collect(split_root / "images")
        masks = collect(split_root / "masks", is_mask=True)
        expected = read_manifest(args.splits_root / manifest_name)
        if images != masks:
            raise RuntimeError(
                f"{args.dataset}/{split}: image-mask mismatch; "
                f"images without masks={sorted(images - masks)[:10]}, "
                f"masks without images={sorted(masks - images)[:10]}"
            )
        if images != expected:
            raise RuntimeError(
                f"{args.dataset}/{split}: local data differs from {manifest_name}; "
                f"missing={sorted(expected - images)[:10]}, extra={sorted(images - expected)[:10]}"
            )
        print(f"PASS {args.dataset}/{split}: {len(images)} matched pairs")


if __name__ == "__main__":
    main()

