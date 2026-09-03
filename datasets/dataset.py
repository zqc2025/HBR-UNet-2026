# Derived from EGE-UNet (https://github.com/JCruan519/EGE-UNet),
# licensed under Apache-2.0. Modified to validate image-mask pairing and
# ignore non-image placeholder files. See NOTICE.

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_SUFFIXES = ("_segmentation", "_mask")


def _sample_id(path, is_mask=False):
    sample_id = path.stem
    if is_mask:
        lowered = sample_id.casefold()
        for suffix in MASK_SUFFIXES:
            if lowered.endswith(suffix):
                sample_id = sample_id[: -len(suffix)]
                break
    return sample_id.casefold()


def _index_images(directory, is_mask=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    indexed = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        sample_id = _sample_id(path, is_mask=is_mask)
        if sample_id in indexed:
            raise RuntimeError(
                f"Duplicate sample ID '{sample_id}' in {directory}: "
                f"{indexed[sample_id].name}, {path.name}"
            )
        indexed[sample_id] = path
    if not indexed:
        raise RuntimeError(f"No supported image files found in {directory}")
    return indexed


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super().__init__()
        split = "train" if train else "val"
        split_root = Path(path_Data) / split
        images = _index_images(split_root / "images")
        masks = _index_images(split_root / "masks", is_mask=True)

        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise RuntimeError(
                "Image-mask IDs do not match. "
                f"Missing masks: {missing_masks[:10]}; "
                f"missing images: {missing_images[:10]}"
            )

        self.data = [(images[sample_id], masks[sample_id]) for sample_id in sorted(images)]
        self.transformer = config.train_transformer if train else config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        with Image.open(img_path) as image:
            img = np.array(image.convert("RGB"))
        with Image.open(msk_path) as mask:
            msk = np.expand_dims(np.array(mask.convert("L")), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
