import argparse
import csv
import inspect
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.dataset import NPY_datasets
from engine import _binary_metrics
from models.egeunet import EGEUNet
from utils import myNormalize, myResize, myToTensor


class PH2Config:
    def __init__(self, training_dataset):
        self.test_transformer = transforms.Compose(
            [myResize(256, 256), myToTensor(), myNormalize(training_dataset, train=False)]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trusted ISIC-trained HBR-EGE-UNet checkpoint on PH2."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-dataset", choices=["isic17", "isic18"], default="isic17")
    parser.add_argument("--data-root", type=Path, default=Path("data/ph2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model():
    return EGEUNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        bridge=True,
        gt_ds=True,
        use_high_level_large_kernel=True,
        use_decoder_brr=True,
        hllk_kernel_size=7,
        use_brr_region_branch=True,
        use_brr_boundary_branch=True,
    )


def load_trusted_state_dict(path):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    else:
        warnings.warn(
            "This PyTorch version cannot restrict unpickling. Only load a checkpoint you trust.",
            RuntimeWarning,
        )
    checkpoint = torch.load(str(path), **load_kwargs)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    return state


def final_output(model_output):
    return model_output[-1] if isinstance(model_output, tuple) else model_output


def evaluate(model, loader, device, threshold):
    model.to(device).eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="PH2", unit="batch"):
            images = images.to(device, non_blocking=True).float()
            output = final_output(model(images))
            predictions.append(output.squeeze(1).cpu().numpy().astype(np.float32))
            targets.append(masks.squeeze(1).numpy().astype(np.float32))
    return _binary_metrics(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        threshold,
    )


def write_result(metrics, output_path, training_dataset, samples):
    row = {
        "model": "HBR-EGE-UNet",
        "training_dataset": training_dataset.upper(),
        "external_test_dataset": "PH2",
        "samples": samples,
        "threshold": metrics["threshold"],
        "foreground_IoU": metrics["miou"],
        "DSC": metrics["dsc"],
        "HD95": metrics["hd95"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "accuracy": metrics["accuracy"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return row


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = NPY_datasets(args.data_root, PH2Config(args.training_dataset), train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model()
    model.load_state_dict(load_trusted_state_dict(checkpoint))
    metrics = evaluate(model, loader, device, args.threshold)

    output_dir = args.output_dir or Path("results") / f"ph2_external_{args.training_dataset}"
    row = write_result(
        metrics,
        output_dir / "metrics.csv",
        args.training_dataset,
        len(dataset),
    )
    print(
        f"PH2: IoU={row['foreground_IoU']:.6f}, DSC={row['DSC']:.6f}, "
        f"HD95={row['HD95']:.6f}, sensitivity={row['sensitivity']:.6f}, "
        f"specificity={row['specificity']:.6f}"
    )
    print(f"Metrics written to: {(output_dir / 'metrics.csv').resolve()}")


if __name__ == "__main__":
    main()
