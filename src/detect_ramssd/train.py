"""
src/detect_ramssd/train.py
---------------------------
YOLOv8n/s fine-tuning for RAM/SSD detection (Dataset 3 or fallback).

IMPORTANT: Trains ONLY on Dataset 3 / fallback. Never merge with Dataset 2.
Extra augmentation is enabled because this is a small dataset (~188 images).

Usage:
    python -m src.detect_ramssd.train \
        --dataset  data/processed/ramssd_detect/dataset.yaml \
        --model    yolov8n.pt \
        --epochs   100
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.seed import set_seed

SEED = 42


def train(
    dataset_yaml: Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,       # smaller default because dataset is tiny
    patience: int = 20,
    output_dir: Path = Path("reports/runs/detect_ramssd"),
    amp: bool = True,
    workers: int = 4,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    set_seed(SEED)

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Run: python -m src.detect_ramssd.prepare"
        )

    fallback_flag = (dataset_yaml.parent / "FALLBACK_USED.txt").exists()
    dataset_key = "dataset3_fallback" if fallback_flag else "dataset3"
    print(f"[detect_ramssd/train] Using: {dataset_key}")
    print(f"[detect_ramssd/train] Model: {model_name}, Epochs: {epochs}, Batch: {batch}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(output_dir),
        name="train",
        amp=amp,
        workers=workers,
        seed=SEED,
        augment=True,      # essential for small dataset
        exist_ok=True,
    )

    best_pt = output_dir / "train" / "weights" / "best.pt"
    print(f"[detect_ramssd/train] Done. Best → {best_pt}")
    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/ramssd_detect/dataset.yaml"))
    parser.add_argument("--model",   type=str,  default="yolov8n.pt")
    parser.add_argument("--epochs",  type=int,  default=100)
    parser.add_argument("--batch",   type=int,  default=8)
    parser.add_argument("--output",  type=Path, default=Path("reports/runs/detect_ramssd"))
    args = parser.parse_args()
    train(args.dataset, args.model, args.epochs, batch=args.batch, output_dir=args.output)
