"""
src/detect_pcb/train.py
------------------------
YOLOv8n/s fine-tuning for PCB part detection (Dataset 2).

IMPORTANT: This trains ONLY on Dataset 2. Never merge with Dataset 3.

Usage:
    python -m src.detect_pcb.train \
        --dataset  data/processed/pcb_detect/dataset.yaml \
        --model    yolov8n.pt \
        --epochs   100 \
        --output   reports/runs/detect_pcb
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
    batch: int = 16,
    patience: int = 20,
    output_dir: Path = Path("reports/runs/detect_pcb"),
    amp: bool = True,
    workers: int = 4,
) -> Path:
    """Train YOLOv8 on the PCB dataset. Returns path to best.pt."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    set_seed(SEED)

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Run: python -m src.detect_pcb.prepare"
        )

    print(f"[detect_pcb/train] Model:   {model_name}")
    print(f"[detect_pcb/train] Dataset: {dataset_yaml}")
    print(f"[detect_pcb/train] Epochs:  {epochs}, Batch: {batch}, AMP: {amp}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    results = model.train(
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
        exist_ok=True,
        augment=True,
    )

    best_pt = output_dir / "train" / "weights" / "best.pt"
    print(f"[detect_pcb/train] Done. Best weights → {best_pt}")
    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=Path, default=Path("data/processed/pcb_detect/dataset.yaml"))
    parser.add_argument("--model",    type=str,  default="yolov8n.pt")
    parser.add_argument("--epochs",   type=int,  default=100)
    parser.add_argument("--batch",    type=int,  default=16)
    parser.add_argument("--output",   type=Path, default=Path("reports/runs/detect_pcb"))
    parser.add_argument("--no_amp",   action="store_true")
    args = parser.parse_args()
    train(
        args.dataset, args.model, args.epochs,
        batch=args.batch, output_dir=args.output, amp=not args.no_amp
    )
