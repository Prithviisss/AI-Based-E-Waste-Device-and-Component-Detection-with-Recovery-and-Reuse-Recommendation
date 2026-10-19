"""
train_all.py
------------
Master training runner for E-Waste RecoveryAI.
Orchestrates preprocessing, training, and evaluation for:
  1. EfficientNet-B0 Device Classifier (Dataset 1)
  2. YOLOv8 PCB Component Detector (Dataset 2)
  3. YOLOv8 RAM/SSD Component Detector (Dataset 3)

Usage:
    # Run full training pipeline
    python train_all.py --all

    # Run specific module
    python train_all.py --module classify
    python train_all.py --module pcb
    python train_all.py --module ramssd

    # Quick dry run / smoke test with low epochs
    python train_all.py --all --epochs-classify 1 --epochs-yolo 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*70}")
    print(f"▶ RUNNING: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"\n❌ FAILED: {desc} (Exit code {ret.returncode})", file=sys.stderr)
        sys.exit(ret.returncode)
    print(f"\n✔ COMPLETED: {desc}\n")


def train_classifier(epochs: int, batch_size: int, lr: float, skip_dedupe: bool = False) -> None:
    print("\n--- [STAGE 1/3] Device Classifier (EfficientNet-B0) ---")
    raw_dir = Path("data/raw/dataset1_classification")
    deduped_dir = Path("data/processed/classification/deduped")
    split_dir = Path("data/processed/classification/split")

    if not skip_dedupe:
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            print(f"⚠ Warning: {raw_dir} is empty or missing. Please place Kaggle dataset 1 there.")
        run_cmd(
            [sys.executable, "-m", "src.classify.dedupe", "--input", str(raw_dir), "--output", str(deduped_dir)],
            "Deduplicating Dataset 1 (pHash / dHash)",
        )
        run_cmd(
            [sys.executable, "-m", "src.classify.split", "--input", str(deduped_dir), "--output", str(split_dir)],
            "Stratified 70/15/15 Split for Dataset 1",
        )

    run_cmd(
        [
            sys.executable, "-m", "src.classify.train",
            "--data_dir", str(split_dir),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
        ],
        f"Training EfficientNet-B0 Classifier ({epochs} epochs)",
    )

    run_cmd(
        [
            sys.executable, "-m", "src.classify.eval",
            "--data_dir", str(split_dir),
            "--weights", "reports/runs/classify/best.pt",
        ],
        "Evaluating EfficientNet-B0 Classifier",
    )


def train_pcb(epochs: int, batch_size: int, model: str, skip_prep: bool = False) -> None:
    print("\n--- [STAGE 2/3] PCB Component Detector (YOLOv8) ---")
    raw_dir = Path("data/raw/dataset2_pcb")
    processed_yaml = Path("data/processed/pcb_detect/dataset.yaml")

    if not skip_prep:
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            print(f"⚠ Warning: {raw_dir} is empty or missing. Please place HuggingFace PCB dataset there.")
        run_cmd(
            [sys.executable, "-m", "src.detect_pcb.prepare", "--raw_dir", str(raw_dir), "--out_dir", "data/processed/pcb_detect"],
            "Preparing Dataset 2 for YOLOv8",
        )

    run_cmd(
        [
            sys.executable, "-m", "src.detect_pcb.train",
            "--dataset", str(processed_yaml),
            "--model", model,
            "--epochs", str(epochs),
            "--batch", str(batch_size),
        ],
        f"Training YOLOv8 PCB Detector ({epochs} epochs)",
    )

    run_cmd(
        [
            sys.executable, "-m", "src.detect_pcb.eval",
            "--dataset", str(processed_yaml),
            "--weights", "reports/runs/detect_pcb/train/weights/best.pt",
        ],
        "Evaluating YOLOv8 PCB Detector",
    )


def train_ramssd(epochs: int, batch_size: int, model: str, skip_prep: bool = False, fallback: bool = False) -> None:
    print("\n--- [STAGE 3/3] RAM/SSD Detector (YOLOv8) ---")
    raw_dir = Path("data/raw/dataset3_fallback" if fallback else "data/raw/dataset3_ramssd")
    processed_yaml = Path("data/processed/ramssd_detect/dataset.yaml")

    if not skip_prep:
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            print(f"⚠ Warning: {raw_dir} is empty or missing. Please place Roboflow RAM/SSD dataset there.")
        cmd = [sys.executable, "-m", "src.detect_ramssd.prepare", "--raw_dir", str(raw_dir), "--out_dir", "data/processed/ramssd_detect"]
        if fallback:
            cmd.append("--fallback")
        run_cmd(cmd, "Preparing Dataset 3 for YOLOv8")

    run_cmd(
        [
            sys.executable, "-m", "src.detect_ramssd.train",
            "--dataset", str(processed_yaml),
            "--model", model,
            "--epochs", str(epochs),
            "--batch", str(batch_size),
        ],
        f"Training YOLOv8 RAM/SSD Detector ({epochs} epochs)",
    )

    run_cmd(
        [
            sys.executable, "-m", "src.detect_ramssd.eval",
            "--dataset", str(processed_yaml),
            "--weights", "reports/runs/detect_ramssd/train/weights/best.pt",
        ],
        "Evaluating YOLOv8 RAM/SSD Detector",
    )


def main():
    parser = argparse.ArgumentParser(description="Master Training Runner for E-Waste RecoveryAI")
    parser.add_argument("--all", action="store_true", help="Run all training stages in sequence")
    parser.add_argument("--module", choices=["classify", "pcb", "ramssd"], help="Run specific training module")
    
    # Hyperparams
    parser.add_argument("--epochs-classify", type=int, default=30, help="Epochs for classifier (default: 30)")
    parser.add_argument("--batch-classify", type=int, default=32, help="Batch size for classifier (default: 32)")
    parser.add_argument("--lr-classify", type=float, default=1e-3, help="Learning rate for classifier (default: 0.001)")
    
    parser.add_argument("--epochs-pcb", type=int, default=100, help="Epochs for PCB YOLO (default: 100)")
    parser.add_argument("--batch-pcb", type=int, default=16, help="Batch size for PCB YOLO (default: 16)")
    parser.add_argument("--model-pcb", type=str, default="yolov8n.pt", help="YOLO model for PCB (default: yolov8n.pt)")

    parser.add_argument("--epochs-ramssd", type=int, default=100, help="Epochs for RAM/SSD YOLO (default: 100)")
    parser.add_argument("--batch-ramssd", type=int, default=8, help="Batch size for RAM/SSD YOLO (default: 8)")
    parser.add_argument("--model-ramssd", type=str, default="yolov8n.pt", help="YOLO model for RAM/SSD (default: yolov8n.pt)")
    parser.add_argument("--ramssd-fallback", action="store_true", help="Use fallback dataset for RAM/SSD")

    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preprocessing if already done")

    args = parser.parse_args()

    if not args.all and not args.module:
        parser.print_help()
        print("\n💡 Example: python train_all.py --all")
        print("💡 Example: python train_all.py --module classify")
        sys.exit(1)

    print("======================================================================")
    print("           E-Waste RecoveryAI — Model Training Suite                  ")
    print("======================================================================")

    if args.all or args.module == "classify":
        train_classifier(
            epochs=args.epochs_classify,
            batch_size=args.batch_classify,
            lr=args.lr_classify,
            skip_dedupe=args.skip_prep,
        )

    if args.all or args.module == "pcb":
        train_pcb(
            epochs=args.epochs_pcb,
            batch_size=args.batch_pcb,
            model=args.model_pcb,
            skip_prep=args.skip_prep,
        )

    if args.all or args.module == "ramssd":
        train_ramssd(
            epochs=args.epochs_ramssd,
            batch_size=args.batch_ramssd,
            model=args.model_ramssd,
            skip_prep=args.skip_prep,
            fallback=args.ramssd_fallback,
        )

    print("\n🎉 Training completed successfully! Check reports/metrics.json for updated scores.")


if __name__ == "__main__":
    main()
