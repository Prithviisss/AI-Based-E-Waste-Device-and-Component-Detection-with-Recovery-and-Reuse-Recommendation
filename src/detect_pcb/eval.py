"""
src/detect_pcb/eval.py
-----------------------
Evaluates the PCB detector on its held-out test split.
Saves mAP@50, mAP@50-95, precision, recall, per-class results
to reports/metrics.json under key "detector_pcb".

Usage:
    python -m src.detect_pcb.eval \
        --model   reports/runs/detect_pcb/train/weights/best.pt \
        --dataset data/processed/pcb_detect/dataset.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.metrics_writer import update_section


def evaluate(model_path: Path = None, dataset_yaml: Path = None, weights: Path = None) -> dict:
    if model_path is None and weights is not None:
        model_path = weights
    if model_path is None:
        raise ValueError("model_path (or weights) must be provided")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    model = YOLO(str(model_path))
    metrics = model.val(data=str(dataset_yaml), split="test", verbose=True)

    # Extract scalar metrics
    mp   = float(metrics.box.mp)     # mean precision
    mr   = float(metrics.box.mr)     # mean recall
    map50    = float(metrics.box.map50)
    map5095  = float(metrics.box.map)

    # Per-class results
    per_class = {}
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names[int(cls_idx)]
            per_class[cls_name] = {
                "ap50": float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") else None,
                "ap50_95": float(metrics.box.ap[i]) if hasattr(metrics.box, "ap") else None,
            }

    result = {
        "model": str(model_path),
        "dataset": "dataset2",
        "map50": round(map50, 6),
        "map50_95": round(map5095, 6),
        "precision": round(mp, 6),
        "recall": round(mr, 6),
        "per_class": per_class,
    }

    print(f"\n[detect_pcb/eval] mAP@50={map50:.4f}  mAP@50-95={map5095:.4f}  P={mp:.4f}  R={mr:.4f}")
    update_section("detector_pcb", result)
    print("[detect_pcb/eval] Metrics saved → reports/metrics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=Path, default=Path("reports/runs/detect_pcb/train/weights/best.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/pcb_detect/dataset.yaml"))
    args = parser.parse_args()
    evaluate(args.model, args.dataset)
