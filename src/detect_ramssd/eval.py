"""
src/detect_ramssd/eval.py
--------------------------
Evaluates the RAM/SSD detector on its held-out test split.
Saves metrics to reports/metrics.json under "detector_ramssd".

Also implements the mAP feasibility check:
  If mAP@50 < 0.10 → prints a clear recommendation to use the fallback.

Usage:
    python -m src.detect_ramssd.eval \
        --model   reports/runs/detect_ramssd/train/weights/best.pt \
        --dataset data/processed/ramssd_detect/dataset.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.metrics_writer import update_section

FEASIBILITY_THRESHOLD = 0.10  # mAP@50 below this → recommend fallback


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

    fallback_used = (dataset_yaml.parent / "FALLBACK_USED.txt").exists()
    dataset_key = "dataset3_fallback" if fallback_used else "dataset3"

    model = YOLO(str(model_path))
    metrics = model.val(data=str(dataset_yaml), split="test", verbose=True)

    mp      = float(metrics.box.mp)
    mr      = float(metrics.box.mr)
    map50   = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    per_class = {}
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names[int(cls_idx)]
            per_class[cls_name] = {
                "ap50":    float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") else None,
                "ap50_95": float(metrics.box.ap[i])  if hasattr(metrics.box, "ap")   else None,
            }

    result = {
        "model": str(model_path),
        "dataset": dataset_key,
        "fallback_used": fallback_used,
        "map50":     round(map50, 6),
        "map50_95":  round(map5095, 6),
        "precision": round(mp, 6),
        "recall":    round(mr, 6),
        "per_class": per_class,
    }

    print(f"\n[detect_ramssd/eval] mAP@50={map50:.4f}  mAP@50-95={map5095:.4f}  P={mp:.4f}  R={mr:.4f}")

    # Feasibility check
    if map50 < FEASIBILITY_THRESHOLD and not fallback_used:
        print(
            f"\n⚠  FEASIBILITY WARNING: mAP@50={map50:.4f} < {FEASIBILITY_THRESHOLD}.\n"
            "   RAM/SSD detection is near-zero. SWITCH TO FALLBACK:\n"
            "   1. Download Components 2 from Roboflow → data/raw/dataset3_fallback/\n"
            "   2. python -m src.detect_ramssd.prepare --raw_dir data/raw/dataset3_fallback --fallback\n"
            "   3. python -m src.detect_ramssd.train\n"
            "   4. python -m src.detect_ramssd.eval\n"
            "   Also document switch in docs/datasets.md."
        )

    update_section("detector_ramssd", result)
    print("[detect_ramssd/eval] Metrics saved → reports/metrics.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=Path, default=Path("reports/runs/detect_ramssd/train/weights/best.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/ramssd_detect/dataset.yaml"))
    args = parser.parse_args()
    evaluate(args.model, args.dataset)
