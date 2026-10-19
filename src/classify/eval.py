"""
src/classify/eval.py
---------------------
Evaluates the saved classifier checkpoint on the test split.
Produces:
  - accuracy (top-1)
  - macro F1
  - per-class F1
  - confusion matrix PNG  →  reports/figures/classifier_confusion.png
  - writes all to reports/metrics.json under key "classifier"

Usage:
    python -m src.classify.eval \
        --data_dir  data/processed/classification/split \
        --ckpt      reports/runs/classify/best_classifier.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.classify.train import build_model, get_transforms
from src.utils.seed import set_seed
from src.utils.metrics_writer import update_section

SEED = 42
FIGURES_DIR = Path("reports/figures")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("E-Waste Classifier — Confusion Matrix (test set)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Confusion matrix → {out_path}")


def evaluate(data_dir: Path, ckpt: Path, batch_size: int = 32, num_workers: int = 4):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = ImageFolder(data_dir / "test", transform=get_transforms(train=False))
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_names = test_ds.classes
    num_classes = len(class_names)
    print(f"[eval] Test set: {len(test_ds)} images, {num_classes} classes")

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   output_dict=True, zero_division=0)
    per_class_f1 = {cls: report[cls]["f1-score"] for cls in class_names}
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n[eval] Test accuracy : {acc:.4f}")
    print(f"[eval] Macro F1      : {macro_f1:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in per_class_f1.items():
        print(f"  {cls:<20} {f1:.4f}")

    # Leakage warning
    if acc > 0.97:
        print(
            "\n⚠  WARNING: accuracy > 97 %. Re-verify split for leakage "
            "by comparing SHA-256 hashes of train/val/test files."
        )

    # Confusion matrix
    cm_path = FIGURES_DIR / "classifier_confusion.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # Persist metrics
    update_section("classifier", {
        "n_test": len(test_ds),
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class_f1": {k: round(v, 6) for k, v in per_class_f1.items()},
        "confusion_matrix_path": str(cm_path),
    })
    print("[eval] Metrics saved → reports/metrics.json")
    return acc, macro_f1, per_class_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=Path, default=Path("data/processed/classification/split"))
    parser.add_argument("--ckpt",      type=Path, default=Path("reports/runs/classify/best_classifier.pt"))
    parser.add_argument("--batch_size",type=int,  default=32)
    args = parser.parse_args()
    evaluate(args.data_dir, args.ckpt, args.batch_size)
