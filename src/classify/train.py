"""
src/classify/train.py
----------------------
EfficientNet-B0 fine-tuning for 12-class e-waste device classification.

Usage:
    python -m src.classify.train \
        --data_dir data/processed/classification/split \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-3

Expects the split directory layout:
    split/
      train/  <class_name>/  *.jpg
      val/    <class_name>/  *.jpg
      test/   <class_name>/  *.jpg  (not touched during training)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.utils.seed import set_seed
from src.utils.metrics_writer import update_section

SEED = 42
CLASSES = [
    "battery", "computer", "keyboard", "microwave", "mobile",
    "mouse", "PCB", "player", "printer", "speaker", "washing_machine", "TV",
]


def get_transforms(train: bool, img_size: int = 224):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def train(
    data_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    img_size: int = 224,
    num_workers: int = 4,
    output_dir: Path = Path("reports/runs/classify"),
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    train_ds = ImageFolder(data_dir / "train", transform=get_transforms(train=True, img_size=img_size))
    val_ds   = ImageFolder(data_dir / "val",   transform=get_transforms(train=False, img_size=img_size))

    num_classes = len(train_ds.classes)
    print(f"[train] Classes ({num_classes}): {train_ds.classes}")
    print(f"[train] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc, patience_counter = 0.0, 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_classifier.pt")
            print(f"  ✓ New best val_acc={val_acc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[train] Early stopping at epoch {epoch}")
                break

    # Save training history
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Update metrics.json (partial — eval.py will add test metrics)
    update_section("classifier", {
        "model": "efficientnet_b0",
        "dataset": "dataset1",
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best_val_acc": best_val_acc,
        "classes": train_ds.classes,
        "checkpoint": str(output_dir / "best_classifier.pt"),
    })
    print(f"[train] Done. Best val accuracy: {best_val_acc:.4f}")
    return model, train_ds.classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=Path, default=Path("data/processed/classification/split"))
    parser.add_argument("--epochs",     type=int,  default=30)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--lr",         type=float,default=1e-3)
    parser.add_argument("--output_dir", type=Path, default=Path("reports/runs/classify"))
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, output_dir=args.output_dir)
