"""
src/detect_pcb/prepare.py
--------------------------
Prepares Dataset 2 (HuggingFace PCB Object Detection) for YOLOv8 training.

Steps:
1. Verify the download exists and read data.yaml
2. Confirm image count and class list — write to docs/datasets.md section
3. Create an 80/10/10 train/val/test split (if not already present)
4. Write a final dataset.yaml pointing to data/processed/pcb_detect/

Usage:
    python -m src.detect_pcb.prepare \
        --raw_dir data/raw/dataset2_pcb \
        --out_dir data/processed/pcb_detect
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.utils.seed import set_seed

SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test


def read_dataset_yaml(raw_dir: Path) -> Dict:
    for candidate in ["data.yaml", "dataset.yaml"]:
        p = raw_dir / candidate
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    # Search recursively if the archive was extracted into a subfolder
    for p in list(raw_dir.rglob("data.yaml")) + list(raw_dir.rglob("dataset.yaml")):
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No data.yaml / dataset.yaml found in {raw_dir}")


def collect_image_label_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """Return (image_path, label_path) pairs from a YOLO-format directory."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        # Look for matching .txt label
        label_candidates = [
            img_path.with_suffix(".txt"),
            img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name,
            img_path.parent.parent / "labels" / img_path.parent.name / img_path.with_suffix(".txt").name,
            img_path.parent / (img_path.stem + ".txt"),
        ]
        label_path = next((lp for lp in label_candidates if lp.exists()), None)
        if label_path:
            pairs.append((img_path, label_path))
    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    ratios: Tuple[float, float, float],
) -> Tuple[List, List, List]:
    set_seed(SEED)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def copy_split(
    pairs: List[Tuple[Path, Path]],
    out_dir: Path,
    split_name: str,
) -> None:
    img_dir = out_dir / split_name / "images"
    lbl_dir = out_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_dir / img_path.name)
        shutil.copy2(lbl_path, lbl_dir / lbl_path.name)


def prepare(raw_dir: Path, out_dir: Path) -> None:
    set_seed(SEED)
    ds_yaml = read_dataset_yaml(raw_dir)
    class_names: List[str] = ds_yaml.get("names", [])
    nc = ds_yaml.get("nc", len(class_names))

    print(f"[prepare_pcb] Dataset classes ({nc}): {class_names}")

    pairs = collect_image_label_pairs(raw_dir)
    print(f"[prepare_pcb] Found {len(pairs)} image-label pairs")

    if len(pairs) == 0:
        raise RuntimeError(
            f"No annotated images found in {raw_dir}. "
            "Download dataset 2 first: see docs/datasets.md"
        )

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, SPLIT_RATIOS)
    print(f"[prepare_pcb] Split: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_split(train_pairs, out_dir, "train")
    copy_split(val_pairs,   out_dir, "val")
    copy_split(test_pairs,  out_dir, "test")

    # Write dataset.yaml for YOLO
    dataset_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    nc,
        "names": class_names,
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"[prepare_pcb] dataset.yaml → {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw/dataset2_pcb"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/pcb_detect"))
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir)
