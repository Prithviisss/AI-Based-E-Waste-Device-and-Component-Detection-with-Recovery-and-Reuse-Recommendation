"""
src/detect_ramssd/prepare.py
-----------------------------
Prepares Dataset 3 (Roboflow RAM-SSD-Detector) or the fallback
(Components 2) for YOLOv8 training.

IMPORTANT:
- This is a SEPARATE training set from Dataset 2 (PCB).
- Never merge them.

Usage:
    python -m src.detect_ramssd.prepare \
        --raw_dir data/raw/dataset3_ramssd \
        --out_dir data/processed/ramssd_detect

For the fallback dataset set --fallback flag:
    python -m src.detect_ramssd.prepare \
        --raw_dir data/raw/dataset3_fallback \
        --out_dir data/processed/ramssd_detect \
        --fallback
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
SPLIT_RATIOS = (0.80, 0.10, 0.10)


def read_dataset_yaml(raw_dir: Path) -> Dict:
    for candidate in ["data.yaml", "dataset.yaml"]:
        p = raw_dir / candidate
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No data.yaml found in {raw_dir}")


def collect_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    img_exts = {".jpg", ".jpeg", ".png"}
    pairs = []
    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        lbl_candidates = [
            img_path.with_suffix(".txt"),
            img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name,
        ]
        lbl = next((lp for lp in lbl_candidates if lp.exists()), None)
        if lbl:
            pairs.append((img_path, lbl))
    return pairs


def split_and_copy(pairs: List[Tuple[Path, Path]], out_dir: Path) -> None:
    set_seed(SEED)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])
    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }
    for split_name, split_pairs in splits.items():
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
        print(f"  {split_name}: {len(split_pairs)} pairs")


def prepare(raw_dir: Path, out_dir: Path, fallback: bool = False) -> None:
    set_seed(SEED)
    ds_yaml = read_dataset_yaml(raw_dir)
    class_names: List[str] = ds_yaml.get("names", [])
    nc = ds_yaml.get("nc", len(class_names))

    print(f"[prepare_ramssd] Dataset classes ({nc}): {class_names}")
    if not fallback and "objects" in class_names:
        print(
            "[prepare_ramssd] ⚠  'objects' class found. "
            "Inspect 20 random images with this label BEFORE training. "
            "Update src/class_map.yaml accordingly. See docs/datasets.md."
        )

    pairs = collect_pairs(raw_dir)
    print(f"[prepare_ramssd] Found {len(pairs)} annotated images")
    if len(pairs) == 0:
        raise RuntimeError(
            f"No annotated images in {raw_dir}. "
            "Download dataset 3 first (see docs/datasets.md)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    split_and_copy(pairs, out_dir)

    dataset_yaml = {
        "path":  str(out_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    nc,
        "names": class_names,
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"[prepare_ramssd] dataset.yaml → {yaml_path}")
    if fallback:
        (out_dir / "FALLBACK_USED.txt").write_text(
            "Components 2 (Roboflow) fallback was used because Dataset 3 mAP was near zero."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",  type=Path, default=Path("data/raw/dataset3_ramssd"))
    parser.add_argument("--out_dir",  type=Path, default=Path("data/processed/ramssd_detect"))
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir, args.fallback)
