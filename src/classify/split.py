"""
src/classify/split.py
----------------------
Stratified train/val/test split for Dataset 1 after deduplication.

Usage:
    python -m src.classify.split \
        --input  data/processed/classification/deduped \
        --output data/processed/classification/split

Ratios: 70 % train / 15 % val / 15 % test (stratified by class).
Writes split_hashes.txt for leakage verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from src.utils.seed import set_seed

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST = 1 - TRAIN - VAL = 0.15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def split_dataset(input_dir: Path, output_dir: Path) -> None:
    set_seed(SEED)

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in {input_dir}")

    print(f"[split] Classes: {[d.name for d in class_dirs]}")

    all_paths: List[Path] = []
    all_labels: List[str] = []
    for cls_dir in class_dirs:
        imgs = sorted(cls_dir.glob("*"))
        imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        all_paths.extend(imgs)
        all_labels.extend([cls_dir.name] * len(imgs))

    print(f"[split] Total images after dedupe: {len(all_paths)}")

    # First split: train vs (val + test)
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        all_paths, all_labels,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=all_labels,
        random_state=SEED,
    )
    # Second split: val vs test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=0.5,   # 50% of 30% = 15%
        stratify=valtest_labels,
        random_state=SEED,
    )

    splits = {"train": (train_paths, train_labels),
              "val":   (val_paths,   val_labels),
              "test":  (test_paths,  test_labels)}

    hash_records: Dict[str, str] = {}
    for split_name, (paths, labels) in splits.items():
        print(f"[split] {split_name}: {len(paths)} images")
        for path, label in zip(paths, labels):
            dst = output_dir / split_name / label / path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            hash_records[f"{split_name}/{label}/{path.name}"] = sha256(dst)

    # Leakage verification: compare sets of hashes
    train_hashes = {v for k, v in hash_records.items() if k.startswith("train/")}
    val_hashes   = {v for k, v in hash_records.items() if k.startswith("val/")}
    test_hashes  = {v for k, v in hash_records.items() if k.startswith("test/")}

    tv_overlap = train_hashes & val_hashes
    tt_overlap = train_hashes & test_hashes
    vt_overlap = val_hashes & test_hashes

    if tv_overlap or tt_overlap or vt_overlap:
        print(f"  ⚠ LEAKAGE DETECTED: train∩val={len(tv_overlap)}, train∩test={len(tt_overlap)}, val∩test={len(vt_overlap)}")
    else:
        print("[split] ✓ Zero hash overlap between splits (no leakage)")

    # Save hashes
    hash_path = output_dir / "split_hashes.txt"
    hash_path.write_text(json.dumps(hash_records, indent=2))
    print(f"[split] Hash log → {hash_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, default=Path("data/processed/classification/deduped"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/classification/split"))
    args = parser.parse_args()
    split_dataset(args.input, args.output)
