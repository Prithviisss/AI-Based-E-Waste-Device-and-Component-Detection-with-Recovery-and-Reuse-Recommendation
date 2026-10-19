"""
src/classify/dedupe.py
-----------------------
Perceptual-hash duplicate removal for Dataset 1.

Usage:
    python -m src.classify.dedupe \
        --input  data/raw/dataset1_classification \
        --output data/processed/classification/deduped

Algorithm:
    - Compute pHash for every image (imagehash library)
    - Build a union-find structure to group near-duplicates (hamming ≤ threshold)
    - Keep exactly one representative per group
    - Log removal statistics to stdout and to data/processed/classification/dedupe_log.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

THRESHOLD = 10  # hamming distance; lower = stricter


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_hashes(image_paths: List[Path]) -> List[imagehash.ImageHash]:
    hashes = []
    for p in tqdm(image_paths, desc="Computing pHashes"):
        try:
            img = Image.open(p).convert("RGB")
            hashes.append(imagehash.phash(img))
        except Exception as e:
            log.warning(f"Cannot hash {p}: {e}")
            hashes.append(None)
    return hashes


def find_duplicates(
    hashes: List[imagehash.ImageHash | None],
    threshold: int,
) -> UnionFind:
    n = len(hashes)
    uf = UnionFind(n)
    for i in range(n):
        if hashes[i] is None:
            continue
        for j in range(i + 1, n):
            if hashes[j] is None:
                continue
            if hashes[i] - hashes[j] <= threshold:
                uf.union(i, j)
    return uf


def dedupe(
    input_dir: Path,
    output_dir: Path,
    threshold: int = THRESHOLD,
) -> Dict:
    """
    Remove near-duplicates from *input_dir* (class subdirs expected) and
    copy kept images to *output_dir* preserving the same subdirectory layout.
    Returns a summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: List[Path] = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    log.info(f"Found {len(all_paths)} images in {input_dir}")

    hashes = compute_hashes(all_paths)
    uf = find_duplicates(hashes, threshold)

    # Keep the lexicographically-first path in each group
    groups: Dict[int, List[int]] = {}
    for idx in range(len(all_paths)):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    kept, removed = [], []
    for root, members in groups.items():
        keep_idx = min(members)  # representative = first alphabetically
        kept.append(all_paths[keep_idx])
        for m in members:
            if m != keep_idx:
                removed.append(all_paths[m])

    # Copy kept images
    for src in tqdm(kept, desc="Copying kept images"):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    summary = {
        "total_original": len(all_paths),
        "total_kept": len(kept),
        "total_removed": len(removed),
        "threshold_hamming": threshold,
        "removed_paths": [str(p) for p in removed],
    }

    log_path = output_dir.parent / "dedupe_log.json"
    log_path.write_text(json.dumps(summary, indent=2))
    log.info(
        f"Deduplication complete: {len(kept)} kept, {len(removed)} removed. "
        f"Log → {log_path}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptual-hash dedupe for Dataset 1")
    parser.add_argument("--input",  type=Path, default=Path("data/raw/dataset1_classification"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/classification/deduped"))
    parser.add_argument("--threshold", type=int, default=THRESHOLD)
    args = parser.parse_args()
    dedupe(args.input, args.output, args.threshold)
