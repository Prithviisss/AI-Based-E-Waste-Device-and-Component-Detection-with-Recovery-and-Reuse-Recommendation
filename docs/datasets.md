# Dataset Documentation

> **Status:** Fill in all ✏ placeholders after you have downloaded and inspected each dataset.
> This file must be completed **before** any training begins.

---

## Dataset 1 — Kaggle E-Waste Image Dataset (Classification)

| Field | Value |
|-------|-------|
| **URL** | ✏ Confirm exact Kaggle slug, e.g. `https://www.kaggle.com/datasets/<user>/<dataset>` |
| **License** | ✏ Confirm on Kaggle page (e.g. CC BY 4.0, ODbL, etc.) |
| **Total images** | ✏ Actual count after listing files (expected ~3,600) |
| **Download command** | `kaggle datasets download -d <slug> -p data/raw/dataset1_classification --unzip` |

### Class List & Image Counts (verify; do not assume)

| Class | Expected ≈ | Actual count |
|-------|-----------|--------------|
| battery | 300 | ✏ |
| computer | 300 | ✏ |
| keyboard | 300 | ✏ |
| microwave | 300 | ✏ |
| mobile | 300 | ✏ |
| mouse | 300 | ✏ |
| PCB | 300 | ✏ |
| player | 300 | ✏ |
| printer | 300 | ✏ |
| speaker | 300 | ✏ |
| washing_machine | 300 | ✏ |
| TV | 300 | ✏ |
| **Total** | **3,600** | ✏ |

### Split Method

- Deduplication: perceptual hash (pHash), hamming threshold ≤ 10
- Near-duplicates removed: ✏ (fill after running `src/classify/dedupe.py`)
- Split: stratified 70 % train / 15 % val / 15 % test
- Leakage check: SHA-256 hash comparison across splits
- Any score > 97 % accuracy triggers a re-check

---

## Dataset 2 — PCB Object Detection Dataset (HuggingFace)

| Field | Value |
|-------|-------|
| **URL** | ✏ Confirm HuggingFace slug |
| **License** | ✏ Expected MIT — confirm on dataset page |
| **Total images** | ✏ Confirm from HF page |
| **Annotation format** | YOLO (expected) — confirm |
| **Download command** | `huggingface-cli download --repo-type dataset <slug> --local-dir data/raw/dataset2_pcb` |

### Class List (21 expected — confirm all 21 from data.yaml)

✏ Copy the full `names:` list from `data/raw/dataset2_pcb/data.yaml` here after download.

### Split Method

- 80 % train / 10 % val / 10 % test
- Split by `src/detect_pcb/prepare.py` (fixed seed = 42)
- Class distribution histogram: ✏ (fill after prepare.py runs)

---

## Dataset 3 — RAM-SSD-Detector (Roboflow)

| Field | Value |
|-------|-------|
| **URL** | ✏ Confirm Roboflow workspace URL |
| **License** | ✏ Expected MIT — confirm |
| **Total images** | ✏ Expected ~188 — confirm |
| **Classes** | `ram`, `SSD`, `objects` |
| **Download command** | Roboflow API export: `rf.workspace("<ws>").project("<proj>").version(<v>).download("yolov8")` |

### `objects` Class Investigation

Inspect at least 20 randomly selected images annotated as `objects`:

- [ ] Done
- Conclusion: ✏ (heterogeneous catch-all / meaningful grouping — describe)
- Action taken: ✏ (mapped to UNMAPPED / mapped to pcb / other)

### Feasibility Check Result

After 20-epoch YOLOv8n baseline:

| Class | mAP@50 |
|-------|--------|
| ram | ✏ |
| SSD | ✏ |
| objects | ✏ |
| **Overall** | ✏ |

**Fallback triggered?** ✏ Yes / No

---

## Fallback — Components 2 (Roboflow) [if triggered]

| Field | Value |
|-------|-------|
| **URL** | ✏ Confirm Roboflow URL |
| **License** | ✏ Confirm |
| **Total images** | ✏ Expected 2,561 — confirm |
| **Classes** | CPU, DISK, GPU, Motherboard, PSU |
| **RAM annotated?** | ✏ Yes / No — confirm |
| **DISK mapping** | DISK → `storage_drive` in class_map.yaml (cannot distinguish HDD/SSD) |

**Reason switch was made:** ✏

---

## Split Reproducibility

All splits use `SEED = 42` via `src/utils/seed.py`. File hashes were compared post-split to confirm zero leakage. Hash comparison log: `data/processed/*/split_hashes.txt`.
