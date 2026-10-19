# E-Waste RecoveryAI — Model Training Guide & Code

This document provides complete instructions and codes for training all three computer vision models in **E-Waste RecoveryAI**:

1. **Device Classifier**: EfficientNet-B0 (12 e-waste categories)
2. **PCB Component Detector**: YOLOv8 (capacitors, ICs, connectors, etc.)
3. **RAM / SSD Component Detector**: YOLOv8 (storage/memory components)

---

## 1. Quick Start: Master Training Script

A unified training orchestrator is available at [train_all.py](file:///c:/Users/prith/OneDrive/ml%20project/train_all.py).

### Commands:

```bash
# Run everything end-to-end (Deduplication -> Split -> Train -> Eval for all models)
python train_all.py --all

# Run only Device Classifier (30 epochs, batch 32)
python train_all.py --module classify --epochs-classify 30 --batch-classify 32

# Run only PCB Detector (100 epochs, batch 16)
python train_all.py --module pcb --epochs-pcb 100 --batch-pcb 16

# Run only RAM/SSD Detector (100 epochs, batch 8)
python train_all.py --module ramssd --epochs-ramssd 100 --batch-ramssd 8

# Smoke test / dry run (1 epoch each to verify everything runs smoothly)
python train_all.py --all --epochs-classify 1 --epochs-pcb 1 --epochs-ramssd 1
```

---

## 2. Dataset Preparation & Download

Before training, download the three public datasets into `data/raw/`:

### Dataset 1: Kaggle E-Waste Image Dataset (Classification)
Target path: `data/raw/dataset1_classification/`
```bash
# Using Kaggle CLI
kaggle datasets download -d <dataset-slug> -p data/raw/dataset1_classification --unzip
```
Expected structure:
```
data/raw/dataset1_classification/
  ├── battery/
  ├── computer/
  ├── keyboard/
  ├── microwave/
  ├── mobile/
  ├── mouse/
  ├── PCB/
  ├── player/
  ├── printer/
  ├── speaker/
  ├── washing_machine/
  └── TV/
```

### Dataset 2: HuggingFace PCB Component Detection
Target path: `data/raw/dataset2_pcb/`
```bash
# Using HuggingFace CLI
huggingface-cli download --repo-type dataset <dataset-slug> --local-dir data/raw/dataset2_pcb
```

### Dataset 3: Roboflow RAM & SSD Detection
Target path: `data/raw/dataset3_ramssd/`
```python
# Download via Roboflow Python SDK:
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("<workspace>").project("ram-ssd-detector").version(1)
project.download("yolov8", location="data/raw/dataset3_ramssd")
```

---

## 3. Step-by-Step Training Workflow & Codes

### Step 3.1: Train Device Classifier (EfficientNet-B0)

#### A. Deduplication (Perceptual Hash)
Removes duplicate and near-duplicate images across classes to prevent data leakage:
```bash
python -m src.classify.dedupe \
    --input data/raw/dataset1_classification \
    --output data/processed/classification/deduped \
    --threshold 5
```

#### B. Stratified Train / Val / Test Split (70 / 15 / 15)
Splits data with a fixed seed (`SEED=42`) and generates SHA-256 integrity hashes:
```bash
python -m src.classify.split \
    --input data/processed/classification/deduped \
    --output data/processed/classification/split \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

#### C. Training EfficientNet-B0
Code file: [src/classify/train.py](file:///c:/Users/prith/OneDrive/ml%20project/src/classify/train.py)
```bash
python -m src.classify.train \
    --data_dir data/processed/classification/split \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001
```
*Trained model will be saved at: `reports/runs/classify/best.pt`*

#### D. Evaluation
Evaluates on the held-out test split and writes classification accuracy, macro-F1, and confusion matrix into `reports/metrics.json`:
```bash
python -m src.classify.eval \
    --data_dir data/processed/classification/split \
    --weights reports/runs/classify/best.pt
```

#### E. Grad-CAM Explainability Check
Generate Grad-CAM heatmaps to inspect what visual features the classifier learned:
```bash
python -m src.classify.gradcam \
    --mode classifier \
    --image demo/sample.jpg \
    --weights reports/runs/classify/best.pt \
    --output demo/gradcam_sample.jpg
```

---

### Step 3.2: Train PCB Component Detector (YOLOv8)

#### A. Prepare YOLO format
Verifies labels, creates split, and generates `dataset.yaml`:
```bash
python -m src.detect_pcb.prepare \
    --raw_dir data/raw/dataset2_pcb \
    --out_dir data/processed/pcb_detect
```

#### B. Train YOLOv8n
Code file: [src/detect_pcb/train.py](file:///c:/Users/prith/OneDrive/ml%20project/src/detect_pcb/train.py)
```bash
python -m src.detect_pcb.train \
    --dataset data/processed/pcb_detect/dataset.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --output reports/runs/detect_pcb
```
*Trained weights will be saved at: `reports/runs/detect_pcb/train/weights/best.pt`*

#### C. Evaluate
Calculates mAP@50, mAP@50-95, precision, recall, and writes to `reports/metrics.json`:
```bash
python -m src.detect_pcb.eval \
    --dataset data/processed/pcb_detect/dataset.yaml \
    --weights reports/runs/detect_pcb/train/weights/best.pt
```

---

### Step 3.3: Train RAM / SSD Detector (YOLOv8)

#### A. Prepare YOLO format
Code file: [src/detect_ramssd/prepare.py](file:///c:/Users/prith/OneDrive/ml%20project/src/detect_ramssd/prepare.py)
```bash
python -m src.detect_ramssd.prepare \
    --raw_dir data/raw/dataset3_ramssd \
    --out_dir data/processed/ramssd_detect
```

*Note: If Roboflow dataset is unavailable or has mAP < 0.10, switch to the fallback dataset using `--fallback`:*
```bash
python -m src.detect_ramssd.prepare \
    --raw_dir data/raw/dataset3_fallback \
    --out_dir data/processed/ramssd_detect \
    --fallback
```

#### B. Train YOLOv8n
Code file: [src/detect_ramssd/train.py](file:///c:/Users/prith/OneDrive/ml%20project/src/detect_ramssd/train.py)
```bash
python -m src.detect_ramssd.train \
    --dataset data/processed/ramssd_detect/dataset.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 8 \
    --output reports/runs/detect_ramssd
```
*Trained weights will be saved at: `reports/runs/detect_ramssd/train/weights/best.pt`*

#### C. Evaluate
```bash
python -m src.detect_ramssd.eval \
    --dataset data/processed/ramssd_detect/dataset.yaml \
    --weights reports/runs/detect_ramssd/train/weights/best.pt
```

---

## 4. Hardware Optimization & Windows Tips

| Situation | Recommended Setting |
|---|---|
| **Low GPU VRAM (< 6 GB)** | Use `--model yolov8n.pt`, `--batch 8` or `16`, and AMP is enabled by default (`--amp`) |
| **CPU Only Training** | Lower epochs (e.g. `--epochs 10`), set `--batch 8` |
| **Windows DataLoader Error (`BrokenPipeError`)** | `workers=0` (automatically handled in scripts for Windows) |
| **Out of Memory (OOM)** | Decrease batch size: `--batch-classify 16` and `--batch-pcb 8` |

---

## 5. Testing the Pipeline with Trained Weights

Once training completes, run an inference test on any image to see full end-to-end device classification, part detection, and recovery recommendation:

```bash
python -m src.pipeline \
    --image demo/sample.jpg \
    --output reports/test_run.html
```

You can also launch the web server and use the UI:
```bash
uvicorn src.api:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.
