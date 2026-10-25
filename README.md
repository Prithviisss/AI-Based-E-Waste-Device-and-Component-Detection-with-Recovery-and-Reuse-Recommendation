# E-Waste RecoveryAI

**AI-Based E-Waste Device and Component Detection with Circular Recovery & Reuse Recommendation**

An end-to-end artificial intelligence and circular economy intelligence system designed to identify discarded electronic devices and components from photos, provide practical reuse, repair, and recycling recommendations, display calibrated detection confidence scores, and calculate realistic financial recovery valuations in **Indian Rupees (₹)** along with carbon emission savings.

---

## 🌟 Key Features

- **Device & Component Identification**: Multi-model computer vision pipeline (EfficientNet-B0 device classifier + parallel YOLOv8 detectors for circuit boards and modular components).
- **Practical Action Guidance ("What Can Be Done With It")**: Clear, straightforward recommendations categorized into:
  - 🟢 **Direct Reuse & Resale**: For working devices and components with immediate secondhand value.
  - 🟡 **Repair & Refurbish**: For repairable hardware where swapping spare parts restores high residual value.
  - 🔵 **Responsible Material Recycling**: For end-of-life electronics to safely reclaim metals (gold, copper, aluminium) and isolate hazardous substances (lithium, lead, mercury).
- **Calibrated Confidence Scoring**: Temperature-scaled probability calibration ($T=2.2$) providing authentic confidence ratings (e.g., High, Moderate, Tentative) rather than saturated 100% figures.
- **Realistic Valuation in Indian Rupees (₹)**: Financial estimations calibrated against Indian secondary and scrap metal recovery benchmarks:
  - Estimated current value with realistic market ranges.
  - Working secondhand resale benchmark.
  - Guaranteed scrap material floor.
- **Clean, Human-Crafted Web UI**: Free of artificial buzzword clutter, raw JSON dumps, and distracting neon elements. Built with React, Vite, and FastAPI.

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and activate your environment:

```bash
conda activate ewaste
pip install -r requirements.txt
```

### 2. Start the Local Web Application

Start the FastAPI application:

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)** (or **[http://127.0.0.1:8000](http://127.0.0.1:8000)**)

---

## 🛠️ Frontend Development

The frontend is built using React and Vite, located in [`frontend/`](file:///c:/Users/prith/OneDrive/ml%20project/frontend/):

```bash
cd frontend

# Install dependencies
npm install

# Start development server with hot-reload (runs on http://localhost:5173)
npm run dev

# Build production bundle (automatically builds into ../web/)
npm run build
```

---

## 🧪 Model Training & Pipeline

### Training Master Runner
Run all models or individual stages using [`train_all.py`](file:///c:/Users/prith/OneDrive/ml%20project/train_all.py):

```bash
# Run complete end-to-end training pipeline
python train_all.py --all

# Or train specific modules:
python train_all.py --module classify   # EfficientNet-B0 device classifier
python train_all.py --module pcb        # YOLOv8 PCB component detector
python train_all.py --module ramssd     # YOLOv8 RAM/SSD detector
```

See the complete [Training Guide](file:///c:/Users/prith/OneDrive/ml%20project/docs/training_guide.md) for data preparation steps and hyperparameters.

### CLI Pipeline Inference
Test inference and rule-based recovery directly from the command line:

```bash
python -m src.pipeline --image demo/sample_pcb.jpg --age 3 --working true --budget 20.0
```

### Run Test Suite
Run the full test suite (covering class mappings, recovery decision rules, and optimizer constraints):

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

- **[`frontend/`](file:///c:/Users/prith/OneDrive/ml%20project/frontend/)**: Modern React + Vite application (clean UI, currency conversion, confidence meters).
- **[`web/`](file:///c:/Users/prith/OneDrive/ml%20project/web/)**: Production web distribution served directly by FastAPI.
- **[`src/api.py`](file:///c:/Users/prith/OneDrive/ml%20project/src/api.py)**: FastAPI backend endpoints (`/analyze`, `/health`, `/metrics`, static mounts).
- **[`src/pipeline.py`](file:///c:/Users/prith/OneDrive/ml%20project/src/pipeline.py)**: End-to-end inference orchestrator with temperature-scaled confidence scoring.
- **[`src/classify/`](file:///c:/Users/prith/OneDrive/ml%20project/src/classify/)**: EfficientNet-B0 device classifier training and inference.
- **[`src/detect_pcb/`](file:///c:/Users/prith/OneDrive/ml%20project/src/detect_pcb/)**: YOLOv8 circuit board component detector.
- **[`src/detect_ramssd/`](file:///c:/Users/prith/OneDrive/ml%20project/src/detect_ramssd/)**: YOLOv8 RAM and SSD detector.
- **[`mcp_server/`](file:///c:/Users/prith/OneDrive/ml%20project/mcp_server/)**: Knowledge-based recovery decision rules and MILP optimization engine.
- **[`demo/`](file:///c:/Users/prith/OneDrive/ml%20project/demo/)**: Sample test photos (`sample_pcb.jpg`, `sample_ramssd.jpg`).
- **[`tests/`](file:///c:/Users/prith/OneDrive/ml%20project/tests/)**: Automated unit test suite (`test_rules.py`, `test_optimizer.py`, `test_class_map.py`).
- **[`docs/`](file:///c:/Users/prith/OneDrive/ml%20project/docs/)**: Dataset specifications, citation sources, and methodology references.

---

## 📜 Citations & Knowledge Base

Recovery pathways, material valuations, and carbon emission factors are cited from established circular electronics literature:
- **UN Global E-Waste Monitor (2024)** &mdash; Avoided greenhouse gas emission baselines.
- **EU WEEE Directive (2012/19/EU)** &mdash; Hazardous materials isolation standards.
- **Secondary Market & Material Scrap Benchmarks** &mdash; Metal recovery rates (copper, steel, gold, aluminium).
Full citations and sources are cataloged in [`docs/sources.md`](file:///c:/Users/prith/OneDrive/ml%20project/docs/sources.md).
