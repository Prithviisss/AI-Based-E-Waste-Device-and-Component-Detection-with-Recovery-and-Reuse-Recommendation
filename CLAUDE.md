# CLAUDE.md — Persistent Instructions for Claude Code
# Read this file before taking ANY action in this repo.

## Project
E-Waste RecoveryAI — AI-based e-waste device + component detection with rule-based recovery.

## NON-NEGOTIABLE RULES

1. **SEED = 42** everywhere. Always: `from src.utils.seed import set_seed; set_seed(42)`
2. **Recovery logic is rule-based + MILP only.** Never call it a model. Never report accuracy for it.
3. **Never merge Dataset 2 (PCB) and Dataset 3 (RAM/SSD).** They are trained separately. Outputs are combined only in `src/pipeline.py` after inference.
4. **All metrics written only by `src/utils/metrics_writer.py`.** Never write `reports/metrics.json` by hand.
5. **Every KB numeric value must have a citation.** Citations go in `docs/sources.md`.
6. **Unmapped class labels must appear in output.** Never silently drop. See `unmapped_policy: report_in_output` in `src/class_map.yaml`.
7. **No custom dataset.** Only the three public datasets (Kaggle, HuggingFace, Roboflow).
8. **demo/ folder is qualitative only.** Never use it for training or metrics.

## BUILD ORDER — complete and verify each step before the next

| Step | What | Command |
|------|------|---------|
| 1 | Scaffold done ✓ | — |
| 2 | Fill `docs/datasets.md` after downloading each dataset | — |
| 3 | Feasibility check (20-epoch baseline) on datasets 2 & 3 | `python -m src.detect_ramssd.train --epochs 20` |
| 3 | Run eval; if mAP@50 < 0.10 → switch to fallback | `python -m src.detect_ramssd.eval` |
| 4 | Deduplicate dataset 1 | `python -m src.classify.dedupe` |
| 4 | Split dataset 1 (70/15/15) | `python -m src.classify.split` |
| 4 | Train classifier | `python -m src.classify.train` |
| 4 | Eval classifier; re-check leakage if acc > 97% | `python -m src.classify.eval` |
| 5 | Prepare dataset 2 | `python -m src.detect_pcb.prepare` |
| 5 | Train PCB detector | `python -m src.detect_pcb.train` |
| 5 | Eval PCB detector | `python -m src.detect_pcb.eval` |
| 5 | Prepare dataset 3 | `python -m src.detect_ramssd.prepare` |
| 5 | Train RAM/SSD detector | `python -m src.detect_ramssd.train` |
| 5 | Eval RAM/SSD detector | `python -m src.detect_ramssd.eval` |
| 6 | Test full pipeline | `python -m src.pipeline --image demo/<photo>.jpg` |
| 7 | Grad-CAM check | `python -m src.classify.gradcam --mode classifier --image ...` |
| 8 | Start API | `uvicorn src.api:app --reload` |
| 9 | Run all tests | `pytest tests/ -v` |
| 10 | Review `docs/limitations.md` | — |

## Hardware hints
- Use `yolov8n.pt` on <6 GB VRAM; set `batch=8` and `amp=True`.
- Use EfficientNet-B0, not B4/B7.
- `num_workers=0` on Windows if DataLoader errors occur.

## Dataset download commands (after credentials configured)
```bash
# Dataset 1 — Kaggle
kaggle datasets download -d <slug> -p data/raw/dataset1_classification --unzip

# Dataset 2 — HuggingFace
huggingface-cli download --repo-type dataset <slug> --local-dir data/raw/dataset2_pcb

# Dataset 3 — Roboflow (Python)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("<ws>").project("<proj>").version(<v>)
project.download("yolov8", location="data/raw/dataset3_ramssd")
```

## Key file locations
- Class map:      `src/class_map.yaml`
- MCP server:     `mcp_server/ewaste_mcp_server.py`
- Pipeline:       `src/pipeline.py`
- API:            `src/api.py`
- Metrics:        `reports/metrics.json` (auto-written)
- Reports:        `reports/run_<id>.html`
