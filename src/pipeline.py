"""
src/pipeline.py
----------------
End-to-end inference pipeline:

  Image → device classifier
        → PCB detector (dataset 2 model)
        → RAM/SSD detector (dataset 3 / fallback model)
        → class_map.yaml routing
        → recommend_recovery (rule-based MCP logic)
        → optimize_pathways (MILP)
        → Grad-CAM figures
        → HTML/PDF report

Usage:
    python -m src.pipeline \
        --image demo/phone1.jpg \
        --age   5 \
        --working true \
        --budget 20.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from PIL import Image

from src.utils.seed import set_seed

SEED = 42

# Default model paths — override with CLI args or env vars
CLASSIFIER_CKPT  = Path("reports/runs/classify/best_classifier.pt")
PCB_MODEL        = Path("reports/runs/detect_pcb/train/weights/best.pt")
RAMSSD_MODEL     = Path("reports/runs/detect_ramssd/train/weights/best.pt")
CLASS_MAP_PATH   = Path("src/class_map.yaml")

CLASSIFIER_CLASSES = [
    "battery", "computer", "keyboard", "microwave", "mobile",
    "mouse", "PCB", "player", "printer", "speaker", "washing_machine", "TV",
]


# ──────────────────────────────────────────────────────────────────────────────
# Class map loader
# ──────────────────────────────────────────────────────────────────────────────

def load_class_map(path: Path = CLASS_MAP_PATH) -> Dict[str, str]:
    """
    Returns a dict keyed by 'source:label' → kb_key.
    Labels with kb_key == 'UNMAPPED' are kept so the pipeline can report them.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mapping = {}
    for entry in data.get("mappings", []):
        key = f"{entry['source']}:{entry['label']}"
        mapping[key] = entry.get("kb_key", "UNMAPPED")
    return mapping


def resolve_kb_key(
    source: str, label: str, mapping: Dict[str, str]
) -> Optional[str]:
    """
    Try source-qualified lookup first, then bare label fallback.
    Returns kb_key or None if not found.
    """
    qualified = f"{source}:{label}"
    if qualified in mapping:
        return mapping[qualified]
    # Bare label fallback (any source)
    for k, v in mapping.items():
        if k.split(":", 1)[1] == label:
            return v
    # Normalized & case-insensitive fallback (e.g. "Television" -> "tv", "Washing Machine" -> "washing_machine")
    norm_label = label.lower().strip().replace(" ", "_")
    for k, v in mapping.items():
        k_label = k.split(":", 1)[1].lower().strip().replace(" ", "_")
        if k_label == norm_label:
            return v
    if norm_label in ("television", "tv"):
        return "tv"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Classifier inference
# ──────────────────────────────────────────────────────────────────────────────

def run_classifier(
    image_path: Path,
    ckpt: Path = CLASSIFIER_CKPT,
    num_classes: Optional[int] = None,
) -> tuple[str, float]:
    """Returns (predicted_label, confidence)."""
    if not ckpt.exists():
        print(f"[pipeline] Classifier checkpoint not found: {ckpt}. Returning 'unknown'.")
        return "unknown", 0.0

    import torchvision.transforms as T
    from src.classify.train import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt, map_location=device)

    # Auto-detect number of classes from state_dict
    if num_classes is None:
        if "classifier.1.weight" in state_dict:
            num_classes = state_dict["classifier.1.weight"].shape[0]
        else:
            num_classes = 10

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Determine class names list matching trained checkpoint
    classes = CLASSIFIER_CLASSES
    try:
        from src.utils.metrics_writer import get_section
        m = get_section("classifier")
        if "classes" in m and len(m["classes"]) == num_classes:
            classes = m["classes"]
        elif "per_class_f1" in m and len(m["per_class_f1"]) == num_classes:
            classes = list(m["per_class_f1"].keys())
    except Exception:
        pass

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        # Temperature scaling calibration (T=2.2) to prevent overconfident logit saturation (100% certainty)
        temperature = 2.2
        calibrated_logits = logits / temperature
        probs  = torch.softmax(calibrated_logits, dim=1)
        conf, pred = probs.max(dim=1)

    label = classes[pred.item()] if pred.item() < len(classes) else "unknown"
    return label, round(float(conf.item()), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Detector inference
# ──────────────────────────────────────────────────────────────────────────────

def run_detector(
    image_path: Path,
    model_path: Path,
    source_tag: str,
    conf_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {label, confidence, bbox, source}.
    Returns [] if model not found (logged as warning).
    """
    if not model_path.exists():
        print(f"[pipeline] ⚠ Detector model not found: {model_path} — skipping.")
        return []

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[pipeline] ultralytics not installed — skipping detector.")
        return []

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_idx  = int(box.cls.item())
            label    = model.names[cls_idx]
            conf     = round(float(box.conf.item()), 4)
            xyxy     = box.xyxy.squeeze().tolist()
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy,
                "source": source_tag,
            })
    return detections


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(
    image_path: str | Path,
    age_years: Optional[int] = None,
    working: Optional[bool] = None,
    budget_usd: float = 20.0,
    classifier_ckpt: Path = CLASSIFIER_CKPT,
    pcb_model: Path = PCB_MODEL,
    ramssd_model: Path = RAMSSD_MODEL,
    generate_report: bool = True,
    generate_heatmaps: bool = True,
) -> Dict[str, Any]:
    set_seed(SEED)
    image_path = Path(image_path)

    # ── 1. Class map ──────────────────────────────────────────────────────────
    class_mapping = load_class_map()
    print(f"[pipeline] Loaded {len(class_mapping)} class-map entries")

    # ── 2. Classify ───────────────────────────────────────────────────────────
    device_label, device_conf = run_classifier(image_path, classifier_ckpt)
    print(f"[pipeline] Device: {device_label} (conf={device_conf:.3f})")

    # ── 3. Detect (two separate detectors — never merged) ─────────────────────
    # Context-aware filter: Peripherals & non-computing appliances do not have modular RAM/SSD
    norm_dev = device_label.lower().strip().replace(" ", "_")
    ramssd_incompatible = norm_dev in {
        "keyboard", "mouse", "battery", "washing_machine", "microwave", "speaker", "tv"
    }

    pcb_dets = run_detector(image_path, pcb_model, source_tag="dataset2", conf_threshold=0.35)

    if ramssd_incompatible:
        print(f"[pipeline] Suppressing RAM/SSD detector for peripheral/appliance: {device_label}")
        ramssd_dets = []
    else:
        ramssd_dets = run_detector(image_path, ramssd_model, source_tag="dataset3", conf_threshold=0.40)

    all_dets    = pcb_dets + ramssd_dets
    print(f"[pipeline] Detections: {len(pcb_dets)} PCB + {len(ramssd_dets)} RAM/SSD")

    # ── 4. Class-map routing ──────────────────────────────────────────────────
    unmapped: List[str] = []
    for det in all_dets:
        kb_key = resolve_kb_key(det["source"], det["label"], class_mapping)
        if kb_key is None:
            print(f"[pipeline] ⚠ Unmapped: {det['source']}:{det['label']}")
            unmapped.append(f"{det['source']}:{det['label']}")
            det["kb_key"] = None
        elif kb_key == "UNMAPPED":
            unmapped.append(f"{det['source']}:{det['label']}")
            det["kb_key"] = None
        else:
            det["kb_key"] = kb_key

    # ── 5. Recovery recommendations (rule-based MCP logic) ───────────────────
    from mcp_server.ewaste_mcp_server import recommend_recovery, optimize_pathways

    recommendations = []
    flags: List[str] = []
    for det in all_dets:
        if det.get("kb_key") is None:
            continue
        rec = recommend_recovery(
            component=det["kb_key"],
            age_years=age_years,
            working=working,
            confidence=det["confidence"],
        )
        det["route"]               = rec.get("route")
        det["estimated_value_usd"] = rec.get("estimated_value_usd")
        det["co2_saved_kg"]        = rec.get("co2_saved_kg")
        if rec.get("flags"):
            flags.extend(rec["flags"])
        recommendations.append(rec)

    # Also get a top-level recommendation for the device itself
    device_kb = resolve_kb_key("dataset1", device_label, class_mapping)
    if device_kb:
        device_rec = recommend_recovery(device_kb, age_years, working, device_conf)
        recommendations.insert(0, device_rec)

    # ── 6. Optimize pathways (MILP) ───────────────────────────────────────────
    optimizer_input = [
        {
            "kb_key": det["kb_key"],
            "route":  det.get("route"),
            "estimated_value_usd": det.get("estimated_value_usd", 0),
            "recovery_cost_usd": 2.0,   # default; overridden from KB in real run
            "co2_saved_kg": det.get("co2_saved_kg", 0),
        }
        for det in all_dets if det.get("kb_key")
    ]
    # Add device itself
    if device_kb and recommendations:
        dr = recommendations[0]
        from mcp_server.ewaste_mcp_server import KNOWLEDGE_BASE
        kb_entry = KNOWLEDGE_BASE.get(device_kb, {})
        optimizer_input.insert(0, {
            "kb_key": device_kb,
            "route": dr.get("route"),
            "estimated_value_usd": dr.get("estimated_value_usd", 0),
            "recovery_cost_usd": kb_entry.get("recovery_cost_usd", 5.0),
            "co2_saved_kg": dr.get("co2_saved_kg", 0),
        })

    optimized = optimize_pathways(optimizer_input, budget_usd=budget_usd)
    print(f"[pipeline] Optimizer status: {optimized['status']}")

    # ── 7. Grad-CAM ───────────────────────────────────────────────────────────
    heatmap_paths: List[Dict] = []
    if generate_heatmaps:
        try:
            from src.classify.gradcam import generate_all_heatmaps
            heatmap_paths = generate_all_heatmaps(
                image_path,
                classifier_ckpt=classifier_ckpt if classifier_ckpt.exists() else None,
                pcb_model=pcb_model if pcb_model.exists() else None,
                ramssd_model=ramssd_model if ramssd_model.exists() else None,
            )
        except Exception as e:
            print(f"[pipeline] Grad-CAM failed (non-fatal): {e}")

    # ── 8. Report ─────────────────────────────────────────────────────────────
    report_path = None
    if generate_report:
        from src.utils.report_gen import generate as gen_report
        report_path = gen_report(
            device=device_label,
            device_confidence=device_conf,
            detections=all_dets,
            optimized=optimized,
            heatmap_paths=heatmap_paths,
            unmapped=unmapped,
            flags=list(set(flags)),
        )

    result = {
        "device": device_label,
        "device_confidence": device_conf,
        "detections": all_dets,
        "unmapped": unmapped,
        "recovery_recommendations": recommendations,
        "optimized_pathways": optimized,
        "report_path": str(report_path) if report_path else None,
        "heatmap_paths": heatmap_paths,
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Waste RecoveryAI Pipeline")
    parser.add_argument("--image",    type=Path, required=True)
    parser.add_argument("--age",      type=int,  default=None, help="Device age in years")
    parser.add_argument("--working",  type=str,  default=None, choices=["true","false","unknown"])
    parser.add_argument("--budget",   type=float,default=20.0)
    parser.add_argument("--no_heatmaps", action="store_true")
    parser.add_argument("--no_report",   action="store_true")
    args = parser.parse_args()

    working_map = {"true": True, "false": False, "unknown": None, None: None}
    result = run(
        image_path=args.image,
        age_years=args.age,
        working=working_map[args.working],
        budget_usd=args.budget,
        generate_report=not args.no_report,
        generate_heatmaps=not args.no_heatmaps,
    )

    import json
    print("\n" + "=" * 60)
    print(json.dumps({k: v for k, v in result.items() if k != "detections"}, indent=2, default=str))
    if result.get("report_path"):
        print(f"\n📄 Report: {result['report_path']}")
