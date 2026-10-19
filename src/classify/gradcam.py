"""
src/classify/gradcam.py
------------------------
Grad-CAM explainability for the EfficientNet-B0 classifier and
EigenCAM for YOLOv8 detectors (via pytorch-grad-cam).

Usage (classifier):
    python -m src.classify.gradcam \
        --mode classifier \
        --image path/to/image.jpg \
        --ckpt  reports/runs/classify/best_classifier.pt \
        --out   reports/figures/gradcam_classifier.png

Usage (detector):
    python -m src.classify.gradcam \
        --mode detector \
        --image path/to/image.jpg \
        --model_path reports/runs/detect_pcb/weights/best.pt \
        --out   reports/figures/gradcam_pcb.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

FIGURES_DIR = Path("reports/figures")


# ──────────────────────────────────────────────────────────────────────────────
# Classifier Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

def classifier_gradcam(
    image_path: str | Path,
    ckpt: str | Path,
    num_classes: int = 12,
    out_path: Optional[Path] = None,
) -> Path:
    """Compute Grad-CAM for the classifier and save an overlay PNG."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError("Install pytorch-grad-cam: pip install pytorch-grad-cam")

    from src.classify.train import build_model, get_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # Target layer: last conv block of EfficientNet-B0
    target_layers = [model.features[-1]]

    img_pil = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Forward to get predicted class
    with torch.no_grad():
        logits = model(input_tensor)
    pred_class = int(torch.argmax(logits, dim=1).item())

    cam_algo = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam_algo(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if out_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"gradcam_classifier_{Path(image_path).stem}.png"

    Image.fromarray(overlay).save(out_path)
    print(f"[gradcam] Classifier Grad-CAM → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Detector EigenCAM
# ──────────────────────────────────────────────────────────────────────────────

def detector_eigencam(
    image_path: str | Path,
    model_path: str | Path,
    detector_name: str = "detector",
    out_path: Optional[Path] = None,
) -> Path:
    """Compute EigenCAM on a YOLO backbone feature map and save overlay PNG."""
    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        raise ImportError("Install pytorch-grad-cam: pip install pytorch-grad-cam")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(model_path))
    model = yolo.model.to(device).eval()

    # Target: last layer of the backbone
    target_layers = [model.model[-2]]  # adjust index if needed

    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    cam_algo = EigenCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam_algo(input_tensor=input_tensor)[0]

    rgb_float = img_resized.astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    if out_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"gradcam_{detector_name}_{Path(image_path).stem}.png"

    Image.fromarray(overlay).save(out_path)
    print(f"[gradcam] {detector_name} EigenCAM → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: run both detectors on one image
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_heatmaps(
    image_path: str | Path,
    classifier_ckpt: Optional[str | Path] = None,
    pcb_model: Optional[str | Path] = None,
    ramssd_model: Optional[str | Path] = None,
    num_classes: int = 12,
) -> List[dict]:
    """Return a list of {label, path} dicts for embedding in the report."""
    results = []
    if classifier_ckpt and Path(classifier_ckpt).exists():
        try:
            p = classifier_gradcam(image_path, classifier_ckpt, num_classes)
            results.append({"label": "Classifier Grad-CAM", "path": str(p)})
        except Exception as e:
            print(f"[gradcam] Classifier failed: {e}")

    if pcb_model and Path(pcb_model).exists():
        try:
            p = detector_eigencam(image_path, pcb_model, detector_name="pcb")
            results.append({"label": "PCB Detector EigenCAM", "path": str(p)})
        except Exception as e:
            print(f"[gradcam] PCB detector failed: {e}")

    if ramssd_model and Path(ramssd_model).exists():
        try:
            p = detector_eigencam(image_path, ramssd_model, detector_name="ramssd")
            results.append({"label": "RAM/SSD Detector EigenCAM", "path": str(p)})
        except Exception as e:
            print(f"[gradcam] RAM/SSD detector failed: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["classifier", "detector"], required=True)
    parser.add_argument("--image",      type=Path, required=True)
    parser.add_argument("--ckpt",       type=Path, help="Classifier checkpoint (.pt)")
    parser.add_argument("--model_path", type=Path, help="YOLO model (.pt)")
    parser.add_argument("--out",        type=Path, default=None)
    parser.add_argument("--num_classes",type=int,  default=12)
    args = parser.parse_args()

    if args.mode == "classifier":
        classifier_gradcam(args.image, args.ckpt, args.num_classes, args.out)
    else:
        detector_eigencam(args.image, args.model_path, out_path=args.out)
