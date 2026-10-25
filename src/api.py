"""
src/api.py
-----------
FastAPI backend for E-Waste RecoveryAI.

Endpoints:
  POST /analyze     — upload image + optional metadata → full analysis
  GET  /health      — liveness check
  GET  /metrics     — returns reports/metrics.json
  Static /reports/  — serve HTML reports and figures

Run:
  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.utils.metrics_writer import get_all

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="E-Waste RecoveryAI",
    description=(
        "AI-based e-waste device detection with rule-based recovery and reuse recommendations. "
        "Recovery logic is NOT a trained model — no accuracy metric applies to it."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static reports and web UI
app.mount("/reports",  StaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/web",      StaticFiles(directory="web"),             name="web")
if Path("web/assets").exists():
    app.mount("/assets", StaticFiles(directory="web/assets"), name="assets")
if Path("demo").exists():
    app.mount("/demo",   StaticFiles(directory="demo"),       name="demo")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = Path("web/index.html")
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>E-Waste RecoveryAI</h1><p>Web UI not found. POST to /analyze.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ewaste-recoveryai"}


@app.get("/metrics")
async def metrics():
    return JSONResponse(get_all())


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(..., description="E-waste image (jpg/png)"),
    age_years: Optional[int] = Form(None, ge=0, le=50, description="Device age in years"),
    working:   Optional[str] = Form(None, description="Working status: true / false / unknown"),
    budget_usd: float        = Form(20.0, ge=0.0, description="Recovery budget in USD"),
):
    """
    Upload an e-waste photo and optionally provide age and working status.
    Returns device classification, component detections, recovery recommendations,
    optimized pathway, and links to the Grad-CAM report.
    """
    # Validate image
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=422, detail="Only JPEG/PNG/WEBP images are accepted.")

    # Parse working status
    working_map = {"true": True, "false": False, "unknown": None, None: None, "": None}
    working_bool = working_map.get((working or "").lower(), None)

    # Save upload to temp file
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = Path(tempfile.mktemp(suffix=suffix, dir=str(REPORTS_DIR)))
    content = await image.read()
    tmp_path.write_bytes(content)

    try:
        from src.pipeline import run
        result = run(
            image_path=tmp_path,
            age_years=age_years,
            working=working_bool,
            budget_usd=budget_usd,
            generate_report=True,
            generate_heatmaps=True,
        )
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    # Convert report path to URL
    report_url = None
    if result.get("report_path"):
        rp = Path(result["report_path"])
        if rp.exists():
            report_url = f"/reports/{rp.name}"

    heatmap_urls = []
    for h in result.get("heatmap_paths", []):
        hp = Path(h.get("path", ""))
        if hp.exists():
            heatmap_urls.append({
                "label": h.get("label", ""),
                "url": f"/reports/figures/{hp.name}",
            })

    return {
        "device": result["device"],
        "device_confidence": result["device_confidence"],
        "detections": result["detections"],
        "unmapped": result["unmapped"],
        "recovery_recommendations": result["recovery_recommendations"],
        "optimized_pathways": result["optimized_pathways"],
        "report_url": report_url,
        "heatmap_urls": heatmap_urls,
    }
