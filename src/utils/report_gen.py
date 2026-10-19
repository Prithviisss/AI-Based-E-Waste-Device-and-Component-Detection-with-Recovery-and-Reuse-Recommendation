"""
src/utils/report_gen.py
------------------------
Generates a self-contained HTML report for a single analysis run.
Optionally produces a PDF via weasyprint (install separately).
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>E-Waste RecoveryAI — Report {{ run_id }}</title>
  <style>
    :root {
      --primary: #00c896;
      --bg: #0f1117;
      --card: #1a1d27;
      --border: #2a2d3a;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --danger: #f87171;
      --warn: #fbbf24;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; line-height: 1.6; }
    h1 { color: var(--primary); font-size: 1.8rem; margin-bottom: 0.25rem; }
    h2 { color: var(--primary); font-size: 1.1rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
    .meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
    .badge-green { background: #064e3b; color: var(--primary); }
    .badge-yellow { background: #451a03; color: var(--warn); }
    .badge-red { background: #450a0a; color: var(--danger); }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: #12151f; color: var(--muted); text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
    td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
    tr:hover td { background: #1e2130; }
    .conf-bar { height: 6px; border-radius: 3px; background: #1e2130; margin-top: 4px; }
    .conf-fill { height: 100%; border-radius: 3px; background: var(--primary); }
    .warn-box { background: #451a03; border: 1px solid var(--warn); border-radius: 8px; padding: 0.8rem; color: var(--warn); font-size: 0.85rem; margin-top: 0.5rem; }
    .heatmap-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; margin-top: 0.5rem; }
    .heatmap-grid img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }
    .limitations { background: #12151f; border-left: 3px solid var(--warn); padding: 1rem; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: var(--muted); }
    .footer { margin-top: 2rem; color: var(--muted); font-size: 0.78rem; text-align: center; }
    .status-optimal { color: var(--primary); font-weight: 600; }
    .status-infeasible { color: var(--danger); font-weight: 600; }
  </style>
</head>
<body>
  <h1>♻ E-Waste RecoveryAI</h1>
  <div class="meta">Run ID: <code>{{ run_id }}</code> &nbsp;|&nbsp; {{ timestamp }}</div>

  <div class="card">
    <h2>Device Classification</h2>
    <p><strong>Identified device:</strong> {{ device | upper }}
      <span class="badge badge-green">{{ (device_confidence * 100) | round(1) }}% confidence</span>
    </p>
    <div class="conf-bar"><div class="conf-fill" style="width:{{ (device_confidence * 100) | round(1) }}%"></div></div>
  </div>

  {% if detections %}
  <div class="card">
    <h2>Detected Components</h2>
    <table>
      <thead><tr><th>Label</th><th>KB Key</th><th>Confidence</th><th>Recovery Route</th><th>Est. Value (USD)</th><th>CO₂ Saved (kg)</th></tr></thead>
      <tbody>
      {% for d in detections %}
        <tr>
          <td>{{ d.label }}</td>
          <td><code>{{ d.kb_key or '⚠ unmapped' }}</code></td>
          <td>{{ (d.confidence * 100) | round(1) }}%</td>
          <td>
            {% if d.route %}
              {% if d.route == 'direct_reuse' %}<span class="badge badge-green">{{ d.route }}</span>
              {% elif d.route == 'refurbishment' %}<span class="badge badge-yellow">{{ d.route }}</span>
              {% else %}<span class="badge badge-red">{{ d.route }}</span>{% endif %}
            {% else %}—{% endif %}
          </td>
          <td>{% if d.estimated_value_usd is not none %}${{ d.estimated_value_usd }}{% else %}—{% endif %}</td>
          <td>{% if d.co2_saved_kg is not none %}{{ d.co2_saved_kg }} kg{% else %}—{% endif %}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% if flags %}
    <div class="warn-box">⚠ <strong>Manual inspection recommended</strong> for: {{ flags | join(', ') }}</div>
    {% endif %}
  </div>
  {% endif %}

  {% if unmapped %}
  <div class="card">
    <h2>Unmapped Components</h2>
    <p style="color:var(--warn)">These labels were detected but have no knowledge-base mapping. Please update <code>src/class_map.yaml</code>.</p>
    <ul style="margin-top:0.5rem;padding-left:1.2rem;">
    {% for u in unmapped %}<li><code>{{ u }}</code></li>{% endfor %}
    </ul>
  </div>
  {% endif %}

  {% if optimized %}
  <div class="card">
    <h2>Optimized Recovery Pathway</h2>
    <p>Status: <span class="{{ 'status-optimal' if optimized.status == 'optimal' else 'status-infeasible' }}">{{ optimized.status | upper }}</span></p>
    {% if optimized.status == 'optimal' %}
    <table style="margin-top:0.75rem;max-width:400px">
      <tr><th>Selected components</th><td>{{ optimized.selected | join(', ') }}</td></tr>
      <tr><th>Total value</th><td>${{ optimized.total_value_usd }}</td></tr>
      <tr><th>Total cost</th><td>${{ optimized.total_cost_usd }}</td></tr>
      <tr><th>Total CO₂ saved</th><td>{{ optimized.total_co2_saved_kg }} kg</td></tr>
    </table>
    {% elif optimized.status == 'infeasible' %}
    <p style="color:var(--danger);margin-top:0.5rem">{{ optimized.message }}</p>
    {% endif %}
    {% if optimized.warnings %}
    <div class="warn-box" style="margin-top:0.75rem">Warnings: {{ optimized.warnings | join('; ') }}</div>
    {% endif %}
  </div>
  {% endif %}

  {% if heatmap_paths %}
  <div class="card">
    <h2>Grad-CAM Explainability</h2>
    <div class="heatmap-grid">
    {% for hp in heatmap_paths %}
      <div>
        <p style="font-size:0.75rem;color:var(--muted);margin-bottom:0.3rem">{{ hp.label }}</p>
        <img src="{{ hp.src }}" alt="{{ hp.label }}">
      </div>
    {% endfor %}
    </div>
  </div>
  {% endif %}

  <div class="card">
    <h2>Limitations</h2>
    <div class="limitations">
      Reuse and recovery potential are determined by a knowledge base because no public dataset provides these labels.
      Visible condition is limited to a low-confidence manual-inspection flag.
      RAM and SSD detection is trained on a very small public dataset and is expected to be weaker than PCB part detection.
      Public images are cleaner than real e-waste photos, so real-world accuracy may be lower.
      Aluminium is inferred, not detected.<br><br>
      <em>Future work: collect an annotated condition dataset and train XGBoost/Random Forest with SHAP to replace the rules.</em>
    </div>
  </div>

  <div class="footer">E-Waste RecoveryAI &nbsp;|&nbsp; Results are advisory only — always consult a certified e-waste recycler.</div>
</body>
</html>
"""


def _encode_image(path: str) -> str:
    """Return a base64 data URI for embedding images inline."""
    p = Path(path)
    if not p.exists():
        return ""
    ext = p.suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "png")
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"


def generate(
    device: str,
    device_confidence: float,
    detections: List[Dict[str, Any]],
    optimized: Optional[Dict[str, Any]],
    heatmap_paths: Optional[List[Dict[str, str]]],
    unmapped: Optional[List[str]] = None,
    flags: Optional[List[str]] = None,
) -> Path:
    """Render and save an HTML report. Returns the path to the saved file."""
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Embed heatmap images as base64 so the report is self-contained
    embedded_heatmaps = []
    for hp in (heatmap_paths or []):
        src = _encode_image(hp.get("path", ""))
        if src:
            embedded_heatmaps.append({"label": hp.get("label", ""), "src": src})

    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(TEMPLATE)
    html = tmpl.render(
        run_id=run_id,
        timestamp=timestamp,
        device=device,
        device_confidence=device_confidence,
        detections=detections,
        optimized=optimized,
        heatmap_paths=embedded_heatmaps,
        unmapped=unmapped or [],
        flags=flags or [],
    )

    out_path = REPORTS_DIR / f"run_{run_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] Saved → {out_path}")

    # Optional PDF
    try:
        from weasyprint import HTML as WP
        pdf_path = REPORTS_DIR / f"run_{run_id}.pdf"
        WP(string=html).write_pdf(str(pdf_path))
        print(f"[report] PDF   → {pdf_path}")
    except ImportError:
        pass

    return out_path
