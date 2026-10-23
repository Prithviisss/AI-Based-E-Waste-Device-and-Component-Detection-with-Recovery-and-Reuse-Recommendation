"""
src/utils/metrics_writer.py
----------------------------
Single writer for reports/metrics.json.
ALL modules must use this — never write metrics.json by hand.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

METRICS_PATH = Path(__file__).resolve().parents[2] / "reports" / "metrics.json"


def _load() -> Dict[str, Any]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seed": 42, "_updated": None, "classifier": {}, "detector_pcb": {}, "detector_ramssd": {}}


def _save(data: Dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["_updated"] = datetime.now(timezone.utc).isoformat()
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[metrics] Saved → {METRICS_PATH}")


def update_section(section: str, payload: Dict[str, Any]) -> None:
    """Merge *payload* into the named top-level section of metrics.json."""
    data = _load()
    if section not in data:
        data[section] = {}
    data[section].update(payload)
    _save(data)


def get_section(section: str) -> Dict[str, Any]:
    return _load().get(section, {})


def get_all() -> Dict[str, Any]:
    return _load()
