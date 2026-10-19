"""
tests/test_rules.py
--------------------
Unit tests for the MCP rule-based recovery logic.
Covers: route assignment, confidence flag, hazardous flag, unknown component.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.ewaste_mcp_server import recommend_recovery


class TestRecoveryRoutes:
    def test_working_young_direct_reuse(self):
        rec = recommend_recovery("ram", age_years=3, working=True, confidence=0.9)
        assert rec["route"] == "direct_reuse"

    def test_working_old_refurbishment(self):
        rec = recommend_recovery("ram", age_years=7, working=True, confidence=0.9)
        assert rec["route"] == "refurbishment"

    def test_broken_salvageable_material_recovery(self):
        rec = recommend_recovery("ram", age_years=5, working=False, confidence=0.9)
        assert rec["route"] == "material_recovery"

    def test_battery_not_direct_reuse(self):
        """Battery is not salvageable — must never be direct_reuse."""
        rec = recommend_recovery("battery", age_years=2, working=True, confidence=0.9)
        assert rec["route"] != "direct_reuse"
        assert rec["route"] == "certified_recycling"

    def test_battery_hazardous_flag(self):
        rec = recommend_recovery("battery", age_years=2, working=True, confidence=0.9)
        assert rec["hazardous"] is True
        assert "hazardous_material" in rec["flags"]

    def test_unknown_working_working_status_flag(self):
        rec = recommend_recovery("ssd", age_years=None, working=None, confidence=0.9)
        assert "working_status_unknown" in rec["flags"]

    def test_value_is_positive(self):
        rec = recommend_recovery("cpu", age_years=2, working=True, confidence=0.95)
        assert rec["estimated_value_usd"] > 0

    def test_co2_saved_nonnegative(self):
        rec = recommend_recovery("pcb", age_years=None, working=False, confidence=0.8)
        assert rec["co2_saved_kg"] >= 0


class TestConfidenceFlag:
    def test_low_conf_manual_inspection(self):
        rec = recommend_recovery("ram", age_years=2, working=True, confidence=0.3)
        assert "needs_manual_inspection" in rec["flags"]

    def test_high_conf_no_manual_inspection(self):
        rec = recommend_recovery("ram", age_years=2, working=True, confidence=0.95)
        assert "needs_manual_inspection" not in rec["flags"]

    def test_boundary_conf(self):
        # exactly at threshold — should NOT flag
        rec = recommend_recovery("ram", age_years=2, working=True, confidence=0.50)
        assert "needs_manual_inspection" not in rec["flags"]

    def test_just_below_threshold(self):
        rec = recommend_recovery("ram", age_years=2, working=True, confidence=0.49)
        assert "needs_manual_inspection" in rec["flags"]


class TestUnknownComponent:
    def test_unknown_component_warning(self):
        rec = recommend_recovery("quantum_flux_capacitor", confidence=0.9)
        assert rec["route"] is None
        assert any("not in the knowledge base" in (w or "") for w in [rec.get("warning", "")])

    def test_unknown_component_in_output(self):
        rec = recommend_recovery("imaginary_device", confidence=0.9)
        assert "unknown_component_not_in_kb" in rec["flags"]
