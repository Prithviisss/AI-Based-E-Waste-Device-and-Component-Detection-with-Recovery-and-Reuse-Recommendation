"""
tests/test_optimizer.py
------------------------
Unit tests for the MILP optimize_pathways tool.
Covers: normal case, infeasible budget, unknown kb_key, empty list.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.ewaste_mcp_server import optimize_pathways


def make_components(*keys, value=10.0, cost=2.0, co2=3.0):
    return [
        {"kb_key": k, "route": "direct_reuse", "estimated_value_usd": value,
         "recovery_cost_usd": cost, "co2_saved_kg": co2}
        for k in keys
    ]


class TestOptimizerNormal:
    def test_status_optimal(self):
        comps = make_components("ram", "ssd")
        res = optimize_pathways(comps, budget_usd=20.0)
        assert res["status"] == "optimal"

    def test_total_cost_within_budget(self):
        comps = make_components("ram", "ssd", cost=5.0)
        res = optimize_pathways(comps, budget_usd=20.0)
        assert res["total_cost_usd"] <= 20.0

    def test_selected_nonempty(self):
        comps = make_components("cpu")
        res = optimize_pathways(comps, budget_usd=50.0)
        assert len(res["selected"]) > 0

    def test_total_value_positive(self):
        comps = make_components("gpu", value=60.0, cost=5.0)
        res = optimize_pathways(comps, budget_usd=50.0)
        assert res["total_value_usd"] > 0

    def test_co2_objective(self):
        comps = make_components("ram", "ssd", co2=5.0)
        res = optimize_pathways(comps, budget_usd=20.0, objective="maximize_co2_saving")
        assert res["status"] == "optimal"
        assert res["total_co2_saved_kg"] >= 0


class TestOptimizerInfeasible:
    def test_infeasible_budget_zero(self):
        """Budget = 0.01 — too low for any component."""
        comps = make_components("ram", "cpu", cost=2.0)
        res = optimize_pathways(comps, budget_usd=0.01)
        assert res["status"] == "infeasible"

    def test_infeasible_message_present(self):
        comps = make_components("ram", cost=100.0)
        res = optimize_pathways(comps, budget_usd=0.01)
        assert "message" in res
        assert len(res["message"]) > 0

    def test_infeasible_selected_empty(self):
        comps = make_components("cpu", cost=999.0)
        res = optimize_pathways(comps, budget_usd=0.01)
        assert res["selected"] == []


class TestOptimizerUnknownComponent:
    def test_unknown_key_in_warnings(self):
        comps = make_components("ram") + [
            {"kb_key": "TOTALLY_UNKNOWN", "estimated_value_usd": 50,
             "recovery_cost_usd": 1, "co2_saved_kg": 10}
        ]
        res = optimize_pathways(comps, budget_usd=50.0)
        assert any("TOTALLY_UNKNOWN" in w for w in res.get("warnings", []))

    def test_unknown_key_not_in_selected(self):
        comps = make_components("ssd") + [
            {"kb_key": "UNMAPPED_THING", "estimated_value_usd": 999,
             "recovery_cost_usd": 0.1, "co2_saved_kg": 99}
        ]
        res = optimize_pathways(comps, budget_usd=100.0)
        assert "UNMAPPED_THING" not in res.get("selected", [])

    def test_known_still_selected_with_unknown(self):
        comps = make_components("ram", cost=2.0) + [
            {"kb_key": "MYSTERY_BOX", "estimated_value_usd": 999,
             "recovery_cost_usd": 0.1, "co2_saved_kg": 99}
        ]
        res = optimize_pathways(comps, budget_usd=50.0)
        assert "ram" in res.get("selected", [])


class TestOptimizerEdgeCases:
    def test_empty_list(self):
        res = optimize_pathways([], budget_usd=50.0)
        assert res["status"] in ("no_components", "no_valid_components")

    def test_all_unmapped(self):
        comps = [
            {"kb_key": "UNMAPPED", "estimated_value_usd": 5,
             "recovery_cost_usd": 1, "co2_saved_kg": 2}
        ]
        res = optimize_pathways(comps, budget_usd=50.0)
        assert res["status"] in ("no_components", "no_valid_components", "infeasible")

    def test_single_component(self):
        comps = make_components("ssd", cost=3.0)
        res = optimize_pathways(comps, budget_usd=10.0)
        assert res["status"] == "optimal"
        assert "ssd" in res["selected"]
