"""
mcp_server/ewaste_mcp_server.py
--------------------------------
MCP server exposing two tools:
  1. recommend_recovery  – rule-based recovery route per component
  2. optimize_pathways   – MILP pathway optimizer (pulp)

Run:  python mcp_server/ewaste_mcp_server.py
Then: claude mcp list   ← should show 'ewaste-recovery'

IMPORTANT:
- Recovery logic is rule-based only.  It is NOT a trained model.
- Never report accuracy for this server.
- Every numeric value below must be cited in docs/sources.md.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

import pulp

# ──────────────────────────────────────────────────────────────────────────────
# Knowledge Base
# All values must be replaced with cited figures before the system goes live.
# Citations → docs/sources.md
# ──────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "ram": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 12.50,        # iFixit / eBay completed listings (DDR4 8 GB avg) — cite
        "material_value_usd": 1.80,        # WEEE material recovery estimate — cite
        "recovery_cost_usd": 2.00,
        "co2_saved_direct_reuse_kg": 3.20, # UN Global E-waste Monitor 2024 — cite
        "co2_saved_material_kg": 0.80,
        "materials": ["gold", "copper", "silicon"],
    },
    "ssd": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 18.00,
        "material_value_usd": 2.10,
        "recovery_cost_usd": 2.50,
        "co2_saved_direct_reuse_kg": 4.10,
        "co2_saved_material_kg": 1.00,
        "materials": ["gold", "copper", "silicon", "lithium"],
    },
    "storage_drive": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 8.00,
        "material_value_usd": 1.50,
        "recovery_cost_usd": 2.00,
        "co2_saved_direct_reuse_kg": 2.50,
        "co2_saved_material_kg": 0.70,
        "materials": ["aluminum", "rare_earths", "copper"],
    },
    "pcb": {
        "salvageable": True,
        "hazardous": True,
        "hazard_note": "Lead solder, brominated flame retardants",
        "resale_value_usd": 25.00,
        "material_value_usd": 10.00,
        "recovery_cost_usd": 3.00,
        "co2_saved_direct_reuse_kg": 2.50,
        "co2_saved_material_kg": 2.00,
        "materials": ["gold", "copper", "silver", "palladium"],
    },
    "ic": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 3.00,
        "material_value_usd": 0.80,
        "recovery_cost_usd": 1.00,
        "co2_saved_direct_reuse_kg": 0.90,
        "co2_saved_material_kg": 0.30,
        "materials": ["gold", "silicon"],
    },
    "connector": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 1.50,
        "material_value_usd": 0.40,
        "recovery_cost_usd": 0.50,
        "co2_saved_direct_reuse_kg": 0.20,
        "co2_saved_material_kg": 0.10,
        "materials": ["copper", "gold"],
    },
    "battery": {
        "salvageable": False,
        "hazardous": True,
        "hazard_note": "Lithium/lead-acid — certified recycling required (WEEE Directive Annex XIV)",
        "resale_value_usd": 0.00,
        "material_value_usd": 2.00,
        "recovery_cost_usd": 1.50,
        "co2_saved_direct_reuse_kg": 0.00,
        "co2_saved_material_kg": 1.20,
        "materials": ["lithium", "cobalt", "nickel"],
    },
    "aluminium": {
        "salvageable": False,
        "hazardous": False,
        "resale_value_usd": 0.50,
        "material_value_usd": 0.50,
        "recovery_cost_usd": 0.30,
        "co2_saved_direct_reuse_kg": 0.00,
        "co2_saved_material_kg": 1.50,
        "materials": ["aluminium"],
        "note": "Inferred from heatsink detection; aluminium is not directly detected",
    },
    "cpu": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 35.00,
        "material_value_usd": 4.00,
        "recovery_cost_usd": 4.00,
        "co2_saved_direct_reuse_kg": 8.00,
        "co2_saved_material_kg": 2.50,
        "materials": ["gold", "copper", "silicon"],
    },
    "gpu": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 60.00,
        "material_value_usd": 6.00,
        "recovery_cost_usd": 5.00,
        "co2_saved_direct_reuse_kg": 12.00,
        "co2_saved_material_kg": 3.00,
        "materials": ["gold", "copper", "silicon", "rare_earths"],
    },
    "psu": {
        "salvageable": True,
        "hazardous": True,
        "hazard_note": "Capacitors may retain charge; handle with care",
        "resale_value_usd": 10.00,
        "material_value_usd": 2.00,
        "recovery_cost_usd": 3.00,
        "co2_saved_direct_reuse_kg": 2.00,
        "co2_saved_material_kg": 1.00,
        "materials": ["copper", "aluminium", "steel"],
    },
    "mobile": {
        "salvageable": True,
        "hazardous": True,
        "hazard_note": "Battery, rare earths",
        "resale_value_usd": 65.00,
        "material_value_usd": 10.00,
        "recovery_cost_usd": 6.00,
        "co2_saved_direct_reuse_kg": 45.00,
        "co2_saved_material_kg": 12.00,
        "materials": ["gold", "silver", "cobalt", "rare_earths"],
    },
    "computer": {
        "salvageable": True,
        "hazardous": True,
        "resale_value_usd": 120.00,
        "material_value_usd": 18.00,
        "recovery_cost_usd": 12.00,
        "co2_saved_direct_reuse_kg": 300.00,
        "co2_saved_material_kg": 35.00,
        "materials": ["gold", "copper", "aluminium", "steel"],
    },
    "keyboard": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 8.00,
        "material_value_usd": 1.50,
        "recovery_cost_usd": 1.00,
        "co2_saved_direct_reuse_kg": 0.50,
        "co2_saved_material_kg": 0.20,
        "materials": ["plastic", "copper"],
    },
    "mouse": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 6.00,
        "material_value_usd": 1.00,
        "recovery_cost_usd": 0.80,
        "co2_saved_direct_reuse_kg": 0.30,
        "co2_saved_material_kg": 0.10,
        "materials": ["plastic", "copper"],
    },
    "printer": {
        "salvageable": True,
        "hazardous": True,
        "hazard_note": "Toner cartridges",
        "resale_value_usd": 35.00,
        "material_value_usd": 8.00,
        "recovery_cost_usd": 6.00,
        "co2_saved_direct_reuse_kg": 30.00,
        "co2_saved_material_kg": 10.00,
        "materials": ["plastic", "copper", "steel"],
    },
    "tv": {
        "salvageable": True,
        "hazardous": True,
        "hazard_note": "LCD backlight may contain mercury",
        "resale_value_usd": 75.00,
        "material_value_usd": 15.00,
        "recovery_cost_usd": 10.00,
        "co2_saved_direct_reuse_kg": 80.00,
        "co2_saved_material_kg": 18.00,
        "materials": ["plastic", "copper", "rare_earths"],
    },
    "speaker": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 22.00,
        "material_value_usd": 5.00,
        "recovery_cost_usd": 3.00,
        "co2_saved_direct_reuse_kg": 1.00,
        "co2_saved_material_kg": 0.40,
        "materials": ["copper", "magnets", "plastic"],
    },
    "washing_machine": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 90.00,
        "material_value_usd": 25.00,
        "recovery_cost_usd": 15.00,
        "co2_saved_direct_reuse_kg": 100.00,
        "co2_saved_material_kg": 25.00,
        "materials": ["steel", "copper", "plastic"],
    },
    "microwave": {
        "salvageable": True,
        "hazardous": True,
        "hazard_note": "Magnetron contains beryllium oxide",
        "resale_value_usd": 35.00,
        "material_value_usd": 12.00,
        "recovery_cost_usd": 6.00,
        "co2_saved_direct_reuse_kg": 25.00,
        "co2_saved_material_kg": 8.00,
        "materials": ["steel", "copper"],
    },
    "player": {
        "salvageable": True,
        "hazardous": False,
        "resale_value_usd": 18.00,
        "material_value_usd": 4.00,
        "recovery_cost_usd": 3.00,
        "co2_saved_direct_reuse_kg": 3.00,
        "co2_saved_material_kg": 0.80,
        "materials": ["plastic", "copper"],
    },
    "metal": {
        "salvageable": False,
        "hazardous": False,
        "resale_value_usd": 0.10,
        "material_value_usd": 0.10,
        "recovery_cost_usd": 0.20,
        "co2_saved_direct_reuse_kg": 0.00,
        "co2_saved_material_kg": 0.05,
        "materials": ["mixed_metal"],
    },
}

CONFIDENCE_THRESHOLD = 0.50  # below this → needs_manual_inspection flag

# ──────────────────────────────────────────────────────────────────────────────
# Rule-based recovery logic
# ──────────────────────────────────────────────────────────────────────────────

def _recommend_recovery(
    component: str,
    age_years: Optional[int],
    working: Optional[bool],
    confidence: float,
) -> Dict[str, Any]:
    """
    Pure rule-based recovery recommendation.
    Returns a dict — no ML, no accuracy metric.
    """
    kb = KNOWLEDGE_BASE.get(component)
    flags: List[str] = []

    if confidence < CONFIDENCE_THRESHOLD:
        flags.append("needs_manual_inspection")

    if kb is None:
        return {
            "component": component,
            "route": None,
            "estimated_value_usd": None,
            "co2_saved_kg": None,
            "flags": flags + ["unknown_component_not_in_kb"],
            "citations": [],
            "warning": f"Component '{component}' is not in the knowledge base.",
        }

    if kb.get("hazardous"):
        flags.append("hazardous_material")

    # ── Decision rules ────────────────────────────────────────────────────────
    if component == "battery" or (not kb.get("salvageable", True) and component != "battery"):
        route = "certified_recycling"
        value = kb["material_value_usd"]
        co2 = kb["co2_saved_material_kg"]
    elif working is True and (age_years is None or age_years < 5):
        route = "direct_reuse"
        value = kb["resale_value_usd"]
        co2 = kb["co2_saved_direct_reuse_kg"]
    elif working is True and age_years is not None and age_years >= 5:
        route = "refurbishment"
        # Dynamic age depreciation (minimum floor of 45% of resale value)
        deprec = max(0.45, 0.70 - (age_years - 5) * 0.04)
        value = kb["resale_value_usd"] * deprec
        co2 = kb["co2_saved_direct_reuse_kg"] * 0.7
    elif working is False:
        route = "material_recovery" if kb.get("salvageable") else "certified_recycling"
        value = kb["material_value_usd"]
        co2 = kb["co2_saved_material_kg"]
    else:
        # working unknown - balanced estimate between refurbished resale and scrap baseline
        route = "refurbishment"
        value = kb["resale_value_usd"] * 0.55 + kb["material_value_usd"] * 0.45
        co2 = kb["co2_saved_direct_reuse_kg"] * 0.5
        flags.append("working_status_unknown")

    return {
        "component": component,
        "route": route,
        "estimated_value_usd": round(value, 2),
        "resale_value_usd": round(float(kb.get("resale_value_usd", 0.0)), 2),
        "material_value_usd": round(float(kb.get("material_value_usd", 0.0)), 2),
        "co2_saved_kg": round(co2, 2),
        "hazardous": kb.get("hazardous", False),
        "hazard_note": kb.get("hazard_note"),
        "materials": kb.get("materials", []),
        "flags": flags,
        "citations": [
            "UN Global E-waste Monitor 2024",
            "WEEE Directive (2012/19/EU)",
            "iFixit Repairability Scores",
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# MILP pathway optimizer
# ──────────────────────────────────────────────────────────────────────────────

def _optimize_pathways(
    components: List[Dict[str, Any]],
    budget_usd: float,
    objective: str = "maximize_value",
) -> Dict[str, Any]:
    """
    Binary MILP: select components to process within budget.
    Maximises total value (or CO₂ saving) subject to cost ≤ budget.
    """
    warnings: List[str] = []

    if not components:
        return {"status": "no_components", "selected": [], "warnings": warnings}

    # Filter out unknown components
    valid = []
    for c in components:
        if c.get("kb_key") and c["kb_key"] not in ("UNMAPPED", None) and c["kb_key"] in KNOWLEDGE_BASE:
            valid.append(c)
        else:
            warnings.append(f"Skipped unknown component: {c.get('kb_key', '?')}")

    if not valid:
        return {"status": "no_valid_components", "selected": [], "warnings": warnings}

    prob = pulp.LpProblem("ewaste_pathway", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(valid))}

    # Objective
    if objective == "maximize_co2_saving":
        prob += pulp.lpSum(valid[i]["co2_saved_kg"] * x[i] for i in range(len(valid)))
    else:
        prob += pulp.lpSum(valid[i]["estimated_value_usd"] * x[i] for i in range(len(valid)))

    # Budget constraint
    prob += pulp.lpSum(valid[i]["recovery_cost_usd"] * x[i] for i in range(len(valid))) <= budget_usd

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[status] == "Infeasible" or all(
        pulp.value(x[i]) == 0 for i in range(len(valid))
    ):
        # Check if truly infeasible (all costs > budget)
        if all(c["recovery_cost_usd"] > budget_usd for c in valid):
            return {
                "status": "infeasible",
                "selected": [],
                "total_value_usd": 0.0,
                "total_cost_usd": 0.0,
                "total_co2_saved_kg": 0.0,
                "message": "Budget too low for any pathway.",
                "warnings": warnings,
            }

    selected_keys = []
    total_value = 0.0
    total_cost = 0.0
    total_co2 = 0.0

    for i in range(len(valid)):
        if pulp.value(x[i]) and pulp.value(x[i]) > 0.5:
            selected_keys.append(valid[i]["kb_key"])
            total_value += valid[i]["estimated_value_usd"]
            total_cost += valid[i]["recovery_cost_usd"]
            total_co2 += valid[i]["co2_saved_kg"]

    return {
        "status": "optimal",
        "selected": selected_keys,
        "total_value_usd": round(total_value, 2),
        "total_cost_usd": round(total_cost, 2),
        "total_co2_saved_kg": round(total_co2, 2),
        "warnings": warnings,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Standalone callable wrappers (used by pipeline.py directly)
# ──────────────────────────────────────────────────────────────────────────────

def recommend_recovery(
    component: str,
    age_years: Optional[int] = None,
    working: Optional[bool] = None,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    return _recommend_recovery(component, age_years, working, confidence)


def optimize_pathways(
    components: List[Dict[str, Any]],
    budget_usd: float = 20.0,
    objective: str = "maximize_value",
) -> Dict[str, Any]:
    return _optimize_pathways(components, budget_usd, objective)


# ──────────────────────────────────────────────────────────────────────────────
# MCP Server (if mcp package is installed)
# ──────────────────────────────────────────────────────────────────────────────

def run_mcp_server() -> None:
    if not MCP_AVAILABLE:
        print("[MCP] 'mcp' package not installed — run: pip install mcp")
        return

    server = Server("ewaste-recovery")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="recommend_recovery",
                description=(
                    "Rule-based recovery recommendation for an e-waste component. "
                    "Returns route, estimated value, CO2 saved, and hazard flags. "
                    "NOT a trained model — no accuracy metric applies."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "component": {"type": "string", "description": "Knowledge-base key (e.g. ram, ssd, pcb)"},
                        "age_years": {"type": ["integer", "null"], "description": "Device age in years (nullable)"},
                        "working": {"type": ["boolean", "null"], "description": "Working status (nullable)"},
                        "confidence": {"type": "number", "description": "Detector confidence 0-1"},
                    },
                    "required": ["component"],
                },
            ),
            Tool(
                name="optimize_pathways",
                description=(
                    "MILP optimizer that selects the best recovery pathways within a budget. "
                    "Returns selected components, total value, cost, and CO2 saved."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kb_key": {"type": "string"},
                                    "route": {"type": "string"},
                                    "estimated_value_usd": {"type": "number"},
                                    "recovery_cost_usd": {"type": "number"},
                                    "co2_saved_kg": {"type": "number"},
                                },
                            },
                        },
                        "budget_usd": {"type": "number", "default": 20.0},
                        "objective": {"type": "string", "enum": ["maximize_value", "maximize_co2_saving"]},
                    },
                    "required": ["components"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        if name == "recommend_recovery":
            result = _recommend_recovery(
                component=arguments["component"],
                age_years=arguments.get("age_years"),
                working=arguments.get("working"),
                confidence=arguments.get("confidence", 1.0),
            )
        elif name == "optimize_pathways":
            result = _optimize_pathways(
                components=arguments["components"],
                budget_usd=arguments.get("budget_usd", 20.0),
                objective=arguments.get("objective", "maximize_value"),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    import asyncio
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    run_mcp_server()
