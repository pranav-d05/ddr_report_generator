"""
LangGraph State Schema for the DDR pipeline.

DDRState is a TypedDict that flows through every node.
Each node reads from and writes back into this shared dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TypedDict


class DDRState(TypedDict, total=False):
    # ── Inputs ─────────────────────────────────────────────────────────────────
    # image_map is injected before the graph starts
    image_map: Dict[str, List[Dict[str, Any]]]
    # "inspection" → list of image metadata dicts
    # "thermal"    → list of image metadata dicts

    # ── Node outputs (sections of the DDR) ────────────────────────────────────
    property_summary: str
    # Node 1 — high-level overview of all identified issues

    area_observations: Dict[str, str]
    # Node 2 — {area_key: observation_text}

    area_images: Dict[str, List[Path]]
    # Node 2 — {area_key: [image_path, ...]} — images assigned per area

    root_causes: Dict[str, str]
    # Node 3 — {area_key: root_cause_text}

    severity: Dict[str, str]
    # Node 4 — {area_key: "Critical / reason..."  |  "High / ..."  etc.}

    recommended_actions: Dict[str, str]
    # Node 5 — {area_key: action_text}

    additional_notes: str
    # Node 6 — precautions, limitations, follow-up suggestions

    missing_info: str
    # Node 7 — items not found / conflicting in source docs

    # ── Meta ───────────────────────────────────────────────────────────────────
    errors: List[str]
    # Accumulated non-fatal errors / warnings from nodes
