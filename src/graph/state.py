"""
LangGraph State Schema for the DDR pipeline.

DDRState is a TypedDict that flows through every node.
Each node reads from and writes back into this shared dict.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Annotated


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    """Reducer to merge dictionaries in LangGraph safely across parallel nodes."""
    res = (a or {}).copy()
    res.update(b or {})
    return res


class DDRState(TypedDict, total=False):
    # ── Inputs ─────────────────────────────────────────────────────────────────
    # image_map is injected before the graph starts
    image_map: Dict[str, List[Dict[str, Any]]]
    # "inspection" → list of image metadata dicts
    # "thermal"    → list of image metadata dicts

    # ── Pre-fetched context cache (populated by Node 1, read by Nodes 2-7) ─────
    # Avoids redundant ChromaDB calls: retrieve once, share across all nodes.
    # Shape: {area_key: {"inspection": str_context, "thermal": str_context}}
    # Special keys: "global" = cross-area context for notes/missing_info.
    prefetched_context: Dict[str, Dict[str, str]]

    # ── Timing telemetry (ms per node, for latency analysis) ────────────────
    timings: Annotated[Dict[str, float], merge_dicts]

    # ── Node outputs (sections of the DDR) ────────────────────────────────────
    property_summary: str
    # Node 1 — high-level overview of all identified issues

    area_observations: Dict[str, str]
    # Node 2 — {area_key: observation_text}

    area_images: Dict[str, Dict[str, List[Path]]]
    # Node 2 — {area_key: {"visual": [Path, ...], "thermal": [Path, ...]}} — images assigned per area

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
    errors: Annotated[List[str], operator.add]
    # Accumulated non-fatal errors / warnings from nodes
