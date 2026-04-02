"""
DDR LangGraph Pipeline.

Builds and compiles a StateGraph that runs all 8 DDR nodes sequentially.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from src.config import Config
from src.graph.state import DDRState
from src.graph.nodes import (
    node_property_summary,
    node_area_observations,
    node_root_causes,
    node_severity,
    node_recommended_actions,
    node_additional_notes,
    node_missing_info,
    node_compile,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DDRPipeline:
    """Builds and runs the LangGraph DDR generation pipeline."""

    def __init__(self, config: Config, vector_store):
        self.config = config
        self.vector_store = vector_store
        self._graph = self._build_graph()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, image_map: Dict[str, Any]) -> DDRState:
        """
        Execute the full DDR pipeline.

        Args:
            image_map: Output of PDFParser.extract_images() —
                       {"inspection": [...], "thermal": [...]}

        Returns:
            Fully populated DDRState dict.
        """
        logger.info("Starting DDR pipeline execution...")
        initial_state: DDRState = {
            "image_map": image_map,
            "errors": [],
        }
        final_state = self._graph.invoke(initial_state)
        return final_state

    # ── Graph construction ──────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """
        Wire up nodes into a parallelised StateGraph.

        Execution order (two levels of parallelism):

        Stage 0:  property_summary   (sequential anchor)
                      |
        Stage 1:  [area_observations, root_causes,         (LangGraph fan-out
                   severity, recommended_actions]           — all 4 run in parallel)
                  +
                  Each of those 4 nodes internally runs 6
                  area sub-tasks with ThreadPoolExecutor.
                      |
        Stage 2:  [additional_notes, missing_info]         (LangGraph fan-out)
                      |
        Stage 3:  compile  (join + validate)

        Total critical-path depth: 4 hops (vs 8 serial).
        Intra-node parallelism further reduces wall time from
        6 serial area calls to ~1 area-call per node.
        """

        # Wrap each node so it receives config + vector_store via closure
        def wrap(node_fn):
            def _node(state: DDRState) -> Dict:
                return node_fn(state, self.config, self.vector_store)
            _node.__name__ = node_fn.__name__
            return _node

        graph = StateGraph(DDRState)

        # ── Register all nodes ────────────────────────────────────────────
        graph.add_node("property_summary",    wrap(node_property_summary))
        graph.add_node("area_observations",   wrap(node_area_observations))
        graph.add_node("root_causes",         wrap(node_root_causes))
        graph.add_node("severity",            wrap(node_severity))
        graph.add_node("recommended_actions", wrap(node_recommended_actions))
        graph.add_node("additional_notes",    wrap(node_additional_notes))
        graph.add_node("missing_info",        wrap(node_missing_info))
        graph.add_node("compile",             wrap(node_compile))

        # ── Stage 0 → Stage 1: fan out to 4 parallel nodes ───────────────
        graph.set_entry_point("property_summary")
        graph.add_edge("property_summary", "area_observations")
        graph.add_edge("property_summary", "root_causes")
        graph.add_edge("property_summary", "severity")
        graph.add_edge("property_summary", "recommended_actions")

        # ── Stage 1 → Stage 2: all 4 nodes feed into 2 parallel nodes ────
        for upstream in ("area_observations", "root_causes",
                         "severity", "recommended_actions"):
            graph.add_edge(upstream, "additional_notes")
            graph.add_edge(upstream, "missing_info")

        # ── Stage 2 → Stage 3: both feed compile ─────────────────────────
        graph.add_edge("additional_notes", "compile")
        graph.add_edge("missing_info",     "compile")
        graph.add_edge("compile",          END)

        compiled = graph.compile()
        logger.info(
            "LangGraph pipeline compiled — parallelised: "
            "Stage1=[obs|roots|severity|actions], Stage2=[notes|missing]"
        )
        return compiled
