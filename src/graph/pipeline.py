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
        """Wire up all nodes into a sequential StateGraph and compile it."""

        # Wrap each node so it receives config + vector_store via closure
        def wrap(node_fn):
            def _node(state: DDRState) -> Dict:
                return node_fn(state, self.config, self.vector_store)
            _node.__name__ = node_fn.__name__
            return _node

        graph = StateGraph(DDRState)

        # Register nodes
        graph.add_node("property_summary",    wrap(node_property_summary))
        graph.add_node("area_observations",   wrap(node_area_observations))
        graph.add_node("root_causes",         wrap(node_root_causes))
        graph.add_node("severity",            wrap(node_severity))
        graph.add_node("recommended_actions", wrap(node_recommended_actions))
        graph.add_node("additional_notes",    wrap(node_additional_notes))
        graph.add_node("missing_info",        wrap(node_missing_info))
        graph.add_node("compile",             wrap(node_compile))

        # Sequential edges
        graph.set_entry_point("property_summary")
        graph.add_edge("property_summary",    "area_observations")
        graph.add_edge("area_observations",   "root_causes")
        graph.add_edge("root_causes",         "severity")
        graph.add_edge("severity",            "recommended_actions")
        graph.add_edge("recommended_actions", "additional_notes")
        graph.add_edge("additional_notes",    "missing_info")
        graph.add_edge("missing_info",        "compile")
        graph.add_edge("compile",             END)

        compiled = graph.compile()
        logger.info("LangGraph pipeline compiled — 8 nodes, sequential execution")
        return compiled
