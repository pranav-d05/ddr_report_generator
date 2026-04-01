"""
Smoke tests for the LangGraph DDR pipeline.

Mocks the LLM and vector store so the full graph can run without API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.graph.state import DDRState
from src.graph.nodes import DDR_AREAS


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_mock_llm(response_text: str = "Test LLM output for this section."):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _make_mock_vector_store():
    from langchain_core.documents import Document
    mock_vs = MagicMock()
    mock_vs.retrieve.return_value = [
        Document(
            page_content="Bathroom shows seepage and dampness.",
            metadata={"source": "test.pdf", "doc_type": "inspection", "page": 1},
        )
    ]
    return mock_vs


def _make_mock_config():
    cfg = MagicMock()
    cfg.openrouter_model = "test-model"
    cfg.openrouter_api_key = "test-key"
    cfg.openrouter_base_url = "https://openrouter.ai/api/v1"
    cfg.retrieval_k = 3
    return cfg


# ── Node unit tests ───────────────────────────────────────────────────────────

class TestNodes:

    @patch("src.graph.nodes._build_llm")
    def test_property_summary_node(self, mock_build_llm):
        mock_build_llm.return_value = _make_mock_llm("High-level summary of property issues.")
        vs = _make_mock_vector_store()
        cfg = _make_mock_config()

        from src.graph.nodes import node_property_summary
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_property_summary(state, cfg, vs)

        assert "property_summary" in result
        assert isinstance(result["property_summary"], str)
        assert len(result["property_summary"]) > 0

    @patch("src.graph.nodes._build_llm")
    def test_area_observations_node(self, mock_build_llm):
        mock_build_llm.return_value = _make_mock_llm("Observations for this area.")
        vs = _make_mock_vector_store()
        cfg = _make_mock_config()

        from src.graph.nodes import node_area_observations
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_area_observations(state, cfg, vs)

        assert "area_observations" in result
        assert "area_images" in result
        # All DDR areas should have an entry
        for area in DDR_AREAS:
            assert area in result["area_observations"]

    @patch("src.graph.nodes._build_llm")
    def test_severity_node_all_areas(self, mock_build_llm):
        mock_build_llm.return_value = _make_mock_llm(
            "Severity: High\nReasoning: Active water ingress present."
        )
        vs = _make_mock_vector_store()
        cfg = _make_mock_config()

        from src.graph.nodes import node_severity
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_severity(state, cfg, vs)

        assert "severity" in result
        for area in DDR_AREAS:
            assert area in result["severity"]

    @patch("src.graph.nodes._build_llm")
    def test_missing_info_node(self, mock_build_llm):
        mock_build_llm.return_value = _make_mock_llm("No significant information gaps identified.")
        vs = _make_mock_vector_store()
        cfg = _make_mock_config()

        from src.graph.nodes import node_missing_info
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_missing_info(state, cfg, vs)

        assert "missing_info" in result
        assert isinstance(result["missing_info"], str)

    def test_compile_node_flags_missing_sections(self):
        from src.graph.nodes import node_compile
        # State with no sections populated
        state: DDRState = {"image_map": {}, "errors": []}
        cfg = _make_mock_config()
        vs = _make_mock_vector_store()
        result = node_compile(state, cfg, vs)
        assert "errors" in result


# ── Full pipeline smoke test ──────────────────────────────────────────────────

class TestPipeline:

    @patch("src.graph.nodes._build_llm")
    def test_pipeline_runs_and_returns_state(self, mock_build_llm):
        mock_build_llm.return_value = _make_mock_llm(
            "Severity: Medium\nReasoning: Cosmetic cracks only."
        )
        vs = _make_mock_vector_store()
        cfg = _make_mock_config()

        from src.graph.pipeline import DDRPipeline
        pipeline = DDRPipeline(cfg, vs)
        result = pipeline.run(image_map={})

        # All 7 content sections should be present
        assert "property_summary" in result
        assert "area_observations" in result
        assert "root_causes" in result
        assert "severity" in result
        assert "recommended_actions" in result
        assert "additional_notes" in result
        assert "missing_info" in result
