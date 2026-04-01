"""
Smoke tests for the LangGraph DDR pipeline nodes and full graph execution.
Uses shared fixtures from conftest.py. No real LLM or API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.graph.state import DDRState
from src.graph.nodes import DDR_AREAS


# ── Node unit tests ───────────────────────────────────────────────────────────

class TestNodes:

    @patch("src.graph.nodes._build_llm")
    def test_property_summary_node(self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response):
        mock_build_llm.return_value = mock_llm_response("High-level summary of all property issues.")

        from src.graph.nodes import node_property_summary
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_property_summary(state, mock_config, mock_vector_store)

        assert "property_summary" in result
        assert isinstance(result["property_summary"], str)
        assert len(result["property_summary"]) > 0

    @patch("src.graph.nodes._build_llm")
    def test_area_observations_node_covers_all_areas(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response, empty_image_map
    ):
        mock_build_llm.return_value = mock_llm_response("Observations for this area.")

        from src.graph.nodes import node_area_observations
        state: DDRState = {"image_map": empty_image_map, "errors": []}
        result = node_area_observations(state, mock_config, mock_vector_store)

        assert "area_observations" in result
        assert "area_images" in result
        # Every DDR area must have an entry
        for area in DDR_AREAS:
            assert area in result["area_observations"], f"Missing area: {area}"

    @patch("src.graph.nodes._build_llm")
    def test_area_images_dict_structure(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response, empty_image_map
    ):
        """area_images must be {area: {"visual": [...], "thermal": [...]}}."""
        mock_build_llm.return_value = mock_llm_response("Observations.")

        from src.graph.nodes import node_area_observations
        state: DDRState = {"image_map": empty_image_map, "errors": []}
        result = node_area_observations(state, mock_config, mock_vector_store)

        for area in DDR_AREAS:
            img_dict = result["area_images"][area]
            assert isinstance(img_dict, dict), f"{area}: area_images entry must be a dict"
            assert "visual"  in img_dict, f"{area}: missing 'visual' key"
            assert "thermal" in img_dict, f"{area}: missing 'thermal' key"

    @patch("src.graph.nodes._build_llm")
    def test_root_causes_node_covers_all_areas(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response
    ):
        mock_build_llm.return_value = mock_llm_response("Root cause: capillary rise from tile joints.")

        from src.graph.nodes import node_root_causes
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_root_causes(state, mock_config, mock_vector_store)

        assert "root_causes" in result
        for area in DDR_AREAS:
            assert area in result["root_causes"]

    @patch("src.graph.nodes._build_llm")
    def test_severity_node_all_areas(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response
    ):
        mock_build_llm.return_value = mock_llm_response(
            "Severity: High\nReasoning: Active water ingress present."
        )

        from src.graph.nodes import node_severity
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_severity(state, mock_config, mock_vector_store)

        assert "severity" in result
        for area in DDR_AREAS:
            assert area in result["severity"]

    @patch("src.graph.nodes._build_llm")
    def test_recommended_actions_node(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response
    ):
        mock_build_llm.return_value = mock_llm_response("Apply crystalline waterproofing compound.")

        from src.graph.nodes import node_recommended_actions
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_recommended_actions(state, mock_config, mock_vector_store)

        assert "recommended_actions" in result
        for area in DDR_AREAS:
            assert area in result["recommended_actions"]

    @patch("src.graph.nodes._build_llm")
    def test_additional_notes_node(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response
    ):
        mock_build_llm.return_value = mock_llm_response("Allow 48 hours curing time before testing.")

        from src.graph.nodes import node_additional_notes
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_additional_notes(state, mock_config, mock_vector_store)

        assert "additional_notes" in result
        assert isinstance(result["additional_notes"], str)

    @patch("src.graph.nodes._build_llm")
    def test_missing_info_node(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response
    ):
        mock_build_llm.return_value = mock_llm_response("No significant information gaps identified.")

        from src.graph.nodes import node_missing_info
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_missing_info(state, mock_config, mock_vector_store)

        assert "missing_info" in result
        assert isinstance(result["missing_info"], str)

    def test_compile_node_flags_missing_sections(self, mock_config, mock_vector_store):
        """compile node with empty state should log missing sections but not crash."""
        from src.graph.nodes import node_compile
        state: DDRState = {"image_map": {}, "errors": []}
        result = node_compile(state, mock_config, mock_vector_store)
        assert "errors" in result

    def test_compile_node_with_all_sections_populated(self, mock_config, mock_vector_store):
        """compile node with a fully populated state should return errors=[]."""
        from src.graph.nodes import node_compile
        state: DDRState = {
            "image_map": {},
            "errors": [],
            "property_summary":    "Summary text",
            "area_observations":   {a: "obs" for a in DDR_AREAS},
            "root_causes":         {a: "cause" for a in DDR_AREAS},
            "severity":            {a: "Severity: High\nReasoning: test" for a in DDR_AREAS},
            "recommended_actions": {a: "action" for a in DDR_AREAS},
            "additional_notes":    "Notes text",
            "missing_info":        "None",
        }
        result = node_compile(state, mock_config, mock_vector_store)
        assert result["errors"] == []


# ── Full pipeline smoke test ──────────────────────────────────────────────────

class TestPipeline:

    @patch("src.graph.nodes._build_llm")
    def test_pipeline_runs_and_returns_all_sections(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response, empty_image_map
    ):
        """Full graph execution (mocked LLM) must populate all 7 DDR sections."""
        mock_build_llm.return_value = mock_llm_response(
            "Severity: Medium\nReasoning: Cosmetic cracks only."
        )

        from src.graph.pipeline import DDRPipeline
        pipeline = DDRPipeline(mock_config, mock_vector_store)
        result = pipeline.run(image_map=empty_image_map)

        required_keys = [
            "property_summary",
            "area_observations",
            "root_causes",
            "severity",
            "recommended_actions",
            "additional_notes",
            "missing_info",
        ]
        for key in required_keys:
            assert key in result, f"Missing section in pipeline output: {key}"

    @patch("src.graph.nodes._build_llm")
    def test_pipeline_area_images_structure(
        self, mock_build_llm, mock_config, mock_vector_store, mock_llm_response, empty_image_map
    ):
        """area_images in pipeline output must have visual/thermal sub-dicts."""
        mock_build_llm.return_value = mock_llm_response("Section output.")

        from src.graph.pipeline import DDRPipeline
        pipeline = DDRPipeline(mock_config, mock_vector_store)
        result = pipeline.run(image_map=empty_image_map)

        for area in DDR_AREAS:
            img_dict = result["area_images"][area]
            assert "visual"  in img_dict
            assert "thermal" in img_dict
            assert isinstance(img_dict["visual"],  list)
            assert isinstance(img_dict["thermal"], list)
