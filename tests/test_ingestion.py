"""
Tests for the ingestion layer: PDFParser and DocumentChunker.
Uses shared fixtures from conftest.py.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document
from src.ingestion.chunker import DocumentChunker


# ── DocumentChunker tests ─────────────────────────────────────────────────────

class TestDocumentChunker:

    def test_chunk_returns_list(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk(inspection_documents)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunks_have_required_metadata(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk(inspection_documents)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "source" in chunk.metadata
            assert "doc_type" in chunk.metadata

    def test_chunk_index_is_zero_based(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk([inspection_documents[0]])
        chunk_indices = [c.metadata["chunk_index"] for c in chunks]
        assert 0 in chunk_indices

    def test_chunk_id_format(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk([inspection_documents[0]])
        for chunk in chunks:
            cid = chunk.metadata["chunk_id"]
            assert "_p" in cid
            assert "_c" in cid

    def test_empty_input_returns_empty_list(self, mock_config):
        chunker = DocumentChunker(mock_config)
        result = chunker.chunk([])
        assert result == []

    def test_metadata_preserved_from_parent(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk([inspection_documents[0]])
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "inspection"
            assert chunk.metadata["section_hint"] == "bathroom"

    def test_chunk_size_respected(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk(inspection_documents)
        for chunk in chunks:
            # Allow up to 2x chunk_size to account for RecursiveCharacterTextSplitter
            # which tries to split on separators and may slightly exceed the target
            assert len(chunk.page_content) <= mock_config.chunk_size * 2

    def test_multiple_documents_produce_multiple_chunks(self, mock_config, all_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk(all_documents)
        # More documents → more chunks
        chunks_single = chunker.chunk([all_documents[0]])
        assert len(chunks) > len(chunks_single)

    def test_chunk_content_not_empty(self, mock_config, inspection_documents):
        chunker = DocumentChunker(mock_config)
        chunks = chunker.chunk(inspection_documents)
        for chunk in chunks:
            assert chunk.page_content.strip() != ""


# ── PDFParser section detection (no real PDF required) ────────────────────────

class TestSectionDetection:
    """Unit tests for the keyword-based section detector."""

    def test_detect_bathroom(self):
        from src.ingestion.pdf_parser import _detect_section
        assert _detect_section("bathroom seepage dampness nahani area") == "bathroom"

    def test_detect_terrace(self):
        from src.ingestion.pdf_parser import _detect_section
        assert _detect_section("terrace roof parapet wall IPS screed") == "terrace"

    def test_detect_external_wall(self):
        from src.ingestion.pdf_parser import _detect_section
        assert _detect_section("external wall crack hairline parapet chajja") == "external_wall"

    def test_detect_thermal(self):
        from src.ingestion.pdf_parser import _detect_section
        assert _detect_section("thermograph temperature reading 45\u00b0C IR bosch") == "thermal"

    def test_detect_structural(self):
        from src.ingestion.pdf_parser import _detect_section
        assert _detect_section("structural beam column reinforcement spalling concrete") == "structural"

    def test_detect_general_fallback(self):
        from src.ingestion.pdf_parser import _detect_section
        result = _detect_section("some random text with no keywords")
        assert result == "general"

    def test_detect_balcony(self):
        from src.ingestion.pdf_parser import _detect_section
        assert _detect_section("open balcony tile joints gap moisture") == "balcony"

    def test_mixed_keywords_picks_highest_score(self):
        from src.ingestion.pdf_parser import _detect_section
        # "bathroom" appears 3 times, "terrace" once → bathroom wins
        text = "bathroom bathroom bathroom terrace"
        assert _detect_section(text) == "bathroom"
