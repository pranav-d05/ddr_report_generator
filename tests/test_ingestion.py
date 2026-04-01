"""
Tests for the ingestion layer: PDFParser and DocumentChunker.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion.chunker import DocumentChunker
from src.config import Config
from langchain_core.documents import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    cfg = MagicMock(spec=Config)
    cfg.chunk_size = 200
    cfg.chunk_overlap = 30
    cfg.images_dir = tmp_path / "images"
    cfg.images_dir.mkdir()
    return cfg


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="This is a test document about bathroom seepage and dampness. "
                         "The walls show efflorescence near the floor junction. "
                         "Thermal imaging confirms moisture ingress." * 3,
            metadata={
                "source": "inspection.pdf",
                "doc_type": "inspection",
                "page": 1,
                "total_pages": 5,
                "section_hint": "bathroom",
                "has_images": False,
                "image_count": 0,
            }
        ),
        Document(
            page_content="External wall shows cracks at parapet level. "
                         "Spalling concrete observed on south-facing facade." * 3,
            metadata={
                "source": "inspection.pdf",
                "doc_type": "inspection",
                "page": 2,
                "total_pages": 5,
                "section_hint": "external_wall",
                "has_images": True,
                "image_count": 2,
            }
        ),
    ]


# ── DocumentChunker tests ─────────────────────────────────────────────────────

class TestDocumentChunker:

    def test_chunk_returns_list(self, config, sample_documents):
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(sample_documents)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunks_have_required_metadata(self, config, sample_documents):
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(sample_documents)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "source" in chunk.metadata
            assert "doc_type" in chunk.metadata

    def test_chunk_index_is_zero_based(self, config, sample_documents):
        chunker = DocumentChunker(config)
        # Use a single document to check indices
        chunks = chunker.chunk([sample_documents[0]])
        chunk_indices = [c.metadata["chunk_index"] for c in chunks]
        assert 0 in chunk_indices

    def test_chunk_id_format(self, config, sample_documents):
        chunker = DocumentChunker(config)
        chunks = chunker.chunk([sample_documents[0]])
        for chunk in chunks:
            cid = chunk.metadata["chunk_id"]
            assert "_p" in cid
            assert "_c" in cid

    def test_empty_input_returns_empty_list(self, config):
        chunker = DocumentChunker(config)
        result = chunker.chunk([])
        assert result == []

    def test_metadata_preserved_from_parent(self, config, sample_documents):
        chunker = DocumentChunker(config)
        chunks = chunker.chunk([sample_documents[0]])
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "inspection"
            assert chunk.metadata["section_hint"] == "bathroom"

    def test_chunk_size_respected(self, config, sample_documents):
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(sample_documents)
        # Each chunk should not wildly exceed chunk_size
        for chunk in chunks:
            assert len(chunk.page_content) <= config.chunk_size * 2


# ── PDFParser (integration-style with mocks) ──────────────────────────────────

class TestPDFParserSectionDetection:
    """Test the _detect_section helper without needing actual PDFs."""

    def test_detect_bathroom(self):
        from src.ingestion.pdf_parser import _detect_section
        result = _detect_section("bathroom seepage dampness nahani area")
        assert result == "bathroom"

    def test_detect_terrace(self):
        from src.ingestion.pdf_parser import _detect_section
        result = _detect_section("terrace roof parapet wall IPS screed")
        assert result == "terrace"

    def test_detect_thermal(self):
        from src.ingestion.pdf_parser import _detect_section
        result = _detect_section("thermograph temperature reading 45°C IR bosch")
        assert result == "thermal"

    def test_detect_general_fallback(self):
        from src.ingestion.pdf_parser import _detect_section
        result = _detect_section("some random text with no keywords")
        assert result == "general"
