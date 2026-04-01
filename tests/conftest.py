"""
tests/conftest.py — Shared pytest fixtures for all test modules.

Centralises common mocks so test_ingestion.py, test_vectorstore.py,
and test_graph.py don't duplicate fixture code.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document


# ── Config mocks ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_config(tmp_path):
    """
    Minimal Config-like mock suitable for unit tests.
    Points all paths to tmp_path so nothing touches the real project dirs.
    """
    cfg = MagicMock()
    # LLM
    cfg.cohere_api_key = "test-cohere-key"
    cfg.cohere_model   = "command-r-plus-08-2024"
    # Embeddings
    cfg.embedding_model = "BAAI/bge-small-en-v1.5"
    # Paths
    cfg.chroma_db_path    = tmp_path / "chroma_db"
    cfg.chroma_collection = "test_collection"
    cfg.images_dir        = tmp_path / "images"
    cfg.output_dir        = tmp_path / "outputs"
    cfg.input_dir         = tmp_path / "inputs"
    # Create dirs
    for d in [cfg.chroma_db_path, cfg.images_dir, cfg.output_dir, cfg.input_dir]:
        d.mkdir(parents=True, exist_ok=True)
    # Chunking
    cfg.chunk_size    = 200
    cfg.chunk_overlap = 30
    # Retrieval
    cfg.retrieval_k = 3
    # Report
    cfg.report_title   = "Test DDR Report"
    cfg.company_name   = "Test Company"
    cfg.company_website = "www.test.com"
    return cfg


# ── Sample documents ──────────────────────────────────────────────────────────

@pytest.fixture
def inspection_documents():
    """Two inspection-PDF page documents covering different areas."""
    return [
        Document(
            page_content=(
                "Bathroom shows seepage and dampness at skirting level. "
                "Tile joints have gaps and blackish deposits. "
                "Nahani trap shows signs of moisture. " * 4
            ),
            metadata={
                "source": "inspection.pdf",
                "doc_type": "inspection",
                "page": 1,
                "total_pages": 5,
                "section_hint": "bathroom",
                "has_images": True,
                "image_count": 2,
                "chunk_id": "inspection.pdf_p1_c0",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "External wall shows hairline cracks at parapet level. "
                "Spalling concrete observed on south-facing facade. "
                "Paint is flaking and algae growth is visible. " * 4
            ),
            metadata={
                "source": "inspection.pdf",
                "doc_type": "inspection",
                "page": 2,
                "total_pages": 5,
                "section_hint": "external_wall",
                "has_images": True,
                "image_count": 3,
                "chunk_id": "inspection.pdf_p2_c0",
                "chunk_index": 0,
            },
        ),
    ]


@pytest.fixture
def thermal_documents():
    """Two thermal-PDF page documents."""
    return [
        Document(
            page_content=(
                "Thermal imaging shows temperature differential at bathroom ceiling. "
                "Bosch IR thermograph: 24.0°C hotspot vs 22.3°C ambient. " * 3
            ),
            metadata={
                "source": "thermal.pdf",
                "doc_type": "thermal",
                "page": 3,
                "total_pages": 10,
                "section_hint": "bathroom",
                "has_images": True,
                "image_count": 2,
                "chunk_id": "thermal.pdf_p3_c0",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "Terrace slab shows high moisture readings. "
                "IR thermograph indicates ponding zone near drain outlet. " * 3
            ),
            metadata={
                "source": "thermal.pdf",
                "doc_type": "thermal",
                "page": 7,
                "total_pages": 10,
                "section_hint": "terrace",
                "has_images": True,
                "image_count": 2,
                "chunk_id": "thermal.pdf_p7_c0",
                "chunk_index": 0,
            },
        ),
    ]


@pytest.fixture
def all_documents(inspection_documents, thermal_documents):
    return inspection_documents + thermal_documents


# ── LLM mocks ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    """Factory: returns a mock LLM that always responds with *text*."""
    def _factory(text: str = "Test LLM output for this section."):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = text
        mock_llm.invoke.return_value = mock_response
        return mock_llm
    return _factory


# ── Vector store mock ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    """
    Mock VectorStore whose retrieve() always returns one inspection document
    about bathroom seepage — enough to satisfy all node prompts.
    """
    mock_vs = MagicMock()
    mock_vs.retrieve.return_value = [
        Document(
            page_content=(
                "Bathroom tile joints show gaps. Dampness at skirting level. "
                "External wall cracks observed. Terrace screed hollow."
            ),
            metadata={"source": "test.pdf", "doc_type": "inspection", "page": 1},
        )
    ]
    return mock_vs


# ── Image map ─────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_image_map():
    return {"inspection": [], "thermal": []}


@pytest.fixture
def sample_image_map(tmp_path):
    """Creates two tiny real PNG files so image-path assertions work."""
    from PIL import Image as PILImage
    import numpy as np

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_paths = {}
    for doc_type, hint in [("inspection", "bathroom"), ("thermal", "bathroom")]:
        p = images_dir / f"{doc_type}_xref00001.png"
        # 300×300 solid-colour PNG
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[:, :] = [200, 200, 200]
        PILImage.fromarray(arr).save(str(p))
        img_paths[doc_type] = [{"path": p, "section_hint": hint, "page": 1}]

    return img_paths
