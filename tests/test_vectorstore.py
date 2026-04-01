"""
Tests for the VectorStore (ChromaDB wrapper).
Uses an in-memory / tmp ChromaDB instance so no persistent state is affected.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Minimal Config mock pointing to a temp ChromaDB path."""
    cfg = MagicMock()
    cfg.embedding_model = "BAAI/bge-small-en-v1.5"
    cfg.chroma_db_path = tmp_path / "chroma_db"
    cfg.chroma_db_path.mkdir()
    cfg.chroma_collection = "test_collection"
    cfg.retrieval_k = 3
    cfg.output_dir = tmp_path / "outputs"
    cfg.images_dir = tmp_path / "images"
    return cfg


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Bathroom shows seepage on the floor junction.",
            metadata={
                "source": "test.pdf",
                "doc_type": "inspection",
                "page": 1,
                "chunk_index": 0,
                "chunk_id": "test.pdf_p1_c0",
                "section_hint": "bathroom",
            },
        ),
        Document(
            page_content="Terrace has cracks in the IPS screed layer.",
            metadata={
                "source": "test.pdf",
                "doc_type": "inspection",
                "page": 2,
                "chunk_index": 0,
                "chunk_id": "test.pdf_p2_c0",
                "section_hint": "terrace",
            },
        ),
        Document(
            page_content="Thermal imaging shows high moisture in balcony slab.",
            metadata={
                "source": "thermal.pdf",
                "doc_type": "thermal",
                "page": 1,
                "chunk_index": 0,
                "chunk_id": "thermal.pdf_p1_c0",
                "section_hint": "balcony",
            },
        ),
    ]


# ── VectorStore tests (with mocked embeddings for speed) ──────────────────────

class TestVectorStore:

    @patch("src.vectorstore.embedder.HuggingFaceEmbeddings")
    def test_is_empty_on_fresh_store(self, mock_emb_cls, config):
        """A newly created store reports as empty."""
        from src.vectorstore.store import VectorStore
        # Use a mock embedding to avoid downloading the actual model
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.0] * 384 for _ in texts]
        mock_emb.embed_query = lambda text: [0.0] * 384
        mock_emb_cls.return_value = mock_emb

        with patch("src.vectorstore.store.get_embeddings", return_value=mock_emb):
            vs = VectorStore(config)
            assert vs.is_empty()

    @patch("src.vectorstore.embedder.HuggingFaceEmbeddings")
    def test_add_and_count(self, mock_emb_cls, config, sample_docs):
        """After adding documents, count should be non-zero."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_query = lambda text: [0.1] * 384
        mock_emb_cls.return_value = mock_emb

        with patch("src.vectorstore.store.get_embeddings", return_value=mock_emb):
            vs = VectorStore(config)
            vs.add_documents(sample_docs)
            assert vs.count() == len(sample_docs)

    @patch("src.vectorstore.embedder.HuggingFaceEmbeddings")
    def test_add_empty_list_is_safe(self, mock_emb_cls, config):
        """Adding an empty list should not raise."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: []
        mock_emb.embed_query = lambda text: [0.0] * 384
        mock_emb_cls.return_value = mock_emb

        with patch("src.vectorstore.store.get_embeddings", return_value=mock_emb):
            vs = VectorStore(config)
            vs.add_documents([])  # should not raise

    @patch("src.vectorstore.embedder.HuggingFaceEmbeddings")
    def test_clear_empties_store(self, mock_emb_cls, config, sample_docs):
        """After clear(), the store should be empty."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_query = lambda text: [0.1] * 384
        mock_emb_cls.return_value = mock_emb

        with patch("src.vectorstore.store.get_embeddings", return_value=mock_emb):
            vs = VectorStore(config)
            vs.add_documents(sample_docs)
            vs.clear()
            assert vs.is_empty()
