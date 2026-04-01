"""
Tests for the VectorStore (ChromaDB wrapper).
Uses shared fixtures from conftest.py and an in-memory / tmp ChromaDB instance.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestVectorStore:

    @patch("src.vectorstore.store.get_embeddings")
    def test_is_empty_on_fresh_store(self, mock_get_emb, mock_config):
        """A newly created store reports as empty."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.0] * 384 for _ in texts]
        mock_emb.embed_query     = lambda text:  [0.0] * 384
        mock_get_emb.return_value = mock_emb

        from src.vectorstore.store import VectorStore
        vs = VectorStore(mock_config)
        assert vs.is_empty()

    @patch("src.vectorstore.store.get_embeddings")
    def test_add_and_count(self, mock_get_emb, mock_config, all_documents):
        """After adding documents, count should equal the number of docs added."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_query     = lambda text:  [0.1] * 384
        mock_get_emb.return_value = mock_emb

        from src.vectorstore.store import VectorStore
        vs = VectorStore(mock_config)
        vs.add_documents(all_documents)
        assert vs.count() == len(all_documents)

    @patch("src.vectorstore.store.get_embeddings")
    def test_add_empty_list_is_safe(self, mock_get_emb, mock_config):
        """Adding an empty list should not raise any exception."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: []
        mock_emb.embed_query     = lambda text:  [0.0] * 384
        mock_get_emb.return_value = mock_emb

        from src.vectorstore.store import VectorStore
        vs = VectorStore(mock_config)
        vs.add_documents([])   # must not raise

    @patch("src.vectorstore.store.get_embeddings")
    def test_clear_empties_store(self, mock_get_emb, mock_config, all_documents):
        """After clear(), is_empty() must return True."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: [[0.1] * 384 for _ in texts]
        mock_emb.embed_query     = lambda text:  [0.1] * 384
        mock_get_emb.return_value = mock_emb

        from src.vectorstore.store import VectorStore
        vs = VectorStore(mock_config)
        vs.add_documents(all_documents)
        assert not vs.is_empty()
        vs.clear()
        assert vs.is_empty()

    @patch("src.vectorstore.store.get_embeddings")
    def test_count_returns_int(self, mock_get_emb, mock_config):
        """count() must always return an int, even on an empty store."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: []
        mock_emb.embed_query     = lambda text:  [0.0] * 384
        mock_get_emb.return_value = mock_emb

        from src.vectorstore.store import VectorStore
        vs = VectorStore(mock_config)
        assert isinstance(vs.count(), int)

    @patch("src.vectorstore.store.get_embeddings")
    def test_double_clear_is_safe(self, mock_get_emb, mock_config):
        """Calling clear() twice must not raise."""
        mock_emb = MagicMock()
        mock_emb.embed_documents = lambda texts: []
        mock_emb.embed_query     = lambda text:  [0.0] * 384
        mock_get_emb.return_value = mock_emb

        from src.vectorstore.store import VectorStore
        vs = VectorStore(mock_config)
        vs.clear()
        vs.clear()   # second clear on already-empty store — must not raise
