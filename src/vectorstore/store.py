"""
ChromaDB Vector Store wrapper.

Provides a clean interface for:
  - Adding documents (embed + upsert)
  - Similarity retrieval with optional metadata filters
  - Emptiness check and collection clearing
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import Config
from src.utils.logger import get_logger
from src.vectorstore.embedder import get_embeddings

logger = get_logger(__name__)


class VectorStore:
    """Persistent ChromaDB vector store backed by HuggingFace embeddings."""

    def __init__(self, config: Config):
        self.config = config
        self._embeddings = get_embeddings(config.embedding_model)

        self._db = Chroma(
            collection_name=config.chroma_collection,
            embedding_function=self._embeddings,
            persist_directory=str(config.chroma_db_path),
        )
        logger.info(
            f"VectorStore ready — collection='{config.chroma_collection}' "
            f"path='{config.chroma_db_path}'"
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Embed *documents* and upsert them into ChromaDB."""
        if not documents:
            logger.warning("add_documents() received empty list — nothing to add")
            return

        ids = [
            doc.metadata.get("chunk_id", f"doc_{i}")
            for i, doc in enumerate(documents)
        ]

        self._db.add_documents(documents=documents, ids=ids)
        logger.info(f"Upserted {len(documents)} chunks into ChromaDB")

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve the top-k most similar documents for *query*.

        Args:
            query:  Natural-language query string.
            k:      Number of results (defaults to config.retrieval_k).
            filter: Optional ChromaDB metadata filter dict,
                    e.g. {"doc_type": "thermal"}.
        """
        k = k or self.config.retrieval_k
        kwargs: Dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter

        try:
            results = self._db.similarity_search(query, **kwargs)
            logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:60]}'")
            return results
        except Exception as exc:
            logger.warning(f"retrieve() failed for query '{query[:40]}': {exc}")
            return []

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """Like retrieve() but also returns similarity scores."""
        k = k or self.config.retrieval_k
        kwargs: Dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self._db.similarity_search_with_score(query, **kwargs)

    def is_empty(self) -> bool:
        """Return True if the collection has no documents."""
        return self.count() == 0

    def clear(self) -> None:
        """
        Delete all documents from the collection.
        Recreates the collection from scratch to avoid filter-on-empty errors.
        """
        try:
            count = self.count()
            if count == 0:
                logger.info("clear(): collection already empty")
                return

            # Delete the underlying collection and recreate it
            self._db._client.delete_collection(self.config.chroma_collection)
            self._db = Chroma(
                collection_name=self.config.chroma_collection,
                embedding_function=self._embeddings,
                persist_directory=str(self.config.chroma_db_path),
            )
            logger.info(f"ChromaDB collection cleared (had {count} documents)")
        except Exception as exc:
            logger.warning(f"clear() encountered an issue (ignored): {exc}")

    def count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            return self._db._collection.count()
        except Exception:
            return 0
