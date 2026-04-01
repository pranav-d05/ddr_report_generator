"""
Document Chunker — splits LangChain Documents into overlapping text chunks
with enriched metadata.

Uses LangChain's RecursiveCharacterTextSplitter so that chunks respect
paragraph / sentence boundaries before falling back to character splits.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Split a list of Documents into smaller, overlapping chunks."""

    def __init__(self, config: Config):
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            # Separators in priority order
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split *documents* into chunks.

        Each chunk inherits the parent document's metadata and gains two
        extra fields:
          - ``chunk_index``  : zero-based index within the parent page
          - ``chunk_id``     : unique string  "<source>_p<page>_c<chunk_index>"
        """
        if not documents:
            logger.warning("DocumentChunker.chunk() received an empty document list")
            return []

        all_chunks: List[Document] = []

        for doc in documents:
            splits = self._splitter.split_documents([doc])
            for idx, chunk in enumerate(splits):
                # Enrich metadata
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["chunk_id"] = (
                    f"{chunk.metadata.get('source', 'unknown')}"
                    f"_p{chunk.metadata.get('page', 0)}"
                    f"_c{idx}"
                )
                all_chunks.append(chunk)

        logger.info(
            f"Chunked {len(documents)} documents → {len(all_chunks)} chunks "
            f"(size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )
        return all_chunks
