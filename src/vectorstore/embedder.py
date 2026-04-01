"""
Embedding model loader — wraps HuggingFace sentence-transformers.

Model: BAAI/bge-small-en-v1.5  (local, no API cost, ~130 MB)
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.logger import get_logger

logger = get_logger(__name__)

_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    """
    Return a (cached) HuggingFaceEmbeddings instance for *model_name*.

    The first call downloads the model weights to the HuggingFace cache;
    subsequent calls return the cached object without reloading.
    """
    if model_name in _embeddings_cache:
        return _embeddings_cache[model_name]

    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _embeddings_cache[model_name] = embeddings
    logger.info(f"Embedding model ready: {model_name}")
    return embeddings
