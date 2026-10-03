"""Embedding strategies: turns text into dense vector representations."""

from rag_eval_lab.embedding.cache import compute_cache_key, load_cached_embeddings, save_cached_embeddings
from rag_eval_lab.embedding.registry import embedding_registry
from rag_eval_lab.embedding.sentence_transformers import SentenceTransformerEmbedding

__all__ = [
    "SentenceTransformerEmbedding",
    "compute_cache_key",
    "embedding_registry",
    "load_cached_embeddings",
    "save_cached_embeddings",
]
