"""Embedding strategies: turns text into dense vector representations."""

from rag_eval_lab.embedding.registry import embedding_registry
from rag_eval_lab.embedding.sentence_transformers import SentenceTransformerEmbedding

__all__ = ["SentenceTransformerEmbedding", "embedding_registry"]
