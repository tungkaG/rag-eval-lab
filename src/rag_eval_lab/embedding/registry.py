"""Registry of embedding strategies, keyed by strategy name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import EmbeddingStrategy
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.embedding.sentence_transformers import SentenceTransformerEmbedding

embedding_registry: Registry[EmbeddingStrategy] = Registry(kind="embedding")

embedding_registry.register("sentence_transformer")(SentenceTransformerEmbedding)
