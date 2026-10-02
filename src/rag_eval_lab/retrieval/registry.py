"""Registry of retrieval strategies, keyed by strategy name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import RetrievalStrategy
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.retrieval.bm25 import BM25Retriever
from rag_eval_lab.retrieval.dense import DenseRetriever
from rag_eval_lab.retrieval.hybrid import HybridRetriever

retrieval_registry: Registry[RetrievalStrategy] = Registry(kind="retrieval")

retrieval_registry.register("dense")(DenseRetriever)
retrieval_registry.register("bm25")(BM25Retriever)
retrieval_registry.register("hybrid")(HybridRetriever)
