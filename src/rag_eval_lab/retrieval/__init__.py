"""Retrieval strategies: dense, BM25, and RRF-fused hybrid retrieval."""

from rag_eval_lab.retrieval.bm25 import BM25Retriever
from rag_eval_lab.retrieval.builders import PreparedRetrieval, RetrievalBuilder, retrieval_builder_registry
from rag_eval_lab.retrieval.dense import DenseRetriever
from rag_eval_lab.retrieval.hybrid import HybridRetriever
from rag_eval_lab.retrieval.registry import retrieval_registry
from rag_eval_lab.retrieval.rrf import reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "PreparedRetrieval",
    "RetrievalBuilder",
    "reciprocal_rank_fusion",
    "retrieval_builder_registry",
    "retrieval_registry",
]
