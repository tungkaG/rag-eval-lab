"""Indexing: dense (FAISS) and sparse (BM25) chunk indexes."""

from rag_eval_lab.indexing.base import Index, SparseIndex
from rag_eval_lab.indexing.bm25_index import BM25Index
from rag_eval_lab.indexing.faiss_index import FaissIndex
from rag_eval_lab.indexing.registry import dense_index_registry, sparse_index_registry

__all__ = [
    "BM25Index",
    "FaissIndex",
    "Index",
    "SparseIndex",
    "dense_index_registry",
    "sparse_index_registry",
]
