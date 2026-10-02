"""Registries of dense and sparse index implementations, keyed by name."""

from __future__ import annotations

from rag_eval_lab.core.registry import Registry
from rag_eval_lab.indexing.base import Index, SparseIndex
from rag_eval_lab.indexing.bm25_index import BM25Index
from rag_eval_lab.indexing.faiss_index import FaissIndex

dense_index_registry: Registry[Index] = Registry(kind="dense_index")
sparse_index_registry: Registry[SparseIndex] = Registry(kind="sparse_index")

dense_index_registry.register("faiss")(FaissIndex)
sparse_index_registry.register("bm25")(BM25Index)
