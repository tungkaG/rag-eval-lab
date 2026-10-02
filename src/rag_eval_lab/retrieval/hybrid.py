"""Hybrid retrieval: fuses dense and sparse candidate lists using Reciprocal Rank Fusion."""

from __future__ import annotations

from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.core.protocols import RetrievalStrategy
from rag_eval_lab.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Combines a dense and a sparse `RetrievalStrategy` using RRF.

    `dense_top_k` / `sparse_top_k` control how many candidates are fetched
    from each side independently of `retrieve`'s `top_k`, which only limits
    the final fused output.
    """

    def __init__(
        self,
        dense_retriever: RetrievalStrategy,
        sparse_retriever: RetrievalStrategy,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: Query, top_k: int) -> list[RetrievedChunk]:
        dense_results = self.dense_retriever.retrieve(query, self.dense_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, self.sparse_top_k)
        return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k, rrf_k=self.rrf_k)
