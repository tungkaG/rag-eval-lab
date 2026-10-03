"""No-op reranker: a baseline that returns candidates unchanged, truncated to `top_k`."""

from __future__ import annotations

from rag_eval_lab.core.models import Query, RetrievedChunk


class NoReranker:
    """Returns the first `top_k` results as-is: same order, scores, and objects."""

    def rerank(
        self,
        query: Query,
        retrieved_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return retrieved_chunks[:top_k]
