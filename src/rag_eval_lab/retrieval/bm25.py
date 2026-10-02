"""BM25 retrieval: searches an already-built sparse index with the raw query text."""

from __future__ import annotations

from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.indexing.base import SparseIndex


class BM25Retriever:
    """Retrieves chunks from a pre-built `SparseIndex`.

    Does not tokenize queries itself; tokenization stays inside the index.
    """

    def __init__(self, index: SparseIndex) -> None:
        self.index = index

    def retrieve(self, query: Query, top_k: int) -> list[RetrievedChunk]:
        return self.index.search(query.text, top_k)
