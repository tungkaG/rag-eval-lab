"""Dense retrieval: embeds the query and searches an already-built dense `Index`."""

from __future__ import annotations

from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.core.protocols import EmbeddingStrategy
from rag_eval_lab.indexing.base import Index


class DenseRetriever:
    """Retrieves chunks by embedding the query and searching a pre-built dense `Index`.

    Both the embedding strategy and the index are injected already
    constructed; this retriever never loads a model or builds an index.
    """

    def __init__(self, embedder: EmbeddingStrategy, index: Index) -> None:
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: Query, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_queries([query.text])[0]
        return self.index.search(query_embedding, top_k)
