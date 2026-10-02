"""Sparse lexical index, wrapping `rank_bm25.BM25Okapi`."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag_eval_lab.core.models import Chunk, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercases and extracts word tokens; shared by documents and queries."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """BM25 search over `Chunk` text. Does not require embeddings.

    `build()` constructs or replaces the index from `rank_bm25.BM25Okapi`,
    which stays internal to this class.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized_corpus = [_tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b) if tokenized_corpus else None

    def search(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks or self._bm25 is None:
            return []
        top_k = min(top_k, len(self._chunks))
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_positions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[position], score=float(scores[position]), rank=rank)
            for rank, position in enumerate(ranked_positions, start=1)
        ]
