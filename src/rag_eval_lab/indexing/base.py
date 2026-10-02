"""Indexing interfaces: the dense `Index` protocol from core, plus the sparse `SparseIndex` protocol.

`SparseIndex` is defined here (not in core) because lexical indexes like
BM25 do not take embeddings and must not pretend to implement `Index`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_eval_lab.core.models import Chunk, RetrievedChunk
from rag_eval_lab.core.protocols import Index

__all__ = ["Index", "SparseIndex"]


@runtime_checkable
class SparseIndex(Protocol):
    """A lexical index (e.g. BM25) that scores chunks directly from query text."""

    def build(self, chunks: list[Chunk]) -> None: ...

    def search(self, query_text: str, top_k: int) -> list[RetrievedChunk]: ...
