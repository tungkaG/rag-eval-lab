"""Dense vector index, wrapping `faiss.IndexFlatIP` for exact cosine similarity search."""

from __future__ import annotations

import faiss
import numpy as np

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.core.models import Chunk, RetrievedChunk


def _normalized_float32(vectors: list[list[float]]) -> np.ndarray:
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(array)
    return array


class FaissIndex:
    """Exact nearest neighbor search over L2-normalized vectors (cosine similarity).

    Stores vectors and their originating `Chunk`s in parallel; `faiss` stays
    internal to this class. The vector dimension is inferred from the first
    `add()` call; subsequent calls append to the same index.
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValidationError(
                f"FaissIndex.add received {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        vectors = _normalized_float32(embeddings)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        top_k = min(top_k, len(self._chunks))
        query = _normalized_float32([query_embedding])
        scores, positions = self._index.search(query, top_k)
        results = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position < 0:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[position], score=float(score), rank=rank))
        return results
