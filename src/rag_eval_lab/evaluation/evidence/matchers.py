"""Built-in evidence matchers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from rag_eval_lab.embedding.registry import embedding_registry

_WORD_PATTERN = re.compile(r"\w+")


def _token_f1(left: str, right: str) -> float:
    left_tokens = Counter(_WORD_PATTERN.findall(left.lower()))
    right_tokens = Counter(_WORD_PATTERN.findall(right.lower()))
    overlap = sum((left_tokens & right_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(left_tokens.values())
    recall = overlap / sum(right_tokens.values())
    return 2 * precision * recall / (precision + recall)


class LexicalEvidenceMatcher:
    """Matches text with lowercase word-token F1."""

    name = "lexical"
    metadata: dict[str, Any] = {"matcher": name}

    def score(self, retrieved_texts: list[str], evidence_texts: list[str]) -> list[list[float]]:
        return [[_token_f1(chunk_text, evidence_text) for evidence_text in evidence_texts] for chunk_text in retrieved_texts]


class EmbeddingEvidenceMatcher:
    """Matches text with cosine similarity from a configured embedding strategy."""

    name = "embedding"

    def __init__(self, embedding_strategy: str, embedding_params: dict[str, Any] | None = None) -> None:
        self.embedding_strategy = embedding_strategy
        self.embedding_params = dict(embedding_params or {})
        self._embedder = embedding_registry.create(embedding_strategy, **self.embedding_params)
        self.metadata = {
            "matcher": self.name,
            "embedding_strategy": self.embedding_strategy,
            "embedding_params": self.embedding_params,
        }

    def score(self, retrieved_texts: list[str], evidence_texts: list[str]) -> list[list[float]]:
        if not retrieved_texts or not evidence_texts:
            return [[0.0] * len(evidence_texts) for _ in retrieved_texts]

        retrieved_vectors = self._embedder.embed_documents(retrieved_texts)
        evidence_vectors = self._embedder.embed_documents(evidence_texts)
        return [
            [self._cosine_similarity(chunk_vector, evidence_vector) for evidence_vector in evidence_vectors]
            for chunk_vector in retrieved_vectors
        ]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have the same dimension for evidence matching")
        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_product / (left_norm * right_norm)