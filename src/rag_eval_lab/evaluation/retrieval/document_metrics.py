"""Document-level retrieval metrics.

Chunks map to parent documents via `chunk.document_id`. These metrics
evaluate a *document* ranking, not a chunk ranking: retrieved chunks are
followed in rank order, deduplicated by parent document while preserving
each document's first occurrence, then truncated to the first `k` unique
documents. A document is "relevant" when its ground truth relevance grade
is greater than zero; zero and absent judgments contribute no credit.
"""

from __future__ import annotations

import math

from rag_eval_lab.core.models import EvaluationSample, MetricResult, RetrievedChunk


def _ranked_document_ids(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """Follows rank order, keeping only each parent document's first occurrence."""
    ordered = sorted(retrieved_chunks, key=lambda result: result.rank)
    seen: set[str] = set()
    document_ids: list[str] = []
    for result in ordered:
        document_id = result.chunk.document_id
        if document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return document_ids


def _relevant_document_ids(document_relevance: dict[str, int]) -> set[str]:
    return {document_id for document_id, grade in document_relevance.items() if grade > 0}


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


class DocumentHitRateAtK:
    """1 if any of the top-`k` ranked parent documents is relevant, else 0."""

    required_capabilities = {"document_relevance"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        relevant_ids = _relevant_document_ids(sample.evaluation_case.ground_truth.document_relevance)
        top_k_ids = _ranked_document_ids(sample.retrieved_chunks)[: self.k]
        hit = 1.0 if any(document_id in relevant_ids for document_id in top_k_ids) else 0.0
        return MetricResult(name="document_hit_rate", value=hit, k=self.k)


class DocumentRecallAtK:
    """Relevant documents retrieved within the top `k`, divided by all positively judged documents."""

    required_capabilities = {"document_relevance"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        relevant_ids = _relevant_document_ids(sample.evaluation_case.ground_truth.document_relevance)
        if not relevant_ids:
            return MetricResult(name="document_recall", value=0.0, k=self.k)
        top_k_ids = _ranked_document_ids(sample.retrieved_chunks)[: self.k]
        retrieved_relevant = sum(1 for document_id in top_k_ids if document_id in relevant_ids)
        return MetricResult(name="document_recall", value=retrieved_relevant / len(relevant_ids), k=self.k)


class DocumentPrecisionAtK:
    """Relevant documents retrieved within the top `k`, divided by `k`."""

    required_capabilities = {"document_relevance"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        if self.k <= 0:
            return MetricResult(name="document_precision", value=0.0, k=self.k)
        relevant_ids = _relevant_document_ids(sample.evaluation_case.ground_truth.document_relevance)
        top_k_ids = _ranked_document_ids(sample.retrieved_chunks)[: self.k]
        retrieved_relevant = sum(1 for document_id in top_k_ids if document_id in relevant_ids)
        return MetricResult(name="document_precision", value=retrieved_relevant / self.k, k=self.k)


class DocumentMRRAtK:
    """Reciprocal rank of the first relevant document within the top `k`, else 0."""

    required_capabilities = {"document_relevance"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        relevant_ids = _relevant_document_ids(sample.evaluation_case.ground_truth.document_relevance)
        top_k_ids = _ranked_document_ids(sample.retrieved_chunks)[: self.k]
        for rank, document_id in enumerate(top_k_ids, start=1):
            if document_id in relevant_ids:
                return MetricResult(name="document_mrr", value=1.0 / rank, k=self.k)
        return MetricResult(name="document_mrr", value=0.0, k=self.k)


class DocumentNDCGAtK:
    """Graded NDCG over the top-`k` document ranking.

    Gain is `2**grade - 1` and discount is `log2(rank + 1)`; ideal DCG comes
    from all ground truth grades sorted descending, limited to `k`.
    """

    required_capabilities = {"document_relevance"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        document_relevance = sample.evaluation_case.ground_truth.document_relevance
        top_k_ids = _ranked_document_ids(sample.retrieved_chunks)[: self.k]
        actual_dcg = _dcg([document_relevance.get(document_id, 0) for document_id in top_k_ids])

        ideal_grades = sorted(document_relevance.values(), reverse=True)[: self.k]
        ideal_dcg = _dcg(ideal_grades)

        if ideal_dcg == 0:
            return MetricResult(name="document_ndcg", value=0.0, k=self.k)
        return MetricResult(name="document_ndcg", value=actual_dcg / ideal_dcg, k=self.k)
