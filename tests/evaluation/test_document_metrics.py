import math

import pytest

from rag_eval_lab.core.models import Chunk, EvaluationCase, EvaluationSample, GroundTruth, RetrievedChunk
from rag_eval_lab.evaluation.retrieval.document_metrics import (
    DocumentHitRateAtK,
    DocumentMRRAtK,
    DocumentNDCGAtK,
    DocumentPrecisionAtK,
    DocumentRecallAtK,
)


def _sample(document_relevance: dict[str, int], ranked_document_ids: list[str]) -> EvaluationSample:
    retrieved_chunks = [
        RetrievedChunk(
            chunk=Chunk(id=f"{document_id}_chunk_{rank}", document_id=document_id, text="text"),
            score=1.0 / rank,
            rank=rank,
        )
        for rank, document_id in enumerate(ranked_document_ids, start=1)
    ]
    case = EvaluationCase(
        query_id="q1",
        question="q?",
        ground_truth=GroundTruth(document_relevance=document_relevance),
    )
    return EvaluationSample(evaluation_case=case, retrieved_chunks=retrieved_chunks)


def test_hit_rate_at_k():
    sample = _sample({"d1": 1}, ["d2", "d1", "d3"])

    assert DocumentHitRateAtK(k=1).evaluate(sample).value == 0.0
    assert DocumentHitRateAtK(k=2).evaluate(sample).value == 1.0


def test_recall_at_k():
    sample = _sample({"d1": 1, "d2": 1, "d3": 0}, ["d2", "d1", "d4"])

    assert DocumentRecallAtK(k=1).evaluate(sample).value == pytest.approx(0.5)
    result = DocumentRecallAtK(k=2).evaluate(sample)
    assert result.value == pytest.approx(1.0)
    assert result.k == 2


def test_recall_returns_zero_when_no_positively_judged_documents():
    sample = _sample({"d1": 0}, ["d1"])

    assert DocumentRecallAtK(k=5).evaluate(sample).value == 0.0


def test_precision_at_k():
    sample = _sample({"d1": 1, "d2": 1}, ["d1", "d3", "d4"])

    result = DocumentPrecisionAtK(k=3).evaluate(sample)
    assert result.value == pytest.approx(1 / 3)


def test_mrr_at_k():
    sample = _sample({"d3": 1}, ["d1", "d2", "d3", "d4"])

    assert DocumentMRRAtK(k=2).evaluate(sample).value == 0.0
    assert DocumentMRRAtK(k=3).evaluate(sample).value == pytest.approx(1 / 3)


def test_ndcg_at_k_graded():
    document_relevance = {"d1": 2, "d2": 0, "d3": 1}
    # d1 appears twice (ranks 2 and 3); the second occurrence is deduplicated,
    # leaving the document ranking d4, d1, d5, d3.
    sample = _sample(document_relevance, ["d4", "d1", "d1", "d5", "d3"])

    result = DocumentNDCGAtK(k=5).evaluate(sample)

    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))

    actual = dcg([0, 2, 0, 1])
    ideal = dcg(sorted(document_relevance.values(), reverse=True)[:5])

    assert result.value == pytest.approx(actual / ideal)
    assert result.k == 5


def test_ndcg_returns_zero_when_ideal_dcg_is_zero():
    sample = _sample({"d1": 0}, ["d1"])

    assert DocumentNDCGAtK(k=5).evaluate(sample).value == 0.0


def test_duplicate_parent_documents_are_deduplicated_preserving_first_occurrence():
    sample = _sample({"d1": 1}, ["d2", "d1", "d1", "d1"])

    assert DocumentHitRateAtK(k=2).evaluate(sample).value == 1.0
    assert DocumentMRRAtK(k=2).evaluate(sample).value == pytest.approx(0.5)


def test_relevant_document_beyond_k_is_not_counted():
    sample = _sample({"d5": 1}, ["d1", "d2", "d3", "d4", "d5"])

    assert DocumentHitRateAtK(k=4).evaluate(sample).value == 0.0
    assert DocumentHitRateAtK(k=5).evaluate(sample).value == 1.0


def test_empty_retrieved_chunks_returns_zero_for_all_metrics():
    sample = _sample({"d1": 1}, [])

    assert DocumentHitRateAtK(k=5).evaluate(sample).value == 0.0
    assert DocumentRecallAtK(k=5).evaluate(sample).value == 0.0
    assert DocumentPrecisionAtK(k=5).evaluate(sample).value == 0.0
    assert DocumentMRRAtK(k=5).evaluate(sample).value == 0.0
    assert DocumentNDCGAtK(k=5).evaluate(sample).value == 0.0


def test_required_capabilities_is_document_relevance():
    for metric_cls in (
        DocumentHitRateAtK,
        DocumentRecallAtK,
        DocumentPrecisionAtK,
        DocumentMRRAtK,
        DocumentNDCGAtK,
    ):
        assert metric_cls(k=1).required_capabilities == {"document_relevance"}
