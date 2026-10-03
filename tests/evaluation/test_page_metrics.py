import pytest

from rag_eval_lab.core.models import (
    Chunk,
    EvaluationCase,
    EvaluationSample,
    GroundTruth,
    PageReference,
    RetrievedChunk,
)
from rag_eval_lab.evaluation.retrieval.page_metrics import PageHitRateAtK, PagePrecisionAtK, PageRecallAtK


def _chunk(chunk_id: str, document_id: str, page_start: int | None = None, page_end: int | None = None) -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, text="text", page_start=page_start, page_end=page_end)


def _sample(gold_page_references: list[PageReference], retrieved_chunks: list[RetrievedChunk]) -> EvaluationSample:
    case = EvaluationCase(
        query_id="q1",
        question="q?",
        ground_truth=GroundTruth(page_references=gold_page_references),
    )
    return EvaluationSample(evaluation_case=case, retrieved_chunks=retrieved_chunks)


def _retrieved() -> list[RetrievedChunk]:
    # Only c1 (rank 3) overlaps the gold pages; c2/c3/c4 do not, so hit rate
    # should flip from 0 to 1 once k reaches rank 3.
    return [
        RetrievedChunk(chunk=_chunk("c2", "d2", 11, 11), score=1.0, rank=1),
        RetrievedChunk(chunk=_chunk("c3", "d1"), score=1.0, rank=2),
        RetrievedChunk(chunk=_chunk("c1", "d1", 11, 11), score=1.0, rank=3),
        RetrievedChunk(chunk=_chunk("c4", "d1", 20, 21), score=1.0, rank=4),
    ]


def test_page_hit_rate_detects_overlap_within_k():
    gold = [PageReference(document_id="d1", page_start=10, page_end=12)]
    sample = _sample(gold, _retrieved())

    assert PageHitRateAtK(k=3).evaluate(sample).value == 1.0
    assert PageHitRateAtK(k=1).evaluate(sample).value == 0.0


def test_page_recall_and_precision_at_k():
    gold = [PageReference(document_id="d1", page_start=10, page_end=12)]
    sample = _sample(gold, _retrieved())

    recall_3 = PageRecallAtK(k=3).evaluate(sample)
    precision_3 = PagePrecisionAtK(k=3).evaluate(sample)
    assert recall_3.value == pytest.approx(1 / 3)
    assert precision_3.value == pytest.approx(1 / 2)

    precision_4 = PagePrecisionAtK(k=4).evaluate(sample)
    assert precision_4.value == pytest.approx(1 / 4)
    assert recall_3.k == 3


def test_different_documents_with_same_page_number_do_not_match():
    retrieved = [RetrievedChunk(chunk=_chunk("c1", "d2", 10, 10), score=1.0, rank=1)]
    gold = [PageReference(document_id="d1", page_start=10, page_end=10)]
    sample = _sample(gold, retrieved)

    assert PageHitRateAtK(k=1).evaluate(sample).value == 0.0


def test_chunks_without_page_ranges_are_ignored():
    retrieved = [RetrievedChunk(chunk=_chunk("c1", "d1"), score=1.0, rank=1)]
    gold = [PageReference(document_id="d1", page_start=1, page_end=5)]
    sample = _sample(gold, retrieved)

    assert PagePrecisionAtK(k=1).evaluate(sample).value == 0.0
    assert PageHitRateAtK(k=1).evaluate(sample).value == 0.0


def test_empty_gold_pages_returns_zero_recall():
    retrieved = [RetrievedChunk(chunk=_chunk("c1", "d1", 1, 1), score=1.0, rank=1)]
    sample = _sample([], retrieved)

    assert PageRecallAtK(k=1).evaluate(sample).value == 0.0


def test_empty_predicted_pages_returns_zero_precision():
    gold = [PageReference(document_id="d1", page_start=1, page_end=1)]
    sample = _sample(gold, [])

    assert PagePrecisionAtK(k=1).evaluate(sample).value == 0.0


def test_required_capabilities_is_page_references():
    for metric_cls in (PageHitRateAtK, PageRecallAtK, PagePrecisionAtK):
        assert metric_cls(k=1).required_capabilities == {"page_references"}
