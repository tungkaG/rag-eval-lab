import pytest

from rag_eval_lab.core.models import Chunk, RetrievedChunk
from rag_eval_lab.retrieval.rrf import reciprocal_rank_fusion


def _chunk(chunk_id: str, document_id: str = "d1") -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, text="text")


def test_fuses_scores_and_orders_by_descending_fused_score():
    chunk_a = _chunk("a")
    chunk_b = _chunk("b")
    chunk_c = _chunk("c")

    list1 = [
        RetrievedChunk(chunk=chunk_a, score=0.9, rank=1),
        RetrievedChunk(chunk=chunk_b, score=0.5, rank=2),
    ]
    list2 = [
        RetrievedChunk(chunk=chunk_b, score=5.0, rank=1),
        RetrievedChunk(chunk=chunk_c, score=4.0, rank=2),
    ]

    fused = reciprocal_rank_fusion([list1, list2], top_k=10, rrf_k=60)

    scores_by_id = {result.chunk.id: result.score for result in fused}
    assert scores_by_id["a"] == pytest.approx(1 / 61)
    assert scores_by_id["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores_by_id["c"] == pytest.approx(1 / 62)
    assert [result.chunk.id for result in fused] == ["b", "a", "c"]
    assert [result.rank for result in fused] == [1, 2, 3]


def test_chunk_in_both_lists_receives_contributions_from_both():
    chunk_shared = _chunk("shared")
    list1 = [RetrievedChunk(chunk=chunk_shared, score=1.0, rank=1)]
    list2 = [RetrievedChunk(chunk=chunk_shared, score=1.0, rank=3)]

    fused = reciprocal_rank_fusion([list1, list2], top_k=10)

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_different_chunks_from_same_document_remain_separate():
    chunk_1 = _chunk("doc1_chunk_0", document_id="doc1")
    chunk_2 = _chunk("doc1_chunk_1", document_id="doc1")
    list1 = [
        RetrievedChunk(chunk=chunk_1, score=1.0, rank=1),
        RetrievedChunk(chunk=chunk_2, score=0.9, rank=2),
    ]

    fused = reciprocal_rank_fusion([list1], top_k=10)

    assert {result.chunk.id for result in fused} == {"doc1_chunk_0", "doc1_chunk_1"}
    assert len(fused) == 2


def test_fusion_leaves_input_results_unchanged():
    original = RetrievedChunk(chunk=_chunk("a"), score=0.9, rank=1)
    list1 = [original]
    list1_snapshot = list(list1)

    reciprocal_rank_fusion([list1], top_k=10)

    assert list1 == list1_snapshot
    assert list1[0].score == 0.9
    assert list1[0].rank == 1


def test_respects_top_k_limit():
    chunks = [_chunk(f"c{i}") for i in range(5)]
    list1 = [RetrievedChunk(chunk=chunk, score=1.0, rank=i + 1) for i, chunk in enumerate(chunks)]

    fused = reciprocal_rank_fusion([list1], top_k=3)

    assert len(fused) == 3
    assert [result.rank for result in fused] == [1, 2, 3]


def test_ties_broken_deterministically_by_chunk_id():
    chunk_b = _chunk("b")
    chunk_a = _chunk("a")
    list1 = [RetrievedChunk(chunk=chunk_b, score=1.0, rank=1)]
    list2 = [RetrievedChunk(chunk=chunk_a, score=1.0, rank=1)]

    fused = reciprocal_rank_fusion([list1, list2], top_k=10)

    assert [result.chunk.id for result in fused] == ["a", "b"]
