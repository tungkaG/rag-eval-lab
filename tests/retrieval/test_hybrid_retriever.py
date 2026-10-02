import pytest

from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.retrieval.hybrid import HybridRetriever


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="d1", text="text")


class _FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.retrieve_calls = []

    def retrieve(self, query, top_k):
        self.retrieve_calls.append((query, top_k))
        return self.results[:top_k]


def test_retrieves_candidates_with_configured_top_k_counts():
    dense_results = [RetrievedChunk(chunk=_chunk(f"d{i}"), score=1.0, rank=i + 1) for i in range(5)]
    sparse_results = [RetrievedChunk(chunk=_chunk(f"s{i}"), score=1.0, rank=i + 1) for i in range(5)]
    dense_retriever = _FakeRetriever(dense_results)
    sparse_retriever = _FakeRetriever(sparse_results)

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_top_k=3,
        sparse_top_k=2,
    )
    query = Query(id="q1", text="q")

    hybrid.retrieve(query, top_k=10)

    assert dense_retriever.retrieve_calls == [(query, 3)]
    assert sparse_retriever.retrieve_calls == [(query, 2)]


def test_final_output_respects_top_k_independent_of_candidate_counts():
    dense_results = [RetrievedChunk(chunk=_chunk(f"d{i}"), score=1.0, rank=i + 1) for i in range(5)]
    sparse_results = [RetrievedChunk(chunk=_chunk(f"s{i}"), score=1.0, rank=i + 1) for i in range(5)]
    dense_retriever = _FakeRetriever(dense_results)
    sparse_retriever = _FakeRetriever(sparse_results)

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_top_k=5,
        sparse_top_k=5,
    )

    fused = hybrid.retrieve(Query(id="q1", text="q"), top_k=4)

    assert len(fused) == 4
    assert [result.rank for result in fused] == [1, 2, 3, 4]


def test_fuses_dense_and_sparse_results_via_rrf():
    shared_chunk = _chunk("shared")
    dense_results = [RetrievedChunk(chunk=shared_chunk, score=0.9, rank=1)]
    sparse_results = [RetrievedChunk(chunk=shared_chunk, score=5.0, rank=1)]
    dense_retriever = _FakeRetriever(dense_results)
    sparse_retriever = _FakeRetriever(sparse_results)

    hybrid = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever)

    fused = hybrid.retrieve(Query(id="q1", text="q"), top_k=5)

    assert len(fused) == 1
    assert fused[0].chunk.id == "shared"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 61)
