from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.reranking.none import NoReranker


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="d1", text="text")


def test_returns_first_top_k_results_preserving_order_scores_and_objects():
    results = [
        RetrievedChunk(chunk=_chunk("c1"), score=0.9, rank=1),
        RetrievedChunk(chunk=_chunk("c2"), score=0.8, rank=2),
        RetrievedChunk(chunk=_chunk("c3"), score=0.7, rank=3),
    ]
    reranker = NoReranker()
    query = Query(id="q1", text="q")

    reranked = reranker.rerank(query, results, top_k=2)

    assert reranked == results[:2]
    assert reranked[0] is results[0]
    assert reranked[1] is results[1]


def test_empty_candidates_returns_empty_list():
    reranker = NoReranker()

    assert reranker.rerank(Query(id="q1", text="q"), [], top_k=5) == []


def test_top_k_larger_than_candidates_returns_all():
    results = [RetrievedChunk(chunk=_chunk("c1"), score=0.9, rank=1)]
    reranker = NoReranker()

    reranked = reranker.rerank(Query(id="q1", text="q"), results, top_k=10)

    assert reranked == results
