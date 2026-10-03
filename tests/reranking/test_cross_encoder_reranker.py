from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.reranking.cross_encoder import CrossEncoderReranker


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(id=chunk_id, document_id="d1", text=text)


def _reranker(mock_model, **kwargs) -> CrossEncoderReranker:
    with patch("rag_eval_lab.reranking.cross_encoder.CrossEncoder", return_value=mock_model):
        return CrossEncoderReranker(model_name="fake-model", **kwargs)


def test_scores_pairs_of_query_text_and_chunk_text():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
    reranker = _reranker(mock_model)

    results = [
        RetrievedChunk(chunk=_chunk("a", "alpha"), score=1.0, rank=1),
        RetrievedChunk(chunk=_chunk("b", "beta"), score=0.9, rank=2),
        RetrievedChunk(chunk=_chunk("c", "gamma"), score=0.8, rank=3),
    ]
    query = Query(id="q1", text="what is x?")

    reranker.rerank(query, results, top_k=3)

    mock_model.predict.assert_called_once()
    pairs = mock_model.predict.call_args.args[0]
    assert pairs == [("what is x?", "alpha"), ("what is x?", "beta"), ("what is x?", "gamma")]


def test_sorts_by_descending_score_and_assigns_ranks_from_one():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
    reranker = _reranker(mock_model)

    results = [
        RetrievedChunk(chunk=_chunk("a"), score=1.0, rank=1),
        RetrievedChunk(chunk=_chunk("b"), score=0.9, rank=2),
        RetrievedChunk(chunk=_chunk("c"), score=0.8, rank=3),
    ]

    reranked = reranker.rerank(Query(id="q1", text="q"), results, top_k=3)

    assert [result.chunk.id for result in reranked] == ["b", "c", "a"]
    assert [result.rank for result in reranked] == [1, 2, 3]
    assert [result.score for result in reranked] == pytest.approx([0.9, 0.5, 0.1])


def test_ties_preserve_input_order():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5, 0.5, 0.5])
    reranker = _reranker(mock_model)

    results = [
        RetrievedChunk(chunk=_chunk("a"), score=1.0, rank=1),
        RetrievedChunk(chunk=_chunk("b"), score=0.9, rank=2),
        RetrievedChunk(chunk=_chunk("c"), score=0.8, rank=3),
    ]

    reranked = reranker.rerank(Query(id="q1", text="q"), results, top_k=3)

    assert [result.chunk.id for result in reranked] == ["a", "b", "c"]


def test_limits_to_top_k():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
    reranker = _reranker(mock_model)

    results = [
        RetrievedChunk(chunk=_chunk("a"), score=1.0, rank=1),
        RetrievedChunk(chunk=_chunk("b"), score=0.9, rank=2),
        RetrievedChunk(chunk=_chunk("c"), score=0.8, rank=3),
    ]

    reranked = reranker.rerank(Query(id="q1", text="q"), results, top_k=2)

    assert len(reranked) == 2
    assert [result.chunk.id for result in reranked] == ["b", "c"]


def test_preserves_original_chunk_objects_and_copies_retriever_metadata():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5])
    reranker = _reranker(mock_model)

    chunk = _chunk("a")
    original = RetrievedChunk(chunk=chunk, score=1.0, rank=1, retriever_metadata={"source": "dense"})

    reranked = reranker.rerank(Query(id="q1", text="q"), [original], top_k=1)

    assert reranked[0].chunk is chunk
    assert reranked[0].retriever_metadata == {"source": "dense"}
    assert reranked[0].retriever_metadata is not original.retriever_metadata


def test_does_not_mutate_input_results():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.9])
    reranker = _reranker(mock_model)

    original = RetrievedChunk(chunk=_chunk("a"), score=1.0, rank=1)
    results = [original]

    reranker.rerank(Query(id="q1", text="q"), results, top_k=1)

    assert results[0] is original
    assert results[0].score == 1.0
    assert results[0].rank == 1


def test_empty_candidates_returns_empty_list_without_calling_predict():
    mock_model = MagicMock()
    reranker = _reranker(mock_model)

    reranked = reranker.rerank(Query(id="q1", text="q"), [], top_k=5)

    assert reranked == []
    mock_model.predict.assert_not_called()


def test_model_is_loaded_once_per_instance():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5])

    with patch(
        "rag_eval_lab.reranking.cross_encoder.CrossEncoder",
        return_value=mock_model,
    ) as mock_cls:
        reranker = CrossEncoderReranker(model_name="fake-model", device="cpu")
        reranker.rerank(Query(id="q1", text="q"), [RetrievedChunk(chunk=_chunk("a"), score=1.0, rank=1)], top_k=1)
        reranker.rerank(Query(id="q1", text="q"), [RetrievedChunk(chunk=_chunk("b"), score=1.0, rank=1)], top_k=1)

    mock_cls.assert_called_once_with("fake-model", device="cpu")
