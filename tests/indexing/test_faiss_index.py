import pytest

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.core.models import Chunk, RetrievedChunk
from rag_eval_lab.indexing.faiss_index import FaissIndex


def _chunk(document_id: str, index: int) -> Chunk:
    return Chunk(id=f"{document_id}_chunk_{index}", document_id=document_id, text="text")


def test_search_ranks_by_cosine_similarity():
    index = FaissIndex()
    chunks = [_chunk("d1", 0), _chunk("d1", 1), _chunk("d1", 2)]
    index.add(chunks, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    results = index.search([0.9, 0.1], top_k=3)

    assert [result.chunk.id for result in results] == ["d1_chunk_0", "d1_chunk_1", "d1_chunk_2"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score > results[1].score > results[2].score


def test_returns_internal_retrieved_chunk_objects():
    index = FaissIndex()
    index.add([_chunk("d1", 0)], [[1.0, 0.0]])

    results = index.search([1.0, 0.0], top_k=1)

    assert all(isinstance(result, RetrievedChunk) for result in results)


def test_top_k_limited_to_indexed_chunk_count():
    index = FaissIndex()
    index.add([_chunk("d1", 0), _chunk("d1", 1)], [[1.0, 0.0], [0.0, 1.0]])

    results = index.search([1.0, 0.0], top_k=10)

    assert len(results) == 2


def test_empty_index_returns_empty_list():
    index = FaissIndex()

    assert index.search([1.0, 0.0], top_k=5) == []


def test_preserves_chunk_mapping_across_multiple_add_calls():
    index = FaissIndex()
    index.add([_chunk("d1", 0)], [[1.0, 0.0]])
    index.add([_chunk("d2", 0)], [[0.0, 1.0]])

    results = index.search([0.0, 1.0], top_k=2)

    assert results[0].chunk.id == "d2_chunk_0"
    assert results[1].chunk.id == "d1_chunk_0"


def test_normalizes_unnormalized_vectors_to_cosine_similarity():
    index = FaissIndex()
    index.add([_chunk("d1", 0)], [[3.0, 0.0]])

    results = index.search([5.0, 0.0], top_k=1)

    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_mismatched_chunk_and_embedding_counts_are_rejected():
    index = FaissIndex()

    with pytest.raises(ValidationError):
        index.add([_chunk("d1", 0), _chunk("d1", 1)], [[1.0, 0.0]])

    assert index.search([1.0, 0.0], top_k=5) == []
