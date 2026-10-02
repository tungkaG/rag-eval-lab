from rag_eval_lab.core.models import Chunk, RetrievedChunk
from rag_eval_lab.indexing.bm25_index import BM25Index


def _chunk(document_id: str, index: int, text: str) -> Chunk:
    return Chunk(id=f"{document_id}_chunk_{index}", document_id=document_id, text=text)


def test_retrieves_matching_chunk_from_small_corpus():
    chunks = [
        _chunk("d1", 0, "The maximum operating temperature is 85 degrees Celsius."),
        _chunk("d2", 0, "Bananas are a good source of potassium."),
        _chunk("d3", 0, "The device should not exceed 85 degrees during operation."),
    ]
    index = BM25Index()
    index.build(chunks)

    results = index.search("maximum operating temperature", top_k=3)

    assert results[0].chunk.id == "d1_chunk_0"


def test_returns_internal_retrieved_chunk_objects_with_ordered_ranks():
    chunks = [_chunk("d1", 0, "alpha beta gamma"), _chunk("d2", 0, "delta epsilon zeta")]
    index = BM25Index()
    index.build(chunks)

    results = index.search("alpha", top_k=2)

    assert all(isinstance(result, RetrievedChunk) for result in results)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))


def test_top_k_limited_to_corpus_size():
    chunks = [_chunk("d1", 0, "alpha"), _chunk("d2", 0, "beta")]
    index = BM25Index()
    index.build(chunks)

    results = index.search("alpha", top_k=10)

    assert len(results) == 2


def test_empty_corpus_returns_empty_list():
    index = BM25Index()
    index.build([])

    assert index.search("alpha", top_k=5) == []


def test_build_replaces_previous_index():
    index = BM25Index()
    index.build([_chunk("d1", 0, "alpha")])
    index.build([_chunk("d2", 0, "beta")])

    results = index.search("beta", top_k=1)

    assert results[0].chunk.id == "d2_chunk_0"


def test_tokenizer_is_case_insensitive():
    chunks = [_chunk("d1", 0, "Temperature Limits"), _chunk("d2", 0, "Bananas are tasty")]
    index = BM25Index()
    index.build(chunks)

    lower_results = index.search("temperature limits", top_k=1)
    upper_results = index.search("TEMPERATURE LIMITS", top_k=1)

    assert lower_results[0].chunk.id == upper_results[0].chunk.id == "d1_chunk_0"
    assert lower_results[0].score == upper_results[0].score
