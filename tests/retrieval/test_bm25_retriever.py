from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.retrieval.bm25 import BM25Retriever


class _FakeSparseIndex:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def build(self, chunks):
        raise AssertionError("BM25Retriever must not build the index")

    def search(self, query_text, top_k):
        self.search_calls.append((query_text, top_k))
        return self.results


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="d1", text="text")


def test_searches_the_supplied_sparse_index_with_raw_query_text():
    results = [RetrievedChunk(chunk=_chunk("c1"), score=2.5, rank=1)]
    index = _FakeSparseIndex(results)
    retriever = BM25Retriever(index=index)
    query = Query(id="q1", text="Temperature Limits")

    retrieved = retriever.retrieve(query, top_k=5)

    assert index.search_calls == [("Temperature Limits", 5)]
    assert retrieved is results


def test_does_not_build_the_index():
    index = _FakeSparseIndex([])
    retriever = BM25Retriever(index=index)

    retriever.retrieve(Query(id="q1", text="q"), top_k=3)
