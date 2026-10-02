from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.retrieval.dense import DenseRetriever


class _FakeEmbedder:
    def __init__(self, query_vector):
        self.query_vector = query_vector
        self.embed_query_calls = []

    def embed_documents(self, texts):
        raise AssertionError("DenseRetriever must not embed documents or build an index")

    def embed_queries(self, texts):
        self.embed_query_calls.append(list(texts))
        return [self.query_vector]


class _FakeIndex:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def add(self, chunks, embeddings):
        raise AssertionError("DenseRetriever must not build the index")

    def search(self, query_embedding, top_k):
        self.search_calls.append((query_embedding, top_k))
        return self.results


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="d1", text="text")


def test_embeds_query_text_and_searches_the_supplied_index():
    results = [RetrievedChunk(chunk=_chunk("c1"), score=0.9, rank=1)]
    embedder = _FakeEmbedder(query_vector=[0.1, 0.2, 0.3])
    index = _FakeIndex(results)
    retriever = DenseRetriever(embedder=embedder, index=index)
    query = Query(id="q1", text="what is x?")

    retrieved = retriever.retrieve(query, top_k=5)

    assert embedder.embed_query_calls == [["what is x?"]]
    assert index.search_calls == [([0.1, 0.2, 0.3], 5)]
    assert retrieved is results


def test_does_not_embed_documents_or_build_the_index():
    embedder = _FakeEmbedder(query_vector=[1.0])
    index = _FakeIndex([])
    retriever = DenseRetriever(embedder=embedder, index=index)

    retriever.retrieve(Query(id="q1", text="q"), top_k=3)
