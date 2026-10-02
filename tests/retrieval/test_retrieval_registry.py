from rag_eval_lab.retrieval.bm25 import BM25Retriever
from rag_eval_lab.retrieval.dense import DenseRetriever
from rag_eval_lab.retrieval.hybrid import HybridRetriever
from rag_eval_lab.retrieval.registry import retrieval_registry


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_queries(self, texts):
        return [[0.0] for _ in texts]


class _FakeIndex:
    def add(self, chunks, embeddings):
        pass

    def search(self, query_embedding, top_k):
        return []


class _FakeSparseIndex:
    def build(self, chunks):
        pass

    def search(self, query_text, top_k):
        return []


def test_registered_retrieval_strategy_names():
    assert retrieval_registry.keys() == ["bm25", "dense", "hybrid"]


def test_creates_dense_retriever_with_constructor_arguments():
    retriever = retrieval_registry.create("dense", embedder=_FakeEmbedder(), index=_FakeIndex())

    assert isinstance(retriever, DenseRetriever)


def test_creates_bm25_retriever_with_constructor_arguments():
    retriever = retrieval_registry.create("bm25", index=_FakeSparseIndex())

    assert isinstance(retriever, BM25Retriever)


def test_creates_hybrid_retriever_with_constructor_arguments():
    dense_retriever = retrieval_registry.create("dense", embedder=_FakeEmbedder(), index=_FakeIndex())
    bm25_retriever = retrieval_registry.create("bm25", index=_FakeSparseIndex())

    hybrid = retrieval_registry.create(
        "hybrid",
        dense_retriever=dense_retriever,
        sparse_retriever=bm25_retriever,
        dense_top_k=10,
        sparse_top_k=10,
    )

    assert isinstance(hybrid, HybridRetriever)
