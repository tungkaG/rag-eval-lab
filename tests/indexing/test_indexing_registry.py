from rag_eval_lab.indexing.bm25_index import BM25Index
from rag_eval_lab.indexing.faiss_index import FaissIndex
from rag_eval_lab.indexing.registry import dense_index_registry, sparse_index_registry


def test_registered_dense_index_names():
    assert dense_index_registry.keys() == ["faiss"]


def test_registered_sparse_index_names():
    assert sparse_index_registry.keys() == ["bm25"]


def test_creates_faiss_index():
    assert isinstance(dense_index_registry.create("faiss"), FaissIndex)


def test_creates_bm25_index_with_constructor_arguments():
    index = sparse_index_registry.create("bm25", k1=1.2, b=0.8)

    assert isinstance(index, BM25Index)
    assert index.k1 == 1.2
    assert index.b == 0.8
