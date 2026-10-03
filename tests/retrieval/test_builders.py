"""Unit tests for `retrieval.builders`, using a fake embedder so no model download is needed."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError
from rag_eval_lab.core.models import Chunk
from rag_eval_lab.embedding.registry import embedding_registry
from rag_eval_lab.experiments.config import (
    CacheConfig,
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    IndexingConfig,
    OutputConfig,
    RerankingConfig,
    RetrievalConfig,
    StrategyConfig,
)
from rag_eval_lab.indexing.registry import dense_index_registry, sparse_index_registry
from rag_eval_lab.retrieval.bm25 import BM25Retriever
from rag_eval_lab.retrieval.builders import (
    Bm25RetrievalBuilder,
    DenseRetrievalBuilder,
    HybridRetrievalBuilder,
    retrieval_builder_registry,
)
from rag_eval_lab.retrieval.dense import DenseRetriever
from rag_eval_lab.retrieval.hybrid import HybridRetriever


class _CallCounter:
    def __init__(self) -> None:
        self.embed_documents_calls = 0


def _fake_embedder_factory(counter: _CallCounter | None = None):
    """A deterministic, network-free stand-in for `SentenceTransformerEmbedding`."""

    class _FakeEmbedder:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if counter is not None:
                counter.embed_documents_calls += 1
            return [[float(len(text))] for text in texts]

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    return _FakeEmbedder


def _make_config(
    retrieval_strategy: str,
    retrieval_params: dict | None = None,
    cache: CacheConfig | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        dataset=DatasetConfig(adapter="custom", params={}),
        chunking=StrategyConfig(strategy="fixed_tokens", params={}),
        retrieval=RetrievalConfig(strategy=retrieval_strategy, top_k=5, params=retrieval_params or {}),
        reranking=RerankingConfig(strategy="none", final_k=5, params={}),
        embedding=EmbeddingConfig(strategy="fake_embedder", params={}, cache=cache or CacheConfig()),
        indexing=IndexingConfig(
            dense=StrategyConfig(strategy="faiss", params={}),
            sparse=StrategyConfig(strategy="bm25", params={}),
        ),
        evaluation=EvaluationConfig(k_values=[1, 5]),
        output=OutputConfig(directory="results"),
        raw={},
    )


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", document_id="doc1", text="hello world"),
        Chunk(id="c2", document_id="doc2", text="goodbye world"),
    ]


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(embedding_registry._factories, "fake_embedder", _fake_embedder_factory())


def test_dense_builder_builds_retriever_and_timings(fake_embedder) -> None:
    config = _make_config(retrieval_strategy="dense")

    prepared = DenseRetrievalBuilder().build(config, _make_chunks())

    assert isinstance(prepared.retriever, DenseRetriever)
    assert "corpus_embedding_seconds" in prepared.timings
    assert "index_building_seconds" in prepared.timings
    assert prepared.metadata["embedding_cache_enabled"] is False


def test_dense_builder_missing_embedding_section_raises(fake_embedder) -> None:
    config = replace(_make_config(retrieval_strategy="dense"), embedding=None)

    with pytest.raises(ConfigurationError):
        DenseRetrievalBuilder().build(config, _make_chunks())


def test_dense_builder_missing_indexing_dense_section_raises(fake_embedder) -> None:
    config = replace(_make_config(retrieval_strategy="dense"), indexing=IndexingConfig(dense=None, sparse=None))

    with pytest.raises(ConfigurationError):
        DenseRetrievalBuilder().build(config, _make_chunks())


def test_bm25_builder_builds_retriever_and_timings() -> None:
    config = _make_config(retrieval_strategy="bm25")

    prepared = Bm25RetrievalBuilder().build(config, _make_chunks())

    assert isinstance(prepared.retriever, BM25Retriever)
    assert "index_building_seconds" in prepared.timings
    assert "corpus_embedding_seconds" not in prepared.timings


def test_bm25_builder_never_creates_an_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("embedding_registry.create must not be called by the BM25 builder")

    monkeypatch.setattr(embedding_registry, "create", _fail)

    Bm25RetrievalBuilder().build(_make_config(retrieval_strategy="bm25"), _make_chunks())


def test_hybrid_builder_builds_retriever_and_merges_timings(fake_embedder) -> None:
    config = _make_config(retrieval_strategy="hybrid", retrieval_params={"dense_top_k": 3, "sparse_top_k": 3})

    prepared = HybridRetrievalBuilder().build(config, _make_chunks())

    assert isinstance(prepared.retriever, HybridRetriever)
    assert "corpus_embedding_seconds" in prepared.timings
    assert "index_building_seconds" in prepared.timings


def test_hybrid_builder_prepares_each_index_once(fake_embedder, monkeypatch: pytest.MonkeyPatch) -> None:
    embedding_calls: list[str] = []
    dense_index_calls: list[str] = []
    sparse_index_calls: list[str] = []

    original_embedding_create = embedding_registry.create
    original_dense_create = dense_index_registry.create
    original_sparse_create = sparse_index_registry.create

    def counting_embedding_create(name: str, *args: object, **kwargs: object):
        embedding_calls.append(name)
        return original_embedding_create(name, *args, **kwargs)

    def counting_dense_create(name: str, *args: object, **kwargs: object):
        dense_index_calls.append(name)
        return original_dense_create(name, *args, **kwargs)

    def counting_sparse_create(name: str, *args: object, **kwargs: object):
        sparse_index_calls.append(name)
        return original_sparse_create(name, *args, **kwargs)

    monkeypatch.setattr(embedding_registry, "create", counting_embedding_create)
    monkeypatch.setattr(dense_index_registry, "create", counting_dense_create)
    monkeypatch.setattr(sparse_index_registry, "create", counting_sparse_create)

    HybridRetrievalBuilder().build(_make_config(retrieval_strategy="hybrid"), _make_chunks())

    assert embedding_calls == ["fake_embedder"]
    assert dense_index_calls == ["faiss"]
    assert sparse_index_calls == ["bm25"]


def test_additional_retrieval_builder_can_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_eval_lab.retrieval.builders import PreparedRetrieval

    class _FakeRetriever:
        def retrieve(self, query, top_k):
            return []

    class _FakeRetrievalBuilder:
        def build(self, config, chunks) -> PreparedRetrieval:
            return PreparedRetrieval(retriever=_FakeRetriever())

    monkeypatch.setitem(retrieval_builder_registry._factories, "fake_strategy", _FakeRetrievalBuilder)

    builder = retrieval_builder_registry.create("fake_strategy")
    prepared = builder.build(_make_config(retrieval_strategy="fake_strategy"), _make_chunks())

    assert isinstance(prepared.retriever, _FakeRetriever)


def test_dense_builder_cache_miss_computes_and_saves_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    monkeypatch.setitem(embedding_registry._factories, "fake_embedder", _fake_embedder_factory(counter))
    cache_dir = tmp_path / "cache"
    config = _make_config(retrieval_strategy="dense", cache=CacheConfig(enabled=True, directory=str(cache_dir)))

    prepared = DenseRetrievalBuilder().build(config, _make_chunks())

    assert counter.embed_documents_calls == 1
    assert prepared.metadata["embedding_cache_enabled"] is True
    assert prepared.metadata["embedding_cache_hit"] is False
    assert "embedding_cache_key" in prepared.metadata
    assert list(cache_dir.glob("*.npy"))


def test_dense_builder_cache_hit_skips_embed_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    monkeypatch.setitem(embedding_registry._factories, "fake_embedder", _fake_embedder_factory(counter))
    cache_dir = tmp_path / "cache"
    config = _make_config(retrieval_strategy="dense", cache=CacheConfig(enabled=True, directory=str(cache_dir)))
    chunks = _make_chunks()

    DenseRetrievalBuilder().build(config, chunks)
    assert counter.embed_documents_calls == 1

    prepared = DenseRetrievalBuilder().build(config, chunks)

    assert counter.embed_documents_calls == 1  # second, identical run is a cache hit
    assert prepared.metadata["embedding_cache_hit"] is True


def test_dense_builder_force_recompute_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    monkeypatch.setitem(embedding_registry._factories, "fake_embedder", _fake_embedder_factory(counter))
    cache_dir = tmp_path / "cache"
    chunks = _make_chunks()

    DenseRetrievalBuilder().build(
        _make_config(retrieval_strategy="dense", cache=CacheConfig(enabled=True, directory=str(cache_dir))), chunks
    )
    assert counter.embed_documents_calls == 1

    forced_config = _make_config(
        retrieval_strategy="dense",
        cache=CacheConfig(enabled=True, directory=str(cache_dir), force_recompute=True),
    )
    prepared = DenseRetrievalBuilder().build(forced_config, chunks)

    assert counter.embed_documents_calls == 2
    assert prepared.metadata["embedding_cache_hit"] is False


def test_dense_builder_disabled_cache_does_not_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _CallCounter()
    monkeypatch.setitem(embedding_registry._factories, "fake_embedder", _fake_embedder_factory(counter))
    cache_dir = tmp_path / "cache"
    config = _make_config(retrieval_strategy="dense", cache=CacheConfig(enabled=False, directory=str(cache_dir)))
    chunks = _make_chunks()

    DenseRetrievalBuilder().build(config, chunks)
    DenseRetrievalBuilder().build(config, chunks)

    assert counter.embed_documents_calls == 2  # recomputed every time, cache never consulted
    assert not cache_dir.exists()


def test_bm25_builder_ignores_embedding_cache_config(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    config = _make_config(retrieval_strategy="bm25", cache=CacheConfig(enabled=True, directory=str(cache_dir)))

    Bm25RetrievalBuilder().build(config, _make_chunks())

    assert not cache_dir.exists()
