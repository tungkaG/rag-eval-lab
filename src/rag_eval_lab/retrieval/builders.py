"""Registered builders that prepare a retriever and its dependencies for one experiment.

Each builder owns preparing whatever its retrieval strategy needs (embedder,
dense index, sparse index); the experiment runner only resolves a builder by
`config.retrieval.strategy` and calls it. Adding a new retrieval strategy
means implementing and registering a new `RetrievalBuilder` here (plus its
retriever in `retrieval/registry.py`, if new) -- `runner.py` never changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rag_eval_lab.core.exceptions import ConfigurationError
from rag_eval_lab.core.models import Chunk
from rag_eval_lab.core.protocols import RetrievalStrategy
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.embedding.cache import compute_cache_key, load_cached_embeddings, save_cached_embeddings
from rag_eval_lab.embedding.registry import embedding_registry
from rag_eval_lab.indexing.base import Index, SparseIndex
from rag_eval_lab.indexing.registry import dense_index_registry, sparse_index_registry
from rag_eval_lab.retrieval.registry import retrieval_registry

if TYPE_CHECKING:
    from rag_eval_lab.experiments.config import ExperimentConfig


@dataclass(slots=True)
class PreparedRetrieval:
    """A ready-to-use retriever plus the preparation timings and metadata it incurred."""

    retriever: RetrievalStrategy
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrievalBuilder(Protocol):
    """Prepares a `RetrievalStrategy` and its dependencies for one experiment."""

    def build(self, config: "ExperimentConfig", chunks: list[Chunk]) -> PreparedRetrieval: ...


def _prepare_dense_index(
    config: "ExperimentConfig", chunks: list[Chunk]
) -> tuple[object, Index, dict[str, float], dict[str, Any]]:
    """Builds the embedder and dense index shared by the dense and hybrid builders.

    Corpus embeddings are loaded from the configured cache when enabled and
    not forced to recompute; the embedder is always constructed since query
    embedding still needs it. The dense index itself is always rebuilt.
    """
    if config.embedding is None:
        raise ConfigurationError(
            f"retrieval strategy '{config.retrieval.strategy}' requires an 'embedding' section"
        )
    if config.indexing.dense is None:
        raise ConfigurationError(
            f"retrieval strategy '{config.retrieval.strategy}' requires an 'indexing.dense' section"
        )

    timings: dict[str, float] = {}
    cache_config = config.embedding.cache
    metadata: dict[str, Any] = {"embedding_cache_enabled": cache_config.enabled}

    started = perf_counter()
    embedder = embedding_registry.create(config.embedding.strategy, **config.embedding.params)

    document_embeddings = None
    if cache_config.enabled:
        cache_key = compute_cache_key(config.embedding.strategy, config.embedding.params, chunks)
        metadata["embedding_cache_key"] = cache_key
        if not cache_config.force_recompute:
            document_embeddings = load_cached_embeddings(cache_config.directory, cache_key, expected_rows=len(chunks))
        metadata["embedding_cache_hit"] = document_embeddings is not None

    if document_embeddings is None:
        document_embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        if cache_config.enabled:
            save_cached_embeddings(cache_config.directory, cache_key, document_embeddings)
    timings["corpus_embedding_seconds"] = perf_counter() - started

    started = perf_counter()
    dense_index = dense_index_registry.create(config.indexing.dense.strategy, **config.indexing.dense.params)
    dense_index.add(chunks, document_embeddings)
    timings["index_building_seconds"] = perf_counter() - started

    return embedder, dense_index, timings, metadata


def _prepare_sparse_index(config: "ExperimentConfig", chunks: list[Chunk]) -> tuple[SparseIndex, dict[str, float]]:
    """Builds the sparse index shared by the bm25 and hybrid builders. Never creates an embedder."""
    if config.indexing.sparse is None:
        raise ConfigurationError(
            f"retrieval strategy '{config.retrieval.strategy}' requires an 'indexing.sparse' section"
        )

    started = perf_counter()
    sparse_index = sparse_index_registry.create(config.indexing.sparse.strategy, **config.indexing.sparse.params)
    sparse_index.build(chunks)
    timings = {"index_building_seconds": perf_counter() - started}

    return sparse_index, timings


def _merge_timings(*timing_dicts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for timings in timing_dicts:
        for key, value in timings.items():
            merged[key] = merged.get(key, 0.0) + value
    return merged


class DenseRetrievalBuilder:
    """Builds a `DenseRetriever` together with its embedder and dense index."""

    def build(self, config: "ExperimentConfig", chunks: list[Chunk]) -> PreparedRetrieval:
        embedder, dense_index, timings, metadata = _prepare_dense_index(config, chunks)
        retriever = retrieval_registry.create(
            "dense", embedder=embedder, index=dense_index, **config.retrieval.params
        )
        return PreparedRetrieval(retriever=retriever, timings=timings, metadata=metadata)


class Bm25RetrievalBuilder:
    """Builds a `BM25Retriever` together with its sparse index. Never creates an embedder."""

    def build(self, config: "ExperimentConfig", chunks: list[Chunk]) -> PreparedRetrieval:
        sparse_index, timings = _prepare_sparse_index(config, chunks)
        retriever = retrieval_registry.create("bm25", index=sparse_index, **config.retrieval.params)
        return PreparedRetrieval(retriever=retriever, timings=timings)


class HybridRetrievalBuilder:
    """Builds a `HybridRetriever` by preparing one dense and one sparse retriever."""

    def build(self, config: "ExperimentConfig", chunks: list[Chunk]) -> PreparedRetrieval:
        embedder, dense_index, dense_timings, metadata = _prepare_dense_index(config, chunks)
        sparse_index, sparse_timings = _prepare_sparse_index(config, chunks)

        dense_retriever = retrieval_registry.create("dense", embedder=embedder, index=dense_index)
        sparse_retriever = retrieval_registry.create("bm25", index=sparse_index)
        retriever = retrieval_registry.create(
            "hybrid",
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            **config.retrieval.params,
        )
        return PreparedRetrieval(
            retriever=retriever, timings=_merge_timings(dense_timings, sparse_timings), metadata=metadata
        )


retrieval_builder_registry: Registry[RetrievalBuilder] = Registry(kind="retrieval_builder")

retrieval_builder_registry.register("dense")(DenseRetrievalBuilder)
retrieval_builder_registry.register("bm25")(Bm25RetrievalBuilder)
retrieval_builder_registry.register("hybrid")(HybridRetrievalBuilder)
