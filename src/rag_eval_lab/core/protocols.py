"""Interfaces implemented by every pluggable strategy in rag_eval_lab.

Only structural contracts live here. No algorithm logic, and no dependency
on any particular third party library, belongs in this module.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rag_eval_lab.core.models import (
    Chunk,
    Document,
    EvaluationCase,
    EvaluationSample,
    ExperimentResult,
    GroundTruthCapability,
    MetricResult,
    Query,
    RetrievedChunk,
)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Normalizes an external benchmark or custom dataset into core models."""

    def documents(self) -> list[Document]: ...

    def evaluation_cases(self) -> list[EvaluationCase]: ...


@runtime_checkable
class DocumentLoader(Protocol):
    """Reads a normalized input format (TXT, Markdown, JSONL, ...) into `Document`s."""

    def load(self, source: Any) -> list[Document]: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Splits a `Document` into `Chunk`s."""

    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class EmbeddingStrategy(Protocol):
    """Turns raw text into dense vector representations.

    Documents and queries are embedded through separate methods so a
    strategy can apply different prompts or encoding methods to each side.
    Symmetric strategies may implement both methods with the same underlying
    model call. In both cases, returned vectors correspond to the input
    texts in the same order.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Index(Protocol):
    """Stores chunk embeddings and serves nearest neighbor search."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Returns ranked `RetrievedChunk`s for a query."""

    def retrieve(self, query: Query, top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class RerankingStrategy(Protocol):
    """Reorders or rescores an existing list of `RetrievedChunk`s."""

    def rerank(
        self,
        query: Query,
        retrieved_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class GenerationStrategy(Protocol):
    """Produces a generated answer from a query and its retrieved chunks."""

    def generate(
        self,
        query: Query,
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[str, dict[str, Any]]: ...


@runtime_checkable
class Metric(Protocol):
    """Evaluates a completed `EvaluationSample`.

    `required_capabilities` declares which ground truth capabilities must be
    available for this metric to be applicable; the evaluation planner uses
    it to automatically skip metrics whose requirements are unmet.
    """

    required_capabilities: set[GroundTruthCapability]

    def evaluate(self, sample: EvaluationSample) -> MetricResult: ...


@runtime_checkable
class ResultStore(Protocol):
    """Persists and retrieves `ExperimentResult`s."""

    def save(self, result: ExperimentResult) -> None: ...

    def load(self, experiment_id: str) -> ExperimentResult: ...

    def list_experiments(self) -> list[str]: ...
