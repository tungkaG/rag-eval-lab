"""Core internal data model shared by every component in rag_eval_lab.

These models are the only objects that cross boundaries between layers
(ingestion, chunking, retrieval, reranking, evaluation, ...). Third party
library types (Haystack, BEIR, FAISS, Sentence Transformers, RAGAS, ...)
must never be used in place of these models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rag_eval_lab.core.exceptions import ValidationError

GroundTruthCapability = Literal[
    "document_relevance",
    "page_references",
    "reference_answer",
    "evidence_texts",
    "keywords",
]


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")


@dataclass(slots=True)
class Document:
    """A single source document in the corpus."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "Document.id")


@dataclass(slots=True)
class Chunk:
    """A slice of a `Document` produced by a chunking strategy."""

    id: str
    document_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "Chunk.id")
        _require_non_empty(self.document_id, "Chunk.document_id")
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValidationError("Chunk.page_end must be >= page_start")
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValidationError("Chunk.char_end must be >= char_start")


@dataclass(slots=True)
class Query:
    """A single evaluation question posed against the corpus."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "Query.id")


@dataclass(slots=True)
class PageReference:
    """A gold page range within a specific document."""

    document_id: str
    page_start: int
    page_end: int

    def __post_init__(self) -> None:
        _require_non_empty(self.document_id, "PageReference.document_id")
        if self.page_end < self.page_start:
            raise ValidationError("PageReference.page_end must be >= page_start")


@dataclass(slots=True)
class GroundTruth:
    """Whatever ground truth information happens to be available for a query.

    Every field is optional. `capabilities` is computed from whichever fields
    are populated and determines which metrics the evaluation planner can
    select (see `Metric.required_capabilities`).
    """

    document_relevance: dict[str, int] = field(default_factory=dict)
    page_references: list[PageReference] = field(default_factory=list)
    reference_answer: str | None = None
    evidence_texts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def capabilities(self) -> set[GroundTruthCapability]:
        """The ground truth capabilities available from the currently populated fields.

        Recomputed from the fields on every access, so it can never drift out
        of sync with the data it describes.
        """
        caps: set[GroundTruthCapability] = set()
        if self.document_relevance:
            caps.add("document_relevance")
        if self.page_references:
            caps.add("page_references")
        if self.reference_answer is not None and self.reference_answer.strip():
            caps.add("reference_answer")
        if self.evidence_texts:
            caps.add("evidence_texts")
        if self.keywords:
            caps.add("keywords")
        return caps


@dataclass(slots=True)
class EvaluationCase:
    """A single benchmark item: a question paired with its ground truth."""

    query_id: str
    question: str
    ground_truth: GroundTruth

    def __post_init__(self) -> None:
        _require_non_empty(self.query_id, "EvaluationCase.query_id")
        _require_non_empty(self.question, "EvaluationCase.question")


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by a retrieval or reranking strategy."""

    chunk: Chunk
    score: float
    rank: int
    retriever_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValidationError("RetrievedChunk.rank must be >= 0")


@dataclass(slots=True)
class EvaluationSample:
    """Everything produced for one `EvaluationCase` during an experiment run."""

    evaluation_case: EvaluationCase
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    generated_answer: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MetricResult:
    """The value produced by a single metric for a single sample or experiment."""

    name: str
    value: float
    k: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "MetricResult.name")


@dataclass(slots=True)
class ExperimentResult:
    """The aggregated, reproducible outcome of a single experiment run."""

    experiment_id: str
    config: dict[str, Any]
    metrics: list[MetricResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.experiment_id, "ExperimentResult.experiment_id")
