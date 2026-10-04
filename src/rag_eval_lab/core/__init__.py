"""Core internal contracts: models, protocols, registry, and exceptions."""

from rag_eval_lab.core.exceptions import (
    ConfigurationError,
    GenerationError,
    RagEvalLabError,
    RegistryError,
    UnknownStrategyError,
    ValidationError,
)
from rag_eval_lab.core.models import (
    Chunk,
    Document,
    EvaluationCase,
    EvaluationSample,
    ExperimentResult,
    GroundTruth,
    GroundTruthCapability,
    MetricResult,
    PageReference,
    Query,
    RetrievedChunk,
)
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.core.paths import REPO_ROOT, resolve_under_repo_root
from rag_eval_lab.core.env import load_env_file

__all__ = [
    "Chunk",
    "ConfigurationError",
    "Document",
    "EvaluationCase",
    "EvaluationSample",
    "ExperimentResult",
    "GenerationError",
    "GroundTruth",
    "GroundTruthCapability",
    "MetricResult",
    "PageReference",
    "Query",
    "RagEvalLabError",
    "REPO_ROOT",
    "Registry",
    "RegistryError",
    "RetrievedChunk",
    "UnknownStrategyError",
    "ValidationError",
    "load_env_file",
    "resolve_under_repo_root",
]
