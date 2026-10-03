"""Lean, stdlib-dataclass experiment configuration loaded from YAML.

Strategy constructor parameters stay as plain dictionaries (`params`) and are
passed straight through to the existing registries via `**params` -- there
are no per-strategy configuration classes.

Relative paths used inside a configuration file (for example
`dataset.params.dataset_dir` or `output.directory`) are interpreted relative
to the current working directory the process was launched from (expected to
be the repository root), not relative to the YAML file's own location.

`embedding.cache.directory` is the exception: it always resolves under the
repository root (via `core.paths.resolve_under_repo_root`), regardless of
the current working directory, so caches land in one stable place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_eval_lab.core.exceptions import ConfigurationError


@dataclass(slots=True)
class StrategyConfig:
    """A registry strategy name plus the keyword arguments to construct it."""

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    """`dataset_registry` strategy name plus its constructor keyword arguments."""

    adapter: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CacheConfig:
    """Optional on-disk corpus embedding cache settings, kept separate from `params`.

    `directory` is resolved under the repository root, not the current
    working directory -- see `core.paths.resolve_under_repo_root`.
    """

    enabled: bool = False
    directory: str = ".cache/embeddings"
    force_recompute: bool = False


@dataclass(slots=True)
class EmbeddingConfig:
    """`embedding_registry` strategy name, its constructor keyword arguments, and cache settings."""

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass(slots=True)
class RetrievalConfig:
    """`retrieval_registry` strategy name, the candidate count, and extra params."""

    strategy: str
    top_k: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankingConfig:
    """`reranking_registry` strategy name, the final result count, and extra params."""

    strategy: str
    final_k: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexingConfig:
    """Optional dense/sparse index strategy configuration.

    Only the sections required by the selected retrieval strategy need to be
    present; a BM25-only experiment can omit `dense` entirely.
    """

    dense: StrategyConfig | None = None
    sparse: StrategyConfig | None = None


@dataclass(slots=True)
class EvaluationConfig:
    """`k_values` passed straight through to `EvaluationEngine`."""

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass(slots=True)
class OutputConfig:
    """Where the `ResultStore` writes experiment results."""

    directory: str = "results"


@dataclass(slots=True)
class ExperimentConfig:
    """The fully parsed configuration for a single experiment run."""

    name: str
    dataset: DatasetConfig
    chunking: StrategyConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    embedding: EmbeddingConfig | None = None
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Loads and validates an `ExperimentConfig` from a YAML file at `path`."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _parse_config(raw)


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigurationError(f"Missing required '{name}' section in experiment configuration")
    return section


def _required(section: dict[str, Any], key: str, path: str) -> Any:
    value = section.get(key)
    if value in (None, ""):
        raise ConfigurationError(f"Missing required '{path}' in experiment configuration")
    return value


def _strategy_config(section: dict[str, Any], path: str) -> StrategyConfig:
    return StrategyConfig(
        strategy=_required(section, "strategy", f"{path}.strategy"),
        params=dict(section.get("params") or {}),
    )


def _embedding_config(section: dict[str, Any]) -> EmbeddingConfig:
    cache_section = section.get("cache") or {}
    cache = CacheConfig(
        enabled=bool(cache_section.get("enabled", False)),
        directory=cache_section.get("directory") or ".cache/embeddings",
        force_recompute=bool(cache_section.get("force_recompute", False)),
    )
    return EmbeddingConfig(
        strategy=_required(section, "strategy", "embedding.strategy"),
        params=dict(section.get("params") or {}),
        cache=cache,
    )


def _parse_config(raw: Any) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ConfigurationError("Experiment configuration must be a YAML mapping")

    experiment_section = _require_section(raw, "experiment")
    name = _required(experiment_section, "name", "experiment.name")

    dataset_section = _require_section(raw, "dataset")
    dataset = DatasetConfig(
        adapter=_required(dataset_section, "adapter", "dataset.adapter"),
        params=dict(dataset_section.get("params") or {}),
    )

    chunking_section = _require_section(raw, "chunking")
    chunking = _strategy_config(chunking_section, "chunking")

    embedding_section = raw.get("embedding")
    embedding = _embedding_config(embedding_section) if embedding_section else None

    indexing_section = raw.get("indexing") or {}
    dense_index_section = indexing_section.get("dense")
    sparse_index_section = indexing_section.get("sparse")
    indexing = IndexingConfig(
        dense=_strategy_config(dense_index_section, "indexing.dense") if dense_index_section else None,
        sparse=_strategy_config(sparse_index_section, "indexing.sparse") if sparse_index_section else None,
    )

    retrieval_section = _require_section(raw, "retrieval")
    top_k = retrieval_section.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ConfigurationError("'retrieval.top_k' must be a positive integer")
    retrieval = RetrievalConfig(
        strategy=_required(retrieval_section, "strategy", "retrieval.strategy"),
        top_k=top_k,
        params=dict(retrieval_section.get("params") or {}),
    )

    reranking_section = _require_section(raw, "reranking")
    final_k = reranking_section.get("final_k")
    if not isinstance(final_k, int) or isinstance(final_k, bool) or final_k <= 0:
        raise ConfigurationError("'reranking.final_k' must be a positive integer")
    reranking = RerankingConfig(
        strategy=_required(reranking_section, "strategy", "reranking.strategy"),
        final_k=final_k,
        params=dict(reranking_section.get("params") or {}),
    )

    evaluation_section = raw.get("evaluation") or {}
    evaluation = EvaluationConfig(k_values=list(evaluation_section.get("k_values") or [1, 5, 10]))

    output_section = raw.get("output") or {}
    output = OutputConfig(directory=output_section.get("directory") or "results")

    return ExperimentConfig(
        name=name,
        dataset=dataset,
        chunking=chunking,
        embedding=embedding,
        indexing=indexing,
        retrieval=retrieval,
        reranking=reranking,
        evaluation=evaluation,
        output=output,
        raw=raw,
    )
