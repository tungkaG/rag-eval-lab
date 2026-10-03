"""Unit tests for YAML experiment configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError
from rag_eval_lab.experiments.config import load_config

MINIMAL_VALID = """
experiment:
  name: test_experiment

dataset:
  adapter: beir
  params:
    dataset_dir: data/scifact
    split: test

chunking:
  strategy: fixed_tokens
  params:
    chunk_size: 256
    overlap: 32

retrieval:
  strategy: bm25
  top_k: 50
  params: {}

reranking:
  strategy: none
  final_k: 10
  params: {}
"""


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_minimal_valid_configuration(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, MINIMAL_VALID))

    assert config.name == "test_experiment"
    assert config.dataset.adapter == "beir"
    assert config.dataset.params == {"dataset_dir": "data/scifact", "split": "test"}
    assert config.chunking.strategy == "fixed_tokens"
    assert config.chunking.params == {"chunk_size": 256, "overlap": 32}
    assert config.retrieval.strategy == "bm25"
    assert config.retrieval.top_k == 50
    assert config.reranking.strategy == "none"
    assert config.reranking.final_k == 10
    assert config.evaluation.k_values == [1, 5, 10]
    assert config.output.directory == "results"
    assert config.embedding is None
    assert config.indexing.dense is None
    assert config.indexing.sparse is None


def test_parses_full_configuration_with_embedding_and_indexing(tmp_path: Path) -> None:
    content = (
        MINIMAL_VALID
        + """
embedding:
  strategy: sentence_transformer
  params:
    model_name: BAAI/bge-base-en-v1.5

indexing:
  dense:
    strategy: faiss
    params: {}
  sparse:
    strategy: bm25
    params: {}

evaluation:
  k_values: [1, 3]

output:
  directory: my_results
"""
    )
    config = load_config(_write(tmp_path, content))

    assert config.embedding is not None
    assert config.embedding.strategy == "sentence_transformer"
    assert config.embedding.params == {"model_name": "BAAI/bge-base-en-v1.5"}
    assert config.embedding.cache.enabled is False
    assert config.embedding.cache.directory == ".cache/embeddings"
    assert config.embedding.cache.force_recompute is False
    assert config.indexing.dense is not None
    assert config.indexing.dense.strategy == "faiss"
    assert config.indexing.sparse is not None
    assert config.indexing.sparse.strategy == "bm25"
    assert config.evaluation.k_values == [1, 3]
    assert config.output.directory == "my_results"
    assert config.raw["experiment"]["name"] == "test_experiment"


def test_parses_explicit_embedding_cache_settings(tmp_path: Path) -> None:
    content = (
        MINIMAL_VALID
        + """
embedding:
  strategy: sentence_transformer
  params:
    model_name: BAAI/bge-base-en-v1.5
  cache:
    enabled: true
    directory: .cache/embeddings
    force_recompute: true
"""
    )
    config = load_config(_write(tmp_path, content))

    # Cache settings stay out of params -- they must never reach the embedding constructor.
    assert config.embedding.params == {"model_name": "BAAI/bge-base-en-v1.5"}
    assert config.embedding.cache.enabled is True
    assert config.embedding.cache.directory == ".cache/embeddings"
    assert config.embedding.cache.force_recompute is True


def test_missing_dataset_section_raises_configuration_error(tmp_path: Path) -> None:
    content = """
experiment:
  name: test_experiment

chunking:
  strategy: fixed_tokens
  params: {}

retrieval:
  strategy: bm25
  top_k: 50
  params: {}

reranking:
  strategy: none
  final_k: 10
  params: {}
"""
    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, content))


def test_missing_retrieval_section_raises_configuration_error(tmp_path: Path) -> None:
    content = """
experiment:
  name: test_experiment

dataset:
  adapter: beir
  params: {}

chunking:
  strategy: fixed_tokens
  params: {}

reranking:
  strategy: none
  final_k: 10
  params: {}
"""
    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, content))


def test_missing_reranking_section_raises_configuration_error(tmp_path: Path) -> None:
    content = """
experiment:
  name: test_experiment

dataset:
  adapter: beir
  params: {}

chunking:
  strategy: fixed_tokens
  params: {}

retrieval:
  strategy: bm25
  top_k: 50
  params: {}
"""
    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, content))


def test_missing_experiment_name_raises_configuration_error(tmp_path: Path) -> None:
    content = MINIMAL_VALID.replace("name: test_experiment", "name:")

    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, content))


def test_non_positive_top_k_raises_configuration_error(tmp_path: Path) -> None:
    content = MINIMAL_VALID.replace("top_k: 50", "top_k: 0")

    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, content))


def test_non_positive_final_k_raises_configuration_error(tmp_path: Path) -> None:
    content = MINIMAL_VALID.replace("final_k: 10", "final_k: -1")

    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, content))


def test_non_mapping_configuration_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))
