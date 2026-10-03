"""Unit tests for `embedding.cache`: key derivation and `.npy` load/save semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_eval_lab.core import paths as paths_module
from rag_eval_lab.core.models import Chunk
from rag_eval_lab.embedding.cache import compute_cache_key, load_cached_embeddings, save_cached_embeddings

_CHUNKS = [
    Chunk(id="c1", document_id="doc1", text="hello world"),
    Chunk(id="c2", document_id="doc2", text="goodbye world"),
]


def test_compute_cache_key_is_deterministic() -> None:
    key_a = compute_cache_key("sentence_transformer", {"model_name": "m"}, _CHUNKS)
    key_b = compute_cache_key("sentence_transformer", {"model_name": "m"}, _CHUNKS)

    assert key_a == key_b


def test_different_chunk_text_changes_key() -> None:
    changed = [Chunk(id="c1", document_id="doc1", text="different text"), _CHUNKS[1]]

    assert compute_cache_key("sentence_transformer", {}, _CHUNKS) != compute_cache_key(
        "sentence_transformer", {}, changed
    )


def test_different_chunk_order_changes_key() -> None:
    reordered = list(reversed(_CHUNKS))

    assert compute_cache_key("sentence_transformer", {}, _CHUNKS) != compute_cache_key(
        "sentence_transformer", {}, reordered
    )


def test_different_strategy_changes_key() -> None:
    assert compute_cache_key("sentence_transformer", {}, _CHUNKS) != compute_cache_key("other_strategy", {}, _CHUNKS)


def test_different_params_changes_key() -> None:
    assert compute_cache_key("sentence_transformer", {"model_name": "a"}, _CHUNKS) != compute_cache_key(
        "sentence_transformer", {"model_name": "b"}, _CHUNKS
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    key = compute_cache_key("sentence_transformer", {}, _CHUNKS)

    save_cached_embeddings(tmp_path, key, embeddings)
    loaded = load_cached_embeddings(tmp_path, key, expected_rows=2)

    assert loaded == embeddings


def test_different_configurations_coexist_as_separate_files(tmp_path: Path) -> None:
    key_a = compute_cache_key("sentence_transformer", {"model_name": "a"}, _CHUNKS)
    key_b = compute_cache_key("sentence_transformer", {"model_name": "b"}, _CHUNKS)

    save_cached_embeddings(tmp_path, key_a, [[1.0], [2.0]])
    save_cached_embeddings(tmp_path, key_b, [[3.0], [4.0]])

    assert load_cached_embeddings(tmp_path, key_a, expected_rows=2) == [[1.0], [2.0]]
    assert load_cached_embeddings(tmp_path, key_b, expected_rows=2) == [[3.0], [4.0]]


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_cached_embeddings(tmp_path, "missing-key", expected_rows=2) is None


def test_load_returns_none_for_wrong_row_count(tmp_path: Path) -> None:
    key = "wrong-row-count"
    save_cached_embeddings(tmp_path, key, [[0.1, 0.2], [0.3, 0.4]])

    assert load_cached_embeddings(tmp_path, key, expected_rows=3) is None


def test_load_returns_none_for_non_finite_values(tmp_path: Path) -> None:
    key = "non-finite"
    np.save(tmp_path / f"{key}.npy", np.array([[1.0, float("nan")], [2.0, 3.0]]))

    assert load_cached_embeddings(tmp_path, key, expected_rows=2) is None


def test_load_returns_none_for_wrong_dimensionality(tmp_path: Path) -> None:
    key = "one-dimensional"
    np.save(tmp_path / f"{key}.npy", np.array([1.0, 2.0, 3.0]))

    assert load_cached_embeddings(tmp_path, key, expected_rows=3) is None


def test_save_creates_cache_directory_if_missing(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "cache"
    key = compute_cache_key("sentence_transformer", {}, _CHUNKS)

    save_cached_embeddings(directory, key, [[1.0], [2.0]])

    assert (directory / f"{key}.npy").exists()


def test_relative_cache_directory_resolves_under_repo_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo_root = tmp_path / "fake_repo"
    fake_repo_root.mkdir()
    monkeypatch.setattr(paths_module, "REPO_ROOT", fake_repo_root)

    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    key = compute_cache_key("sentence_transformer", {}, _CHUNKS)
    save_cached_embeddings(".cache/embeddings", key, [[1.0], [2.0]])

    assert (fake_repo_root / ".cache" / "embeddings" / f"{key}.npy").exists()
    assert not (other_cwd / ".cache").exists()
    assert load_cached_embeddings(".cache/embeddings", key, expected_rows=2) == [[1.0], [2.0]]
