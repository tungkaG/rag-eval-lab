"""Unit tests for `core.paths`: resolving relative paths under the repository root."""

from __future__ import annotations

from pathlib import Path

from rag_eval_lab.core.paths import REPO_ROOT, resolve_under_repo_root


def test_repo_root_points_at_the_actual_repository_root() -> None:
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert (REPO_ROOT / "src" / "rag_eval_lab").is_dir()


def test_resolve_under_repo_root_returns_absolute_paths_unchanged(tmp_path: Path) -> None:
    absolute = tmp_path / "somewhere"

    assert resolve_under_repo_root(absolute) == absolute


def test_resolve_under_repo_root_resolves_relative_paths_under_repo_root() -> None:
    assert resolve_under_repo_root(".cache/embeddings") == REPO_ROOT / ".cache" / "embeddings"
