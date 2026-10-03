"""Resolves the repository root so on-disk caches stay stable regardless of cwd.

`REPO_ROOT` is anchored on this installed package's own file location, not
`Path.cwd()`, so a relative cache directory like `.cache/embeddings` always
resolves under the repository root no matter where a command is launched
from. This is deliberately different from dataset/output paths in
`experiments/config.py`, which are documented to resolve relative to the
current working directory instead.
"""

from __future__ import annotations

from pathlib import Path

# src/rag_eval_lab/core/paths.py -> core -> rag_eval_lab -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_under_repo_root(path: str | Path) -> Path:
    """Returns `path` unchanged if absolute, otherwise resolved under `REPO_ROOT`."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
