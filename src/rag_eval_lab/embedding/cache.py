"""Optional on-disk cache for corpus embeddings, keyed by a deterministic SHA256 fingerprint.

Caches one NumPy `.npy` matrix per fingerprint of (embedding strategy,
constructor params, ordered chunk ids/texts, cache format version), so
different configurations coexist as separate files. Only ever used for
whole-corpus document embeddings -- never for queries, chunks, or indexes.

A relative `directory` (e.g. the default `.cache/embeddings`) always
resolves under the repository root via `resolve_under_repo_root`, regardless
of the current working directory a command was launched from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_eval_lab.core.models import Chunk
from rag_eval_lab.core.paths import resolve_under_repo_root

CACHE_FORMAT_VERSION = 1


def compute_cache_key(strategy: str, params: dict[str, Any], chunks: list[Chunk]) -> str:
    """Derives a deterministic SHA256 cache key for one corpus embedding run."""
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "strategy": strategy,
        "params": params,
        "chunk_ids": [chunk.id for chunk in chunks],
        "chunk_texts": [chunk.text for chunk in chunks],
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cached_embeddings(directory: str | Path, cache_key: str, expected_rows: int) -> list[list[float]] | None:
    """Loads a cached embedding matrix, or `None` on a cache miss or any corruption/mismatch."""
    path = resolve_under_repo_root(directory) / f"{cache_key}.npy"
    if not path.exists():
        return None
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or not np.isfinite(matrix).all():
        return None
    return matrix.tolist()


def save_cached_embeddings(directory: str | Path, cache_key: str, embeddings: list[list[float]]) -> None:
    """Saves an embedding matrix as `<directory>/<cache_key>.npy`, replacing any existing file."""
    directory_path = resolve_under_repo_root(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    np.save(directory_path / f"{cache_key}.npy", np.asarray(embeddings, dtype=np.float64))
