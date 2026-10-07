"""Contracts for matching retrieved chunks to gold evidence text."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvidenceMatcher(Protocol):
    """Scores every retrieved text against every gold evidence text."""

    name: str
    metadata: dict[str, Any]

    def score(self, retrieved_texts: list[str], evidence_texts: list[str]) -> list[list[float]]:
        """Returns a row-per-retrieved-text, column-per-evidence score matrix."""
        ...