"""Interface for the file processing placeholder layer.

See `file_processing/README.md` for the intended future scope and the rule
that file processors must not create `Document` objects directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileProcessor(Protocol):
    """Converts a source file into a normalized format (e.g. Markdown, JSONL)."""

    def process(self, input_path: str | Path, output_path: str | Path) -> Path: ...
