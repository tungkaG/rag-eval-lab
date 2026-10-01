"""Loads Markdown files into `Document`s, preserving raw Markdown content."""

from __future__ import annotations

from pathlib import Path

from rag_eval_lab.core.models import Document


class MarkdownLoader:
    """Loads `.md` files from a single file path or a directory path.

    Directories are scanned for `.md` files in sorted filename order, non
    recursively. Each file's name (including extension) becomes the
    deterministic `Document.id`. Markdown content is stored as-is; headings
    and structure are not parsed.
    """

    extension = ".md"

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source)
        paths = sorted(path.glob(f"*{self.extension}")) if path.is_dir() else [path]
        return [self._load_file(p) for p in paths]

    def _load_file(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        return Document(id=path.name, text=text, metadata={"source_path": str(path)})
