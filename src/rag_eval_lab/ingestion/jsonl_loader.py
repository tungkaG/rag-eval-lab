"""Loads JSONL records into `Document`s using configurable field mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_eval_lab.core.models import Document

DEFAULT_FIELDS: dict[str, str] = {"id": "id", "text": "text"}


class JsonlLoader:
    """Loads a JSONL file, one `Document` per record.

    `fields` maps the core `id` / `text` / optional `title` names to the
    record's actual key names, e.g. `{"id": "_id", "text": "text", "title": "title"}`.
    Unmapped names fall back to `DEFAULT_FIELDS`.
    """

    def __init__(self, fields: dict[str, str] | None = None) -> None:
        self.fields = {**DEFAULT_FIELDS, **(fields or {})}

    def load(self, source: str | Path) -> list[Document]:
        path = Path(source)
        documents = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                documents.append(self._to_document(json.loads(line)))
        return documents

    def _to_document(self, record: dict[str, Any]) -> Document:
        doc_id = str(record[self.fields["id"]])
        text = record[self.fields["text"]]
        metadata: dict[str, Any] = {}
        title_field = self.fields.get("title")
        if title_field and title_field in record:
            metadata["title"] = record[title_field]
        return Document(id=doc_id, text=text, metadata=metadata)
