"""Adapter for local MuSiQue JSONL records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_eval_lab.core.models import Document, EvaluationCase, GroundTruth


def _paragraph_id(title: str, paragraph_text: str) -> str:
    """Returns a stable identifier for an exact title/paragraph pair."""
    payload = json.dumps({"title": title, "paragraph_text": paragraph_text}, ensure_ascii=False)
    return f"musique_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class MusiqueDatasetAdapter:
    """Builds a shared paragraph corpus from selected answerable MuSiQue records."""

    def __init__(self, dataset_path: str | Path, limit: int | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.limit = limit
        self._documents: list[Document] | None = None
        self._evaluation_cases: list[EvaluationCase] | None = None

    def documents(self) -> list[Document]:
        self._parse()
        assert self._documents is not None
        return self._documents

    def evaluation_cases(self) -> list[EvaluationCase]:
        self._parse()
        assert self._evaluation_cases is not None
        return self._evaluation_cases

    def _parse(self) -> None:
        if self._documents is not None and self._evaluation_cases is not None:
            return

        records = self._answerable_records()
        documents: list[Document] = []
        documents_by_key: dict[tuple[str, str], str] = {}
        cases: list[EvaluationCase] = []

        for record in records:
            document_ids_by_paragraph_index: dict[int, str] = {}
            for paragraph in record.get("paragraphs", []):
                title = paragraph.get("title", "")
                paragraph_text = paragraph.get("paragraph_text", "")
                document_key = (title, paragraph_text)
                document_id = documents_by_key.get(document_key)
                if document_id is None:
                    document_id = _paragraph_id(title, paragraph_text)
                    documents_by_key[document_key] = document_id
                    documents.append(Document(id=document_id, text=paragraph_text, metadata={"title": title}))
                document_ids_by_paragraph_index[paragraph.get("idx")] = document_id

            supporting_paragraphs = [paragraph for paragraph in record.get("paragraphs", []) if paragraph.get("is_supporting")]
            document_relevance = {
                document_ids_by_paragraph_index[paragraph["idx"]]: 1
                for paragraph in supporting_paragraphs
            }
            evidence_texts = list(dict.fromkeys(paragraph["paragraph_text"] for paragraph in supporting_paragraphs))
            cases.append(
                EvaluationCase(
                    query_id=record["id"],
                    question=record["question"],
                    ground_truth=GroundTruth(
                        document_relevance=document_relevance,
                        reference_answer=record["answer"],
                        evidence_texts=evidence_texts,
                    ),
                )
            )

        self._documents = documents
        self._evaluation_cases = cases

    def _answerable_records(self) -> list[dict[str, Any]]:
        if self.limit is not None and self.limit <= 0:
            return []

        records: list[dict[str, Any]] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("answerable"):
                    records.append(record)
                    if self.limit is not None and len(records) >= self.limit:
                        break
        return records