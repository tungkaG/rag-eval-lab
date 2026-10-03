"""Adapter for this project's native custom dataset format."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval_lab.core.models import Document, EvaluationCase
from rag_eval_lab.ground_truth.adapters import evaluation_case_from_record
from rag_eval_lab.ground_truth.validator import check_document_references_exist, normalize_ground_truth
from rag_eval_lab.ingestion.registry import ingestion_registry


class CustomDatasetAdapter:
    """Loads a corpus through `ingestion_registry` and native evaluation JSONL records.

    Each evaluation record is a JSON object with `query_id`, `question`, and
    an optional `ground_truth` object (see `ground_truth.adapters`). Only
    `query_id` and `question` are required; missing ground truth fields fall
    back to the `GroundTruth` model defaults.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        evaluation_path: str | Path,
        corpus_format: str = "jsonl",
        corpus_fields: dict[str, str] | None = None,
    ) -> None:
        self.corpus_path = corpus_path
        self.evaluation_path = evaluation_path
        self.corpus_format = corpus_format
        self.corpus_fields = corpus_fields
        self._documents: list[Document] | None = None
        self._evaluation_cases: list[EvaluationCase] | None = None

    def documents(self) -> list[Document]:
        if self._documents is None:
            kwargs = {"fields": self.corpus_fields} if self.corpus_format == "jsonl" else {}
            loader = ingestion_registry.create(self.corpus_format, **kwargs)
            self._documents = loader.load(self.corpus_path)
        return self._documents

    def evaluation_cases(self) -> list[EvaluationCase]:
        if self._evaluation_cases is None:
            documents = self.documents()
            cases = []
            with Path(self.evaluation_path).open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    case = evaluation_case_from_record(json.loads(line))
                    normalize_ground_truth(case.ground_truth)
                    check_document_references_exist(case.ground_truth, documents)
                    cases.append(case)
            self._evaluation_cases = cases
        return self._evaluation_cases
