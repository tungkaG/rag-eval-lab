"""Adapter for the official SQuAD 1.1 JSON distribution (`data -> articles -> paragraphs -> qas`).

Builds ONE shared corpus from ALL paragraphs in the file, independent of any
question `limit` -- so even a small generation test still retrieves against
the full set of distractor paragraphs. Source paragraphs provide POSITIVE
relevance labels only: a paragraph containing a question's answer is judged
relevant, but absence from `document_relevance` is not an exhaustive judgment
that every other paragraph was checked and found irrelevant. These
experiments are therefore not directly comparable to the official SQuAD
reading comprehension leaderboard (which scores span extraction given the
gold paragraph, not corpus retrieval).

All SQuAD-specific file/record knowledge stays in this module; no other
component (including the runner) knows SQuAD exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_eval_lab.core.models import Document, EvaluationCase, GroundTruth
from rag_eval_lab.ground_truth.validator import normalize_ground_truth


def _paragraph_id(title: str, context: str) -> str:
    """A deterministic id from a paragraph's exact article title and context text.

    Two paragraphs are the same document only when both the title AND the
    context text match exactly.
    """
    payload = json.dumps({"title": title, "context": context}, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"squad_{digest}"


class SquadDatasetAdapter:
    """Reads the official SQuAD 1.1 JSON distribution (a top-level `data` array of articles).

    The shared corpus always contains every paragraph in the file; `limit`
    (first N questions in file order) only trims `evaluation_cases()`.
    """

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

        with self.dataset_path.open(encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)

        documents: list[Document] = []
        seen_ids: set[str] = set()
        cases: list[EvaluationCase] = []

        for article in payload.get("data", []):
            title = article.get("title", "")
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                document_id = _paragraph_id(title, context)
                if document_id not in seen_ids:
                    seen_ids.add(document_id)
                    documents.append(Document(id=document_id, text=context, metadata={"title": title}))

                for qa in paragraph.get("qas", []):
                    # Deliberately NOT populated from answer spans: an answer
                    # span is not necessarily sufficient supporting evidence.
                    answer_texts = [answer["text"] for answer in qa.get("answers", []) if answer.get("text")]
                    # Only the FIRST annotated answer becomes reference_answer
                    # (the current answer evaluators use one reference answer
                    # only); the rest are kept as ground truth metadata, never
                    # concatenated into reference_answer.
                    reference_answer = answer_texts[0] if answer_texts else None
                    alternatives = list(dict.fromkeys(answer_texts))

                    ground_truth = GroundTruth(
                        document_relevance={document_id: 1},
                        reference_answer=reference_answer,
                        metadata={"reference_answer_alternatives": alternatives},
                    )
                    normalize_ground_truth(ground_truth)
                    cases.append(
                        EvaluationCase(query_id=qa["id"], question=qa["question"], ground_truth=ground_truth)
                    )

        if self.limit is not None:
            cases = cases[: self.limit]

        self._documents = documents
        self._evaluation_cases = cases
