"""Converts native evaluation records into core ground truth models.

This is the only place that understands this project's native JSON record
shape; it maps directly onto the existing `GroundTruth` / `PageReference` /
`EvaluationCase` models without introducing parallel ones.
"""

from __future__ import annotations

from typing import Any

from rag_eval_lab.core.models import EvaluationCase, GroundTruth, PageReference


def ground_truth_from_record(record: dict[str, Any]) -> GroundTruth:
    """Builds a `GroundTruth` from a native `ground_truth` record dict.

    Every key is optional; missing keys fall back to the `GroundTruth` field
    defaults.
    """
    page_references = [
        PageReference(
            document_id=page["document_id"],
            page_start=page["page_start"],
            page_end=page["page_end"],
        )
        for page in record.get("page_references", [])
    ]
    return GroundTruth(
        document_relevance=dict(record.get("document_relevance", {})),
        page_references=page_references,
        reference_answer=record.get("reference_answer"),
        evidence_texts=list(record.get("evidence_texts", [])),
        keywords=list(record.get("keywords", [])),
        metadata=dict(record.get("metadata", {})),
    )


def evaluation_case_from_record(record: dict[str, Any]) -> EvaluationCase:
    """Builds an `EvaluationCase` from a native evaluation JSONL record.

    Only `query_id` and `question` are required; `ground_truth` is optional.
    """
    ground_truth = ground_truth_from_record(record.get("ground_truth", {}))
    return EvaluationCase(
        query_id=record["query_id"],
        question=record["question"],
        ground_truth=ground_truth,
    )
