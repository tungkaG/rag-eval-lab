"""Basic, benchmark agnostic ground truth normalization and relation checks."""

from __future__ import annotations

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.core.models import Document, GroundTruth


def normalize_ground_truth(ground_truth: GroundTruth) -> GroundTruth:
    """Normalizes a blank `reference_answer` to `None`, in place."""
    if ground_truth.reference_answer is not None and not ground_truth.reference_answer.strip():
        ground_truth.reference_answer = None
    return ground_truth


def check_document_references_exist(ground_truth: GroundTruth, documents: list[Document]) -> None:
    """Raises `ValidationError` if a referenced document id is absent from `documents`."""
    known_ids = {document.id for document in documents}
    referenced_ids = set(ground_truth.document_relevance) | {
        page.document_id for page in ground_truth.page_references
    }
    missing = referenced_ids - known_ids
    if missing:
        raise ValidationError(f"Ground truth references unknown document ids: {sorted(missing)}")
