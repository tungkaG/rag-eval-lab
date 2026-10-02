import pytest

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.core.models import Document, GroundTruth, PageReference
from rag_eval_lab.ground_truth.validator import check_document_references_exist, normalize_ground_truth


def test_normalize_ground_truth_converts_blank_answer_to_none():
    ground_truth = GroundTruth(reference_answer="   ")

    normalize_ground_truth(ground_truth)

    assert ground_truth.reference_answer is None


def test_normalize_ground_truth_leaves_non_blank_answer_untouched():
    ground_truth = GroundTruth(reference_answer="85 degrees Celsius")

    normalize_ground_truth(ground_truth)

    assert ground_truth.reference_answer == "85 degrees Celsius"


def test_normalize_ground_truth_leaves_none_answer_untouched():
    ground_truth = GroundTruth(reference_answer=None)

    normalize_ground_truth(ground_truth)

    assert ground_truth.reference_answer is None


def test_check_document_references_exist_passes_for_known_ids():
    documents = [Document(id="manual_1", text="body")]
    ground_truth = GroundTruth(
        document_relevance={"manual_1": 1},
        page_references=[PageReference(document_id="manual_1", page_start=1, page_end=1)],
    )

    check_document_references_exist(ground_truth, documents)  # should not raise


def test_check_document_references_exist_raises_for_unknown_relevance_id():
    documents = [Document(id="manual_1", text="body")]
    ground_truth = GroundTruth(document_relevance={"manual_unknown": 1})

    with pytest.raises(ValidationError):
        check_document_references_exist(ground_truth, documents)


def test_check_document_references_exist_raises_for_unknown_page_reference_id():
    documents = [Document(id="manual_1", text="body")]
    ground_truth = GroundTruth(
        page_references=[PageReference(document_id="manual_unknown", page_start=1, page_end=1)]
    )

    with pytest.raises(ValidationError):
        check_document_references_exist(ground_truth, documents)
