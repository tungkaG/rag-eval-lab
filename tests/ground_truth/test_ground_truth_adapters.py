from rag_eval_lab.core.models import PageReference
from rag_eval_lab.ground_truth.adapters import evaluation_case_from_record, ground_truth_from_record


def test_ground_truth_from_record_defaults_when_empty():
    ground_truth = ground_truth_from_record({})

    assert ground_truth.document_relevance == {}
    assert ground_truth.page_references == []
    assert ground_truth.reference_answer is None
    assert ground_truth.capabilities == set()


def test_ground_truth_from_record_converts_page_references():
    ground_truth = ground_truth_from_record(
        {"page_references": [{"document_id": "manual_1", "page_start": 23, "page_end": 24}]}
    )

    assert ground_truth.page_references == [PageReference(document_id="manual_1", page_start=23, page_end=24)]


def test_ground_truth_from_record_preserves_document_relevance():
    ground_truth = ground_truth_from_record({"document_relevance": {"manual_1": 1, "manual_2": 0}})

    assert ground_truth.document_relevance == {"manual_1": 1, "manual_2": 0}


def test_evaluation_case_from_record_requires_only_query_id_and_question():
    case = evaluation_case_from_record({"query_id": "q1", "question": "What temperature?"})

    assert case.query_id == "q1"
    assert case.question == "What temperature?"
    assert case.ground_truth.document_relevance == {}


def test_evaluation_case_from_record_full_ground_truth():
    case = evaluation_case_from_record(
        {
            "query_id": "q1",
            "question": "What is the maximum operating temperature?",
            "ground_truth": {
                "document_relevance": {"manual_1": 1},
                "page_references": [{"document_id": "manual_1", "page_start": 23, "page_end": 23}],
                "reference_answer": "85 degrees Celsius",
            },
        }
    )

    assert case.ground_truth.document_relevance == {"manual_1": 1}
    assert case.ground_truth.reference_answer == "85 degrees Celsius"
    assert case.ground_truth.capabilities == {"document_relevance", "page_references", "reference_answer"}
