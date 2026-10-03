import json

import pytest

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.core.models import PageReference
from rag_eval_lab.datasets.custom_dataset import CustomDatasetAdapter


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_loads_documents_and_full_ground_truth(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"id": "manual_1", "text": "operates up to 85C"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(
        evaluation_path,
        [
            {
                "query_id": "q1",
                "question": "What is the maximum operating temperature?",
                "ground_truth": {
                    "document_relevance": {"manual_1": 1},
                    "page_references": [{"document_id": "manual_1", "page_start": 23, "page_end": 23}],
                    "reference_answer": "85 degrees Celsius",
                    "evidence_texts": [],
                    "keywords": [],
                    "metadata": {},
                },
            }
        ],
    )

    adapter = CustomDatasetAdapter(corpus_path=corpus_path, evaluation_path=evaluation_path)

    documents = adapter.documents()
    assert [doc.id for doc in documents] == ["manual_1"]

    cases = adapter.evaluation_cases()
    assert len(cases) == 1
    case = cases[0]
    assert case.query_id == "q1"
    assert case.question == "What is the maximum operating temperature?"
    assert case.ground_truth.document_relevance == {"manual_1": 1}
    assert case.ground_truth.reference_answer == "85 degrees Celsius"


def test_page_reference_dicts_are_converted_to_page_reference_objects(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"id": "manual_1", "text": "body"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(
        evaluation_path,
        [
            {
                "query_id": "q1",
                "question": "q?",
                "ground_truth": {
                    "page_references": [{"document_id": "manual_1", "page_start": 1, "page_end": 2}],
                },
            }
        ],
    )

    adapter = CustomDatasetAdapter(corpus_path=corpus_path, evaluation_path=evaluation_path)
    case = adapter.evaluation_cases()[0]

    assert case.ground_truth.page_references == [PageReference(document_id="manual_1", page_start=1, page_end=2)]


def test_only_query_id_and_question_are_required(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"id": "manual_1", "text": "body"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(evaluation_path, [{"query_id": "q1", "question": "q?"}])

    adapter = CustomDatasetAdapter(corpus_path=corpus_path, evaluation_path=evaluation_path)
    case = adapter.evaluation_cases()[0]

    assert case.ground_truth.document_relevance == {}
    assert case.ground_truth.page_references == []
    assert case.ground_truth.reference_answer is None
    assert case.ground_truth.capabilities == set()


def test_capabilities_reflect_loaded_ground_truth(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"id": "manual_1", "text": "body"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(
        evaluation_path,
        [
            {
                "query_id": "q1",
                "question": "q?",
                "ground_truth": {
                    "document_relevance": {"manual_1": 1},
                    "reference_answer": "85 degrees Celsius",
                },
            }
        ],
    )

    adapter = CustomDatasetAdapter(corpus_path=corpus_path, evaluation_path=evaluation_path)
    case = adapter.evaluation_cases()[0]

    assert case.ground_truth.capabilities == {"document_relevance", "reference_answer"}


def test_blank_reference_answer_is_normalized_to_none(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"id": "manual_1", "text": "body"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(
        evaluation_path,
        [{"query_id": "q1", "question": "q?", "ground_truth": {"reference_answer": "   "}}],
    )

    adapter = CustomDatasetAdapter(corpus_path=corpus_path, evaluation_path=evaluation_path)
    case = adapter.evaluation_cases()[0]

    assert case.ground_truth.reference_answer is None


def test_unknown_referenced_document_id_raises_validation_error(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"id": "manual_1", "text": "body"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(
        evaluation_path,
        [
            {
                "query_id": "q1",
                "question": "q?",
                "ground_truth": {"document_relevance": {"manual_unknown": 1}},
            }
        ],
    )

    adapter = CustomDatasetAdapter(corpus_path=corpus_path, evaluation_path=evaluation_path)

    with pytest.raises(ValidationError):
        adapter.evaluation_cases()


def test_custom_corpus_fields_are_passed_to_jsonl_loader(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [{"_id": "manual_1", "text": "body", "title": "Manual"}])

    evaluation_path = tmp_path / "evaluation.jsonl"
    _write_jsonl(evaluation_path, [{"query_id": "q1", "question": "q?"}])

    adapter = CustomDatasetAdapter(
        corpus_path=corpus_path,
        evaluation_path=evaluation_path,
        corpus_fields={"id": "_id", "text": "text", "title": "title"},
    )

    documents = adapter.documents()
    assert documents[0].id == "manual_1"
    assert documents[0].metadata["title"] == "Manual"
