import json

import pytest

from rag_eval_lab.core.exceptions import ValidationError
from rag_eval_lab.datasets.beir_adapter import BeirDatasetAdapter


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, corpus_id, score in rows:
            handle.write(f"{query_id}\t{corpus_id}\t{score}\n")


def _make_dataset(tmp_path):
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"_id": "doc1", "title": "Doc One", "text": "first document body"},
            {"_id": "doc2", "title": "Doc Two", "text": "second document body"},
        ],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"_id": "q1", "text": "What is in doc1?"},
            {"_id": "q2", "text": "What is in doc2?"},
        ],
    )
    _write_tsv(tmp_path / "qrels" / "test.tsv", [("q1", "doc1", 1), ("q1", "doc2", 2)])
    _write_tsv(tmp_path / "qrels" / "train.tsv", [("q2", "doc2", 1)])
    return tmp_path


def test_loads_corpus_with_titles_preserved(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    documents = BeirDatasetAdapter(dataset_dir).documents()

    assert [doc.id for doc in documents] == ["doc1", "doc2"]
    assert documents[0].metadata["title"] == "Doc One"
    assert documents[0].text == "first document body"


def test_defaults_to_test_split(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    cases = BeirDatasetAdapter(dataset_dir).evaluation_cases()

    assert [case.query_id for case in cases] == ["q1"]
    assert cases[0].question == "What is in doc1?"


def test_preserves_graded_relevance_scores(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    cases = BeirDatasetAdapter(dataset_dir).evaluation_cases()

    assert cases[0].ground_truth.document_relevance == {"doc1": 1, "doc2": 2}


def test_selected_split_filters_queries(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    cases = BeirDatasetAdapter(dataset_dir, split="train").evaluation_cases()

    assert [case.query_id for case in cases] == ["q2"]
    assert cases[0].ground_truth.document_relevance == {"doc2": 1}


def test_capabilities_reflect_document_relevance(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    cases = BeirDatasetAdapter(dataset_dir).evaluation_cases()

    assert cases[0].ground_truth.capabilities == {"document_relevance"}


def test_unknown_corpus_id_in_qrels_raises_validation_error(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    _write_tsv(dataset_dir / "qrels" / "test.tsv", [("q1", "doc_missing", 1)])

    with pytest.raises(ValidationError):
        BeirDatasetAdapter(dataset_dir).evaluation_cases()
