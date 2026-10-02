from pathlib import Path

import pytest

from rag_eval_lab.datasets.registry import dataset_registry

DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "datasets" / "scifact"
)


def test_real_scifact_loading():
    required_files = [
        DATASET_DIR / "corpus.jsonl",
        DATASET_DIR / "queries.jsonl",
        DATASET_DIR / "qrels" / "test.tsv",
    ]
    if not all(path.exists() for path in required_files):
        pytest.skip("Local SciFact dataset is not available")

    dataset = dataset_registry.create(
        "beir",
        dataset_dir=DATASET_DIR,
        split="test",
    )

    documents = dataset.documents()
    cases = dataset.evaluation_cases()

    assert len(documents) == 5183
    assert len(cases) == 300

    document_ids = {doc.id for doc in documents}

    for case in cases:
        assert case.question.strip()
        assert case.ground_truth.document_relevance
        assert case.ground_truth.capabilities == {"document_relevance"}
        assert set(case.ground_truth.document_relevance) <= document_ids

    cases_by_id = {case.query_id: case for case in cases}
    assert cases_by_id["1"].ground_truth.document_relevance == {
        "31715818": 1
    }