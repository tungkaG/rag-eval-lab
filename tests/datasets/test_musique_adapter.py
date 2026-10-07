import json

from rag_eval_lab.datasets.musique_adapter import MusiqueDatasetAdapter, _paragraph_id

_RECORDS = [
    {
        "id": "case-1",
        "question": "What connects these facts?",
        "answer": "Answer one",
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "Distractor", "paragraph_text": "Distractor only for case one.", "is_supporting": False},
            {"idx": 1, "title": "Shared", "paragraph_text": "Shared supporting paragraph.", "is_supporting": True},
            {"idx": 2, "title": "Second", "paragraph_text": "Second supporting paragraph.", "is_supporting": True},
        ],
    },
    {
        "id": "unanswerable",
        "question": "Should be skipped?",
        "answer": "",
        "answerable": False,
        "paragraphs": [
            {"idx": 0, "title": "Skipped", "paragraph_text": "Never enters the corpus.", "is_supporting": False}
        ],
    },
    {
        "id": "case-2",
        "question": "What shares a paragraph?",
        "answer": "Answer two",
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "Shared", "paragraph_text": "Shared supporting paragraph.", "is_supporting": True},
            {"idx": 1, "title": "Case two", "paragraph_text": "Case two distractor.", "is_supporting": False},
        ],
    },
]


def _write_dataset(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in _RECORDS), encoding="utf-8")
    return path


def test_shared_corpus_contains_supporting_and_distractor_paragraphs(tmp_path) -> None:
    adapter = MusiqueDatasetAdapter(_write_dataset(tmp_path))

    documents = adapter.documents()

    assert {document.text for document in documents} == {
        "Distractor only for case one.",
        "Shared supporting paragraph.",
        "Second supporting paragraph.",
        "Case two distractor.",
    }
    assert len(documents) == 4
    assert all(document.metadata.keys() == {"title"} for document in documents)


def test_cases_map_answers_supporting_documents_and_whole_paragraph_evidence(tmp_path) -> None:
    adapter = MusiqueDatasetAdapter(_write_dataset(tmp_path))
    cases = {case.query_id: case for case in adapter.evaluation_cases()}
    shared_id = _paragraph_id("Shared", "Shared supporting paragraph.")
    second_id = _paragraph_id("Second", "Second supporting paragraph.")

    assert cases["case-1"].ground_truth.reference_answer == "Answer one"
    assert cases["case-1"].ground_truth.document_relevance == {shared_id: 1, second_id: 1}
    assert cases["case-1"].ground_truth.evidence_texts == ["Shared supporting paragraph.", "Second supporting paragraph."]
    assert cases["case-2"].ground_truth.reference_answer == "Answer two"
    assert cases["case-2"].ground_truth.document_relevance == {shared_id: 1}
    assert cases["case-2"].ground_truth.evidence_texts == ["Shared supporting paragraph."]


def test_deduplication_uses_stable_title_and_paragraph_ids(tmp_path) -> None:
    adapter = MusiqueDatasetAdapter(_write_dataset(tmp_path))
    documents = {document.id: document for document in adapter.documents()}
    shared_id = _paragraph_id("Shared", "Shared supporting paragraph.")

    assert len(documents) == 4
    assert documents[shared_id].metadata == {"title": "Shared"}
    assert documents[shared_id].text == "Shared supporting paragraph."


def test_limit_selects_answerable_records_in_file_order_before_building_corpus(tmp_path) -> None:
    adapter = MusiqueDatasetAdapter(_write_dataset(tmp_path), limit=1)

    assert [case.query_id for case in adapter.evaluation_cases()] == ["case-1"]
    assert {document.text for document in adapter.documents()} == {
        "Distractor only for case one.",
        "Shared supporting paragraph.",
        "Second supporting paragraph.",
    }
    assert MusiqueDatasetAdapter(_write_dataset(tmp_path), limit=0).evaluation_cases() == []


def test_adapter_caches_parsed_documents_and_cases(tmp_path) -> None:
    adapter = MusiqueDatasetAdapter(_write_dataset(tmp_path))

    assert adapter.documents() is adapter.documents()
    assert adapter.evaluation_cases() is adapter.evaluation_cases()