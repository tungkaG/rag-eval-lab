import json

from rag_eval_lab.datasets.squad_adapter import SquadDatasetAdapter, _paragraph_id

_PAYLOAD = {
    "version": "1.1",
    "data": [
        {
            "title": "Article One",
            "paragraphs": [
                {
                    "context": "Paragraph one text about Paris.",
                    "qas": [
                        {
                            "id": "q1",
                            "question": "Where?",
                            "answers": [
                                {"text": "Paris", "answer_start": 10},
                                {"text": "Paris", "answer_start": 10},
                                {"text": "France", "answer_start": 0},
                            ],
                        }
                    ],
                },
                {
                    "context": "Paragraph two text about Everest.",
                    "qas": [{"id": "q2", "question": "How tall?", "answers": [{"text": "8849 meters", "answer_start": 5}]}],
                },
            ],
        },
        {
            # Same title as above, and the first paragraph here has the SAME
            # content as "Article One"'s first paragraph -> must dedup.
            "title": "Article One",
            "paragraphs": [
                {
                    "context": "Paragraph one text about Paris.",
                    "qas": [{"id": "q3", "question": "Another question?", "answers": [{"text": "Paris", "answer_start": 10}]}],
                },
                {
                    # Same title, but DIFFERENT content -> must get a different document id.
                    "context": "Different content under Article One this time.",
                    "qas": [{"id": "q4", "question": "Different para?", "answers": [{"text": "Something", "answer_start": 0}]}],
                },
            ],
        },
    ],
}


def _write_dataset(tmp_path):
    path = tmp_path / "dev-v1.1.json"
    path.write_text(json.dumps(_PAYLOAD), encoding="utf-8")
    return path


def test_documents_and_parent_paragraph_relevance_judgment(tmp_path):
    adapter = SquadDatasetAdapter(_write_dataset(tmp_path))

    cases = {case.query_id: case for case in adapter.evaluation_cases()}
    case1 = cases["q1"]

    paragraph1_id = _paragraph_id("Article One", "Paragraph one text about Paris.")
    assert case1.question == "Where?"
    assert case1.ground_truth.document_relevance == {paragraph1_id: 1}
    assert case1.ground_truth.evidence_texts == []  # never populated from answer spans


def test_first_answer_is_reference_answer_rest_are_metadata_alternatives(tmp_path):
    adapter = SquadDatasetAdapter(_write_dataset(tmp_path))
    case1 = next(case for case in adapter.evaluation_cases() if case.query_id == "q1")

    assert case1.ground_truth.reference_answer == "Paris"
    assert case1.ground_truth.metadata["reference_answer_alternatives"] == ["Paris", "France"]


def test_stable_ids_and_deduplication_across_articles(tmp_path):
    adapter = SquadDatasetAdapter(_write_dataset(tmp_path))

    documents = adapter.documents()
    ids = [document.id for document in documents]

    assert len(ids) == len(set(ids))
    # paragraph1 (shared) + Everest paragraph + "Different content..." paragraph = 3 unique.
    assert len(documents) == 3

    paragraph1_id = _paragraph_id("Article One", "Paragraph one text about Paris.")
    matching = [document for document in documents if document.id == paragraph1_id]
    assert len(matching) == 1
    assert matching[0].metadata["title"] == "Article One"
    assert matching[0].text == "Paragraph one text about Paris."
    assert "text" not in matching[0].metadata and "answers" not in matching[0].metadata


def test_same_title_different_context_gets_different_document_id(tmp_path):
    adapter = SquadDatasetAdapter(_write_dataset(tmp_path))

    documents_by_id = {document.id: document for document in adapter.documents()}
    v1_id = _paragraph_id("Article One", "Paragraph one text about Paris.")
    v2_id = _paragraph_id("Article One", "Different content under Article One this time.")

    assert v1_id != v2_id
    assert documents_by_id[v2_id].text == "Different content under Article One this time."


def test_limit_trims_questions_in_file_order_without_shrinking_the_corpus(tmp_path):
    adapter = SquadDatasetAdapter(_write_dataset(tmp_path), limit=2)

    cases = adapter.evaluation_cases()
    assert [case.query_id for case in cases] == ["q1", "q2"]

    # Corpus is independent of the question limit -- the paragraph only
    # reachable via q4 (3rd/4th questions, excluded by limit) must still be present.
    titles = {document.text for document in adapter.documents()}
    assert "Different content under Article One this time." in titles
    assert len(adapter.documents()) == 3


def test_caches_parsed_results(tmp_path):
    adapter = SquadDatasetAdapter(_write_dataset(tmp_path))

    assert adapter.documents() is adapter.documents()
    assert adapter.evaluation_cases() is adapter.evaluation_cases()
