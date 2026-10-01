import json

from rag_eval_lab.ingestion.jsonl_loader import JsonlLoader


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_loads_with_default_field_mapping(tmp_path):
    file_path = tmp_path / "corpus.jsonl"
    _write_jsonl(file_path, [{"id": "doc1", "text": "hello world"}])

    documents = JsonlLoader().load(file_path)

    assert len(documents) == 1
    assert documents[0].id == "doc1"
    assert documents[0].text == "hello world"


def test_loads_with_custom_field_mapping(tmp_path):
    file_path = tmp_path / "corpus.jsonl"
    _write_jsonl(file_path, [{"_id": "4983", "text": "custom mapped text"}])

    documents = JsonlLoader(fields={"id": "_id", "text": "text"}).load(file_path)

    assert documents[0].id == "4983"
    assert documents[0].text == "custom mapped text"


def test_loads_multiple_records_in_file_order(tmp_path):
    file_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        file_path,
        [
            {"id": "doc1", "text": "first"},
            {"id": "doc2", "text": "second"},
        ],
    )

    documents = JsonlLoader().load(file_path)

    assert [doc.id for doc in documents] == ["doc1", "doc2"]


def test_preserves_optional_title_in_metadata(tmp_path):
    file_path = tmp_path / "corpus.jsonl"
    _write_jsonl(file_path, [{"_id": "doc1", "text": "body text", "title": "Doc Title"}])

    documents = JsonlLoader(fields={"id": "_id", "text": "text", "title": "title"}).load(file_path)

    assert documents[0].metadata["title"] == "Doc Title"
    assert documents[0].text == "body text"


def test_missing_title_mapping_leaves_metadata_without_title(tmp_path):
    file_path = tmp_path / "corpus.jsonl"
    _write_jsonl(file_path, [{"id": "doc1", "text": "body text"}])

    documents = JsonlLoader().load(file_path)

    assert "title" not in documents[0].metadata


def test_skips_blank_lines(tmp_path):
    file_path = tmp_path / "corpus.jsonl"
    file_path.write_text('{"id": "doc1", "text": "hello"}\n\n', encoding="utf-8")

    documents = JsonlLoader().load(file_path)

    assert len(documents) == 1
