from rag_eval_lab.ingestion.markdown_loader import MarkdownLoader


def test_loads_single_markdown_file(tmp_path):
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Heading\n\nSome **bold** text.", encoding="utf-8")

    documents = MarkdownLoader().load(file_path)

    assert len(documents) == 1
    assert documents[0].id == "doc.md"
    assert documents[0].metadata["source_path"] == str(file_path)


def test_preserves_raw_markdown_content_without_parsing(tmp_path):
    raw = "# Heading\n\n- item one\n- item two\n\n```python\nprint('x')\n```\n"
    file_path = tmp_path / "doc.md"
    file_path.write_text(raw, encoding="utf-8")

    documents = MarkdownLoader().load(file_path)

    assert documents[0].text == raw


def test_loads_directory_with_deterministic_ids_and_sorted_order(tmp_path):
    (tmp_path / "b.md").write_text("second", encoding="utf-8")
    (tmp_path / "a.md").write_text("first", encoding="utf-8")

    documents = MarkdownLoader().load(tmp_path)

    assert [doc.id for doc in documents] == ["a.md", "b.md"]
