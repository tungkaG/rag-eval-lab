from rag_eval_lab.ingestion.txt_loader import TxtLoader


def test_loads_single_txt_file(tmp_path):
    file_path = tmp_path / "document1.txt"
    file_path.write_text("hello world", encoding="utf-8")

    documents = TxtLoader().load(file_path)

    assert len(documents) == 1
    assert documents[0].id == "document1.txt"
    assert documents[0].text == "hello world"
    assert documents[0].metadata["source_path"] == str(file_path)


def test_loads_directory_with_deterministic_ids_and_sorted_order(tmp_path):
    (tmp_path / "document2.txt").write_text("second", encoding="utf-8")
    (tmp_path / "document1.txt").write_text("first", encoding="utf-8")
    (tmp_path / "document3.txt").write_text("third", encoding="utf-8")

    documents = TxtLoader().load(tmp_path)

    assert [doc.id for doc in documents] == ["document1.txt", "document2.txt", "document3.txt"]
    assert [doc.text for doc in documents] == ["first", "second", "third"]


def test_ignores_non_txt_files_in_directory(tmp_path):
    (tmp_path / "document1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")

    documents = TxtLoader().load(tmp_path)

    assert [doc.id for doc in documents] == ["document1.txt"]


def test_preserves_content_without_stripping(tmp_path):
    file_path = tmp_path / "document1.txt"
    file_path.write_text("  leading and trailing whitespace  \n\n", encoding="utf-8")

    documents = TxtLoader().load(file_path)

    assert documents[0].text == "  leading and trailing whitespace  \n\n"
