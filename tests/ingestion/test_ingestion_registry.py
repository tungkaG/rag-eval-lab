from rag_eval_lab.ingestion.jsonl_loader import JsonlLoader
from rag_eval_lab.ingestion.markdown_loader import MarkdownLoader
from rag_eval_lab.ingestion.registry import ingestion_registry
from rag_eval_lab.ingestion.txt_loader import TxtLoader


def test_registered_loader_names():
    assert ingestion_registry.keys() == ["jsonl", "markdown", "txt"]


def test_creates_txt_loader():
    assert isinstance(ingestion_registry.create("txt"), TxtLoader)


def test_creates_markdown_loader():
    assert isinstance(ingestion_registry.create("markdown"), MarkdownLoader)


def test_creates_jsonl_loader_with_default_fields():
    loader = ingestion_registry.create("jsonl")
    assert isinstance(loader, JsonlLoader)
    assert loader.fields == {"id": "id", "text": "text"}


def test_creates_jsonl_loader_with_custom_fields():
    loader = ingestion_registry.create("jsonl", fields={"id": "_id", "text": "text", "title": "title"})
    assert loader.fields == {"id": "_id", "text": "text", "title": "title"}
