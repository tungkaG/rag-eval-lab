from rag_eval_lab.datasets.beir_adapter import BeirDatasetAdapter
from rag_eval_lab.datasets.custom_dataset import CustomDatasetAdapter
from rag_eval_lab.datasets.musique_adapter import MusiqueDatasetAdapter
from rag_eval_lab.datasets.registry import dataset_registry
from rag_eval_lab.datasets.squad_adapter import SquadDatasetAdapter


def test_registered_dataset_adapter_names():
    assert dataset_registry.keys() == ["beir", "custom", "musique", "squad"]


def test_creates_custom_adapter_with_constructor_arguments(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text('{"id": "d1", "text": "hello"}\n', encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_path.write_text('{"query_id": "q1", "question": "q?"}\n', encoding="utf-8")

    adapter = dataset_registry.create("custom", corpus_path=corpus_path, evaluation_path=evaluation_path)

    assert isinstance(adapter, CustomDatasetAdapter)
    assert adapter.documents()[0].id == "d1"


def test_creates_beir_adapter_with_constructor_arguments(tmp_path):
    adapter = dataset_registry.create("beir", dataset_dir=tmp_path, split="train")

    assert isinstance(adapter, BeirDatasetAdapter)
    assert adapter.split == "train"


def test_creates_squad_adapter_with_constructor_arguments(tmp_path):
    dataset_path = tmp_path / "dev-v1.1.json"
    dataset_path.write_text('{"version": "1.1", "data": []}', encoding="utf-8")

    adapter = dataset_registry.create("squad", dataset_path=dataset_path, limit=10)

    assert isinstance(adapter, SquadDatasetAdapter)
    assert adapter.limit == 10


def test_creates_musique_adapter_with_constructor_arguments(tmp_path):
    dataset_path = tmp_path / "validation.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    adapter = dataset_registry.create("musique", dataset_path=dataset_path, limit=10)

    assert isinstance(adapter, MusiqueDatasetAdapter)
    assert adapter.limit == 10
