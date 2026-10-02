from pathlib import Path

import pytest

from rag_eval_lab.chunking.registry import chunking_registry
from rag_eval_lab.core.models import Chunk
from rag_eval_lab.datasets.registry import dataset_registry

DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "datasets" / "scifact"
)


@pytest.fixture(scope="module")
def scifact_documents():
    if not (DATASET_DIR / "corpus.jsonl").exists():
        pytest.skip("Local SciFact corpus is not available")

    dataset = dataset_registry.create(
        "beir",
        dataset_dir=DATASET_DIR,
        split="test",
    )
    return dataset.documents()


@pytest.mark.parametrize("strategy", ["fixed_tokens", "recursive"])
def test_chunking_real_scifact(scifact_documents, strategy):
    documents = scifact_documents
    assert len(documents) == 5183

    chunker = chunking_registry.create(
        strategy,
        chunk_size=256,
        overlap=32,
    )

    chunk_ids = set()
    chunk_count = 0

    for document in documents:
        chunks = chunker.chunk(document)
        assert chunks

        for index, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.text.strip()
            assert chunk.document_id == document.id
            assert chunk.id == f"{document.id}_chunk_{index}"
            assert chunk.id not in chunk_ids

            assert chunk.metadata["strategy"] == strategy
            assert chunk.metadata["chunk_size"] == 256
            assert chunk.metadata["overlap"] == 32
            assert chunk.metadata["encoding_name"] == "cl100k_base"

            if "title" in document.metadata:
                assert chunk.metadata["title"] == document.metadata["title"]

            chunk_ids.add(chunk.id)

        chunk_count += len(chunks)

    assert chunk_count > len(documents)

    first_run = chunker.chunk(documents[0])
    second_run = chunker.chunk(documents[0])
    assert first_run == second_run