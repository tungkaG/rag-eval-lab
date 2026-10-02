from rag_eval_lab.chunking.fixed_tokens import FixedTokenChunker
from rag_eval_lab.chunking.recursive import RecursiveChunker
from rag_eval_lab.chunking.registry import chunking_registry
from rag_eval_lab.core.models import Chunk, Document


def _document() -> Document:
    text = " ".join(f"word{i}" for i in range(200))
    return Document(id="doc_1", text=text)


def test_registered_chunking_strategy_names():
    assert chunking_registry.keys() == ["fixed_tokens", "recursive"]


def test_creates_fixed_tokens_chunker_with_constructor_arguments():
    chunker = chunking_registry.create("fixed_tokens", chunk_size=256, overlap=32)

    assert isinstance(chunker, FixedTokenChunker)
    chunks = chunker.chunk(_document())
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_creates_recursive_chunker_with_constructor_arguments():
    chunker = chunking_registry.create("recursive", chunk_size=256, overlap=32)

    assert isinstance(chunker, RecursiveChunker)
    chunks = chunker.chunk(_document())
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_switching_strategy_name_requires_no_other_code_changes():
    document = _document()

    for strategy_name in ("fixed_tokens", "recursive"):
        chunker = chunking_registry.create(strategy_name, chunk_size=256, overlap=32)
        chunks = chunker.chunk(document)
        assert chunks
        assert all(chunk.document_id == "doc_1" for chunk in chunks)
