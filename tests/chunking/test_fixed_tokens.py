import tiktoken

from rag_eval_lab.chunking.fixed_tokens import FixedTokenChunker
from rag_eval_lab.core.models import Chunk, Document

ENCODING = tiktoken.get_encoding("cl100k_base")


def _long_document(num_words: int = 500, **metadata) -> Document:
    text = " ".join(f"word{i}" for i in range(num_words))
    return Document(id="doc_1", text=text, metadata=metadata or {"source": "test"})


def test_returns_internal_chunk_objects():
    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(_long_document())

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_long_text_produces_multiple_chunks():
    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(_long_document())

    assert len(chunks) > 1


def test_respects_configured_chunk_size_and_overlap():
    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(_long_document())

    for chunk in chunks[:-1]:
        assert len(ENCODING.encode(chunk.text)) == 50

    first_tokens = ENCODING.encode(chunks[0].text)
    second_tokens = ENCODING.encode(chunks[1].text)
    assert first_tokens[-10:] == second_tokens[:10]


def test_repeated_calls_produce_same_ids_and_text():
    chunker = FixedTokenChunker(chunk_size=50, overlap=10)
    document = _long_document()

    chunks_a = chunker.chunk(document)
    chunks_b = chunker.chunk(document)

    assert [chunk.id for chunk in chunks_a] == [chunk.id for chunk in chunks_b]
    assert [chunk.text for chunk in chunks_a] == [chunk.text for chunk in chunks_b]


def test_deterministic_ids_start_at_zero():
    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(_long_document())

    assert [chunk.id for chunk in chunks] == [f"doc_1_chunk_{i}" for i in range(len(chunks))]


def test_preserves_parent_document_id_and_metadata():
    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(_long_document(source="test"))

    for chunk in chunks:
        assert chunk.document_id == "doc_1"
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["strategy"] == "fixed_tokens"
        assert chunk.metadata["chunk_size"] == 50
        assert chunk.metadata["overlap"] == 10
        assert chunk.metadata["encoding_name"] == "cl100k_base"


def test_page_and_char_offsets_default_to_none_when_unavailable():
    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(_long_document())

    for chunk in chunks:
        assert chunk.page_start is None
        assert chunk.page_end is None
        assert chunk.char_start is None
        assert chunk.char_end is None


def test_preserves_page_range_from_document_metadata_when_available():
    document = _long_document(page_start=3, page_end=9)

    chunks = FixedTokenChunker(chunk_size=50, overlap=10).chunk(document)

    for chunk in chunks:
        assert chunk.page_start == 3
        assert chunk.page_end == 9
