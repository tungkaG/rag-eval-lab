"""Unit tests for `generation.prompt_builder.build_prompt`."""

from __future__ import annotations

from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.generation.prompt_builder import build_prompt


def _retrieved_chunk(chunk_id: str, document_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=chunk_id, document_id=document_id, text=text), score=1.0, rank=rank)


def test_prompt_includes_the_question() -> None:
    query = Query(id="q1", text="What is the capital of France?")

    prompt = build_prompt(query, [])

    assert query.text in prompt


def test_prompt_labels_passages_with_chunk_and_document_id_in_retrieval_order() -> None:
    query = Query(id="q1", text="question")
    chunks = [
        _retrieved_chunk("c2", "doc2", "second passage", rank=1),
        _retrieved_chunk("c1", "doc1", "first passage", rank=2),
    ]

    prompt = build_prompt(query, chunks)

    # Labeled with both ids, and kept in the order the chunks were given (not re-sorted).
    assert prompt.index("c2") < prompt.index("c1")
    assert "doc2" in prompt
    assert "doc1" in prompt
    assert "second passage" in prompt
    assert "first passage" in prompt


def test_prompt_instructs_context_only_insufficiency_citation_and_evidence_framing() -> None:
    query = Query(id="q1", text="question")

    prompt = build_prompt(query, [_retrieved_chunk("c1", "doc1", "text", rank=1)])

    lowered = prompt.lower()
    assert "context" in lowered
    assert "does not contain enough information" in lowered
    assert "cite" in lowered
    assert "evidence" in lowered


def test_prompt_handles_no_retrieved_chunks() -> None:
    query = Query(id="q1", text="question with no candidates")

    prompt = build_prompt(query, [])

    assert query.text in prompt
    assert "no passages retrieved" in prompt.lower()
