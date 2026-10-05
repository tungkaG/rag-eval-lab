"""Unit tests for `RagGenerator` (fake provider registered in `provider_registry`, no SDK/model needed)."""

from __future__ import annotations

from typing import Any

import pytest

from rag_eval_lab.core.exceptions import GenerationError
from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.generation.generator import RagGenerator
from rag_eval_lab.generation.providers.registry import provider_registry


class _FakeProvider:
    instances: list[_FakeProvider] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        _FakeProvider.instances.append(self)

    def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]:
        self.calls.append(prompt)
        return "the answer", {"provider": "fake", "model_name": self.kwargs.get("model_name")}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _register_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeProvider.instances = []
    monkeypatch.setitem(provider_registry._factories, "fake_provider", _FakeProvider)


def _query_and_chunks() -> tuple[Query, list[RetrievedChunk]]:
    query = Query(id="q1", text="What is the capital of France?")
    chunks = [
        RetrievedChunk(chunk=Chunk(id="c1", document_id="d1", text="Paris is the capital of France."), score=0.9, rank=1)
    ]
    return query, chunks


def test_constructs_the_registered_provider_with_forwarded_params() -> None:
    RagGenerator(provider="fake_provider", model_name="fake-model", temperature=0.5)

    assert len(_FakeProvider.instances) == 1
    assert _FakeProvider.instances[0].kwargs == {"model_name": "fake-model", "temperature": 0.5}


def test_reuses_one_provider_across_multiple_generate_calls() -> None:
    generator = RagGenerator(provider="fake_provider", model_name="fake-model")

    generator.generate(*_query_and_chunks())
    generator.generate(*_query_and_chunks())

    assert len(_FakeProvider.instances) == 1
    assert len(_FakeProvider.instances[0].calls) == 2


def test_builds_the_rag_prompt_with_question_and_chunk_label() -> None:
    generator = RagGenerator(provider="fake_provider", model_name="fake-model")
    query, chunks = _query_and_chunks()

    generator.generate(query, chunks)

    prompt = _FakeProvider.instances[0].calls[0]
    assert query.text in prompt
    assert chunks[0].chunk.id in prompt


def test_calls_the_provider_without_a_response_schema() -> None:
    schema_calls: list[type | None] = []

    class _SchemaCapturingProvider(_FakeProvider):
        def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]:
            schema_calls.append(response_schema)
            return super().generate(prompt, response_schema)

    provider_registry._factories["fake_provider"] = _SchemaCapturingProvider
    generator = RagGenerator(provider="fake_provider", model_name="fake-model")

    generator.generate(*_query_and_chunks())

    assert schema_calls == [None]


def test_returns_answer_text_and_metadata() -> None:
    generator = RagGenerator(provider="fake_provider", model_name="fake-model")

    answer, metadata = generator.generate(*_query_and_chunks())

    assert answer == "the answer"
    assert metadata == {"provider": "fake", "model_name": "fake-model"}


def test_raises_generation_error_when_provider_returns_no_answer_text() -> None:
    class _BlankProvider(_FakeProvider):
        def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]:
            return "   ", {}

    provider_registry._factories["fake_provider"] = _BlankProvider
    generator = RagGenerator(provider="fake_provider", model_name="fake-model")

    with pytest.raises(GenerationError):
        generator.generate(*_query_and_chunks())


def test_close_delegates_to_the_underlying_provider() -> None:
    generator = RagGenerator(provider="fake_provider", model_name="fake-model")

    generator.close()

    assert _FakeProvider.instances[0].closed is True
