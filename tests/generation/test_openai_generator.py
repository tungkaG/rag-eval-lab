"""Light unit tests for `OpenAIGenerator` (fake `openai` module, no live API calls)."""

from __future__ import annotations

import sys
import types

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError
from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.generation.openai import OpenAIGenerator


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, answer: str = "Paris."):
    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Message:
        content = answer

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    instances: list[_FakeClient] = []

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            self.chat = _Chat()
            self.closed = False
            instances.append(self)

        def close(self) -> None:
            self.closed = True

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return instances


def _query_and_chunks() -> tuple[Query, list[RetrievedChunk]]:
    return Query(id="q1", text="What is the capital of France?"), [
        RetrievedChunk(chunk=Chunk(id="c1", document_id="doc1", text="Paris is the capital of France."), score=0.9, rank=1)
    ]


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        OpenAIGenerator(model_name="gpt-test")


def test_returns_answer_and_metadata_reusing_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    generator = OpenAIGenerator(model_name="gpt-test")
    query, chunks = _query_and_chunks()

    answer, metadata = generator.generate(query, chunks)
    generator.generate(query, chunks)

    assert answer == "Paris."
    assert metadata["provider"] == "openai"
    assert metadata["total_tokens"] == 15
    assert len(instances) == 1  # one client, reused across both calls
    generator.close()
    assert instances[0].closed is True


def test_raises_generation_error_when_no_answer_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, answer="")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    generator = OpenAIGenerator(model_name="gpt-test")
    query, chunks = _query_and_chunks()

    with pytest.raises(GenerationError):
        generator.generate(query, chunks)
