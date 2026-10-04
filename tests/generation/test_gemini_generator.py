"""Unit tests for `GeminiGenerator`, using a fake `google.genai` module (no live API calls)."""

from __future__ import annotations

import sys
import types

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError
from rag_eval_lab.core.models import Chunk, Query, RetrievedChunk
from rag_eval_lab.generation.gemini import GeminiGenerator


class _FakeModels:
    def __init__(self, make_response) -> None:
        self._make_response = make_response
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._make_response()


class _FakeClient:
    def __init__(self, instances: list, make_response, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.models = _FakeModels(make_response)
        self.closed = False
        instances.append(self)

    def close(self) -> None:
        self.closed = True


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch, make_response) -> list[_FakeClient]:
    instances: list[_FakeClient] = []

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = lambda **kwargs: _FakeClient(instances, make_response, **kwargs)

    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)

    return instances


def _make_response(
    text: str = "the answer",
    prompt_tokens: int | None = 10,
    candidates_tokens: int | None = 5,
    total_tokens: int | None = 15,
    finish_reason: object | None = "STOP",
):
    class _Usage:
        pass

    usage = _Usage()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = candidates_tokens
    usage.total_token_count = total_tokens

    class _Candidate:
        pass

    candidate = _Candidate()
    candidate.finish_reason = finish_reason

    class _Response:
        pass

    response = _Response()
    response.text = text
    response.usage_metadata = usage
    response.candidates = [candidate]
    return response


def _make_query_and_chunks() -> tuple[Query, list[RetrievedChunk]]:
    query = Query(id="q1", text="What is the capital of France?")
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id="c1", document_id="doc1", text="Paris is the capital of France."), score=0.9, rank=1
        )
    ]
    return query, chunks


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, _make_response)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        GeminiGenerator(model_name="gemini-test")


def test_missing_sdk_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # No fake `google.genai` installed here -- the real package genuinely
    # isn't a dependency of this project's base install.
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(ConfigurationError):
        GeminiGenerator(model_name="gemini-test")


def test_reuses_one_client_across_multiple_generate_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_genai(monkeypatch, _make_response)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiGenerator(model_name="gemini-test")
    query, chunks = _make_query_and_chunks()

    generator.generate(query, chunks)
    generator.generate(query, chunks)

    assert len(instances) == 1
    assert len(instances[0].models.calls) == 2


def test_returns_answer_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(
        monkeypatch,
        lambda: _make_response(
            text="Paris.", prompt_tokens=12, candidates_tokens=4, total_tokens=16, finish_reason="STOP"
        ),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiGenerator(model_name="gemini-test")
    query, chunks = _make_query_and_chunks()

    answer, metadata = generator.generate(query, chunks)

    assert answer == "Paris."
    assert metadata == {
        "provider": "gemini",
        "model_name": "gemini-test",
        "prompt_token_count": 12,
        "candidates_token_count": 4,
        "total_token_count": 16,
        "finish_reason": "STOP",
    }


def test_metadata_never_contains_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, _make_response)
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")

    generator = GeminiGenerator(model_name="gemini-test")
    query, chunks = _make_query_and_chunks()

    _, metadata = generator.generate(query, chunks)

    assert "super-secret-key" not in str(metadata)
    assert "api_key" not in metadata


def test_prompt_sent_to_model_contains_question_and_chunk_label(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, _make_response)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiGenerator(model_name="gemini-test")
    query, chunks = _make_query_and_chunks()
    generator.generate(query, chunks)

    call = generator._client.models.calls[0]
    assert query.text in call["contents"]
    assert chunks[0].chunk.id in call["contents"]


def test_constructor_passes_temperature_and_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, _make_response)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiGenerator(model_name="gemini-test", temperature=0.2, max_output_tokens=256)
    query, chunks = _make_query_and_chunks()
    generator.generate(query, chunks)

    call = generator._client.models.calls[0]
    assert call["config"]["temperature"] == 0.2
    assert call["config"]["max_output_tokens"] == 256


def test_raises_generation_error_when_response_has_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, lambda: _make_response(text=""))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiGenerator(model_name="gemini-test")
    query, chunks = _make_query_and_chunks()

    with pytest.raises(GenerationError):
        generator.generate(query, chunks)


def test_close_closes_the_underlying_client(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_genai(monkeypatch, _make_response)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiGenerator(model_name="gemini-test")
    generator.close()

    assert instances[0].closed is True
