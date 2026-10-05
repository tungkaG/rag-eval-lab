"""Light unit tests for `OpenAIProvider` (fake `openai` module, no live API calls)."""

from __future__ import annotations

import sys
import types

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError
from rag_eval_lab.generation.providers.openai import OpenAIProvider


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


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        OpenAIProvider(model_name="gpt-test")


def test_returns_answer_and_metadata_reusing_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider(model_name="gpt-test")

    text, metadata = provider.generate("What is the capital of France?")
    provider.generate("What is the capital of France?")

    assert text == "Paris."
    assert metadata["provider"] == "openai"
    assert metadata["total_tokens"] == 15
    assert len(instances) == 1  # one client, reused across both calls
    provider.close()
    assert instances[0].closed is True


def test_response_schema_appends_json_schema_to_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {"score": {"type": "integer"}}}

    provider = OpenAIProvider(model_name="gpt-test")
    provider.generate("evaluate this", _Schema)

    sent_prompt = instances[0].chat.completions.calls[0]["messages"][0]["content"]
    assert "evaluate this" in sent_prompt
    assert "score" in sent_prompt


def test_raises_generation_error_when_no_answer_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, answer="")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider(model_name="gpt-test")

    with pytest.raises(GenerationError):
        provider.generate("prompt")
