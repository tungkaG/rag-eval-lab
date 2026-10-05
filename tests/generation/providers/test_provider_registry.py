"""Unit test for `generation.providers.registry.provider_registry`."""

from __future__ import annotations

from rag_eval_lab.generation.providers.gemini import GeminiProvider
from rag_eval_lab.generation.providers.huggingface import HuggingFaceProvider
from rag_eval_lab.generation.providers.openai import OpenAIProvider
from rag_eval_lab.generation.providers.registry import provider_registry


def test_gemini_is_registered() -> None:
    assert "gemini" in provider_registry.keys()
    assert provider_registry._factories["gemini"] is GeminiProvider


def test_huggingface_is_registered() -> None:
    assert "huggingface" in provider_registry.keys()
    assert provider_registry._factories["huggingface"] is HuggingFaceProvider


def test_openai_is_registered() -> None:
    assert "openai" in provider_registry.keys()
    assert provider_registry._factories["openai"] is OpenAIProvider
