"""Unit test for `generation.registry.generation_registry`."""

from __future__ import annotations

from rag_eval_lab.generation.gemini import GeminiGenerator
from rag_eval_lab.generation.openai import OpenAIGenerator
from rag_eval_lab.generation.registry import generation_registry


def test_gemini_is_registered() -> None:
    assert "gemini" in generation_registry.keys()
    assert generation_registry._factories["gemini"] is GeminiGenerator


def test_openai_is_registered() -> None:
    assert "openai" in generation_registry.keys()
    assert generation_registry._factories["openai"] is OpenAIGenerator
