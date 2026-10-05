"""Shared LLM-provider backends, reused by RAG answer generation and LLM judging."""

from rag_eval_lab.generation.providers.base import LlmProvider, close_provider
from rag_eval_lab.generation.providers.gemini import GeminiProvider
from rag_eval_lab.generation.providers.huggingface import HuggingFaceProvider
from rag_eval_lab.generation.providers.openai import OpenAIProvider
from rag_eval_lab.generation.providers.registry import provider_registry

__all__ = [
    "GeminiProvider",
    "HuggingFaceProvider",
    "LlmProvider",
    "OpenAIProvider",
    "close_provider",
    "provider_registry",
]
