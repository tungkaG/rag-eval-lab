"""Registry of answer generation strategies, keyed by provider name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import GenerationStrategy
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.generation.gemini import GeminiGenerator
from rag_eval_lab.generation.openai import OpenAIGenerator

generation_registry: Registry[GenerationStrategy] = Registry(kind="generation")

generation_registry.register("gemini")(GeminiGenerator)
generation_registry.register("openai")(OpenAIGenerator)
