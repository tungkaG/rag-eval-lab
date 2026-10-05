"""Registry of shared LLM-provider backends, keyed by provider name."""

from __future__ import annotations

from rag_eval_lab.core.registry import Registry
from rag_eval_lab.generation.providers.base import LlmProvider
from rag_eval_lab.generation.providers.gemini import GeminiProvider
from rag_eval_lab.generation.providers.huggingface import HuggingFaceProvider
from rag_eval_lab.generation.providers.openai import OpenAIProvider

provider_registry: Registry[LlmProvider] = Registry(kind="llm_provider")

provider_registry.register("gemini")(GeminiProvider)
provider_registry.register("huggingface")(HuggingFaceProvider)
provider_registry.register("openai")(OpenAIProvider)
