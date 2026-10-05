"""Generic RAG answer generator: builds the RAG prompt and delegates execution to any registered LLM provider.

Any provider registered in `generation.providers.provider_registry` (Gemini,
local Hugging Face, ...) works here with no provider-specific code -- the
provider owns model/client setup, execution, response text, usage metadata,
and cleanup (see `generation/providers/`). Adding a new provider there makes
it usable for both RAG answer generation (here) and LLM judging
(`evaluation.answer.llm_judge`) without another generator wrapper.
"""

from __future__ import annotations

from typing import Any

from rag_eval_lab.core.exceptions import GenerationError
from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.generation.prompt_builder import build_prompt
from rag_eval_lab.generation.providers.base import close_provider
from rag_eval_lab.generation.providers.registry import provider_registry


class RagGenerator:
    """Generates answers by building the RAG prompt and delegating to a registered LLM provider.

    Constructs the provider once in `__init__` and reuses it across every
    `generate()` call; call `close()` once after the experiment to release
    it (see `generation.base.close_generator`).
    """

    def __init__(self, provider: str, **provider_params: Any) -> None:
        self._provider = provider_registry.create(provider, **provider_params)

    def generate(self, query: Query, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, dict[str, Any]]:
        prompt = build_prompt(query, retrieved_chunks)

        answer, metadata = self._provider.generate(prompt)
        if not answer or not answer.strip():
            raise GenerationError("Generation provider returned no answer text")

        return answer, metadata

    def close(self) -> None:
        """Releases the underlying provider."""
        close_provider(self._provider)
