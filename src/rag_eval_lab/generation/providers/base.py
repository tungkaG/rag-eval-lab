"""Shared LLM-provider protocol: owns model/client setup and one prompt->response call.

Reused by both RAG answer generation (`generation.gemini`/`generation.huggingface`,
which build the RAG prompt and pass no response schema) and LLM judging
(`evaluation.answer.llm_judge`, which builds its own rubric prompt and passes
a Pydantic response schema). A provider never builds task-specific prompts or
validates responses -- it only executes a prompt against a model and returns
the raw response text plus plain-JSON usage metadata. `response_schema` is
optional and provider-specific handling of it (native structured output,
or appending the schema to the prompt) stays inside each backend.
`close_provider` is the same kind of lifecycle escape hatch as
`generation.close_generator`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LlmProvider(Protocol):
    """Executes one prompt and returns `(response_text, usage_metadata)`."""

    def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]: ...

    def close(self) -> None: ...


def close_provider(provider: LlmProvider) -> None:
    """Calls `provider.close()`."""
    provider.close()
