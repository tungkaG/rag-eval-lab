"""OpenAI-backed `GenerationStrategy`, wrapping the `openai` SDK.

`openai` is an optional dependency (the `generation` extra) and is imported
lazily, only when an `OpenAIGenerator` is constructed -- so retrieval-only
experiments never need it installed.
"""

from __future__ import annotations

import os
from typing import Any

from rag_eval_lab.core.env import load_env_file
from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError
from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.generation.prompt_builder import build_prompt

_API_KEY_ENV_VAR = "OPENAI_API_KEY"


class OpenAIGenerator:
    """Generates answers with the OpenAI Chat Completions API.

    Builds one `openai.OpenAI` client per instance in `__init__` and reuses
    it across every `generate()` call; call `close()` once after the
    experiment to release it (see `generation.base.close_generator`).
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_output_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError(
                "openai is required for the 'openai' generator; install it with "
                'pip install -e ".[generation]"'
            ) from exc

        load_env_file()
        api_key = os.environ.get(_API_KEY_ENV_VAR)
        if not api_key:
            raise ConfigurationError(f"{_API_KEY_ENV_VAR} environment variable is not set")

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=api_key)

    def generate(self, query: Query, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, dict[str, Any]]:
        prompt = build_prompt(query, retrieved_chunks)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        choice = response.choices[0] if response.choices else None
        answer = choice.message.content if choice is not None else None
        if not answer or not answer.strip():
            raise GenerationError("OpenAI response contained no answer text")

        return answer, self._response_metadata(response, choice)

    def _response_metadata(self, response: Any, choice: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {"provider": "openai", "model_name": self.model_name}

        usage = getattr(response, "usage", None)
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, key, None)
                if value is not None:
                    metadata[key] = value

        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
        if finish_reason is not None:
            metadata["finish_reason"] = str(finish_reason)

        return metadata

    def close(self) -> None:
        """Releases the underlying `openai.OpenAI` client."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
