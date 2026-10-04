"""Gemini-backed `GenerationStrategy`, wrapping the `google-genai` SDK.

`google.genai` is an optional dependency (the `generation` extra) and is
imported lazily, only when a `GeminiGenerator` is constructed -- so
retrieval-only experiments never need it installed.
"""

from __future__ import annotations

import os
from typing import Any

from rag_eval_lab.core.env import load_env_file
from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError
from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.generation.prompt_builder import build_prompt

_API_KEY_ENV_VAR = "GEMINI_API_KEY"


class GeminiGenerator:
    """Generates answers with the Gemini Developer API.

    Builds one `genai.Client` per instance in `__init__` and reuses it across
    every `generate()` call; call `close()` once after the experiment to
    release it (see `generation.base.close_generator`).
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_output_tokens: int = 1024) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ConfigurationError(
                "google-genai is required for the 'gemini' generator; install it with "
                'pip install -e ".[generation]"'
            ) from exc

        load_env_file()
        api_key = os.environ.get(_API_KEY_ENV_VAR)
        if not api_key:
            raise ConfigurationError(f"{_API_KEY_ENV_VAR} environment variable is not set")

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=api_key)

    def generate(self, query: Query, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, dict[str, Any]]:
        prompt = build_prompt(query, retrieved_chunks)

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"temperature": self.temperature, "max_output_tokens": self.max_output_tokens},
        )

        answer = getattr(response, "text", None)
        if not answer or not answer.strip():
            raise GenerationError("Gemini response contained no answer text")

        return answer, self._response_metadata(response)

    def _response_metadata(self, response: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {"provider": "gemini", "model_name": self.model_name}

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
                value = getattr(usage, key, None)
                if value is not None:
                    metadata[key] = value

        candidates = getattr(response, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason is not None:
                metadata["finish_reason"] = str(finish_reason)

        return metadata

    def close(self) -> None:
        """Releases the underlying `genai.Client`."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
