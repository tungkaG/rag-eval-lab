"""OpenAI-backed `LlmProvider`, wrapping the `openai` SDK.

`openai` is an optional dependency (the `generation` extra) and is imported
lazily, only when an `OpenAIProvider` is constructed -- so retrieval-only
experiments never need it installed. This model has no native
structured-output handling here, so when a `response_schema` is supplied
its JSON representation is appended to the prompt and only JSON is
requested, same as `HuggingFaceProvider`; the caller is still responsible
for validating the response.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rag_eval_lab.core.env import load_env_file
from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError

_API_KEY_ENV_VAR = "OPENAI_API_KEY"


class OpenAIProvider:
    """Runs prompts through the OpenAI Chat Completions API.

    Builds one `openai.OpenAI` client per instance in `__init__` and reuses
    it across every `generate()` call; call `close()` once after the
    experiment to release it (see `generation.providers.base.close_provider`).
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_output_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError(
                "openai is required for the 'openai' provider; install it with " 'pip install -e ".[generation]"'
            ) from exc

        load_env_file()
        api_key = os.environ.get(_API_KEY_ENV_VAR)
        if not api_key:
            raise ConfigurationError(f"{_API_KEY_ENV_VAR} environment variable is not set")

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]:
        if response_schema is not None:
            schema_json = json.dumps(response_schema.model_json_schema())
            prompt = f"{prompt}\n\nRespond with ONLY a JSON object matching this schema:\n{schema_json}"

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice is not None else None
        if not text or not text.strip():
            raise GenerationError("OpenAI response contained no text")

        return text, self._response_metadata(response, choice)

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
