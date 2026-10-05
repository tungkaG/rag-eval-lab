"""Gemini-backed `LlmProvider`, wrapping the `google-genai` SDK.

`google.genai` is an optional dependency (the `generation` extra) and is
imported lazily, only when a `GeminiProvider` is constructed -- so
retrieval-only experiments never need it installed.
"""

from __future__ import annotations

import os
from typing import Any

from rag_eval_lab.core.env import load_env_file
from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError

_API_KEY_ENV_VAR = "GEMINI_API_KEY"


class GeminiProvider:
    """Runs prompts through the Gemini Developer API.

    Builds one `genai.Client` per instance in `__init__` and reuses it across
    every `generate()` call; call `close()` once after the experiment to
    release it (see `generation.providers.base.close_provider`).

    Without a `response_schema`, mirrors plain-text answer generation
    (`temperature`/`max_output_tokens`). With one, switches to Gemini's
    native JSON response mode for LLM judging instead.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_output_tokens: int = 1024) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ConfigurationError(
                "google-genai is required for the 'gemini' provider; install it with "
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

    def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]:
        config: dict[str, Any] = {"temperature": self.temperature}
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        else:
            config["max_output_tokens"] = self.max_output_tokens

        response = self._client.models.generate_content(model=self.model_name, contents=prompt, config=config)

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise GenerationError("Gemini response contained no text")

        return text, self._response_metadata(response)

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
