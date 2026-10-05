"""Local Hugging Face Transformers `LlmProvider`.

Shared by RAG answer generation (`generation.huggingface`, which builds the
RAG prompt and passes no response schema) and LLM judging
(`evaluation.answer.llm_judge`, which builds its own rubric prompt and passes
a Pydantic response schema): loads the tokenizer and model once per instance,
reuses them across every `generate()` call, applies the single-user-message
chat template, runs under inference mode, decodes only the newly generated
tokens, and respects the tokenizer's context budget. This model has no native
structured-output API, so when a `response_schema` is supplied its JSON
representation is appended to the prompt and only JSON is requested; the
caller is still responsible for validating the response (prompting for JSON
does not guarantee valid JSON).

`transformers` (the `nli` extra; `torch` already ships with
sentence-transformers) is imported lazily, only when a `HuggingFaceProvider`
is constructed, so retrieval-only experiments never need it installed or any
model downloaded.
"""

from __future__ import annotations

import json
from typing import Any

from rag_eval_lab.core.exceptions import ConfigurationError, GenerationError

# Hugging Face tokenizers without a configured limit report a sentinel this
# large; anything below it is treated as a real context length to respect.
_UNSET_CONTEXT_LENGTH_THRESHOLD = 1_000_000


class HuggingFaceProvider:
    """Runs prompts through a local instruction-tuned causal LM via Transformers.

    Loads the tokenizer and model once per instance in `__init__` and reuses
    them across every `generate()` call.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float | None = None,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "transformers is required for the 'huggingface' provider; install it with "
                'pip install -e ".[nli]"'
            ) from exc

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()

    def generate(self, prompt: str, response_schema: type | None = None) -> tuple[str, dict[str, Any]]:
        import torch

        if response_schema is not None:
            schema_dict = response_schema.model_json_schema()
            top_level_keys = list(schema_dict.get("properties", {}).keys())

            prompt = (
                f"{prompt}\n\n"
                "IMPORTANT OUTPUT FORMAT:\n"
                "Return ONLY the JSON object itself.\n"
                f"The top-level keys must be exactly: {top_level_keys}.\n"
                "Do NOT wrap the result inside another object.\n"
                "Do NOT use the schema class/model name as a JSON key.\n"
                "Do NOT include Markdown or ```json fences."
            )
            
        model_inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        input_length = input_ids.shape[1]

        context_limit = getattr(self._tokenizer, "model_max_length", None)
        if context_limit is not None and context_limit < _UNSET_CONTEXT_LENGTH_THRESHOLD:
            if input_length + self.max_new_tokens > context_limit:
                raise GenerationError(
                    f"Prompt ({input_length} tokens) plus max_new_tokens ({self.max_new_tokens}) exceeds "
                    f"'{self.model_name}'s context length ({context_limit}); reduce retrieval/reranking "
                    "top_k, shorten the prompt, or lower max_new_tokens."
                )

        model_inputs = model_inputs.to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample and self.temperature is not None:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**model_inputs, **generation_kwargs)

        new_tokens = output_ids[0][input_length:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise GenerationError("Hugging Face model produced no response text")

        metadata: dict[str, Any] = {
            "provider": "huggingface",
            "model_name": self.model_name,
            "input_token_count": int(input_length),
            "output_token_count": int(new_tokens.shape[0]),
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.do_sample and self.temperature is not None:
            metadata["temperature"] = self.temperature

        return text, metadata

    def close(self) -> None:
        """No long-lived client to release; local tensors/model are left to normal GC."""
