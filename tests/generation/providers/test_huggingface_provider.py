"""Light unit tests for `HuggingFaceProvider` (fake `transformers` module, no model download)."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from rag_eval_lab.core.exceptions import GenerationError
from rag_eval_lab.generation.providers.huggingface import HuggingFaceProvider


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, model_max_length: int | None = None):
    load_calls = {"tokenizer": 0, "model": 0}
    generate_calls: list[dict] = []
    chat_template_calls: list[list[dict]] = []
    decode_calls: list[torch.Tensor] = []

    class _FakeModelInputs(dict):
        def to(self, device):
            return self

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.model_max_length = model_max_length if model_max_length is not None else int(1e30)

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ):
            chat_template_calls.append(messages)
            return _FakeModelInputs(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))

        def decode(self, token_ids, skip_special_tokens=True):
            decode_calls.append(token_ids)
            return "the answer"

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, input_ids, **kwargs):
            generate_calls.append(kwargs)
            new_tokens = torch.tensor([[6, 7, 8]])
            return torch.cat([input_ids, new_tokens], dim=1)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            load_calls["tokenizer"] += 1
            return _FakeTokenizer()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name):
            load_calls["model"] += 1
            return _FakeModel()

    fake_module = types.ModuleType("transformers")
    fake_module.AutoTokenizer = _AutoTokenizer
    fake_module.AutoModelForCausalLM = _AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    return load_calls, generate_calls, chat_template_calls, decode_calls


def test_model_and_tokenizer_loaded_once_and_reused_across_generate_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls, generate_calls, _, _ = _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    provider.generate("prompt one")
    provider.generate("prompt two")

    assert load_calls == {"tokenizer": 1, "model": 1}
    assert len(generate_calls) == 2


def test_uses_chat_template_with_single_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, chat_template_calls, _ = _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    provider.generate("What is the capital of France?")

    messages = chat_template_calls[0]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "What is the capital of France?" in messages[0]["content"]


def test_response_schema_appends_json_schema_to_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, chat_template_calls, _ = _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    class _Schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {"score": {"type": "integer"}}}

    provider.generate("evaluate this", _Schema)

    content = chat_template_calls[0][0]["content"]
    assert "evaluate this" in content
    assert "score" in content


def test_decodes_only_newly_generated_tokens_excluding_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, decode_calls = _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    provider.generate("prompt")

    assert decode_calls[0].tolist() == [6, 7, 8]  # the 5 fake prompt tokens are excluded


def test_passes_temperature_only_when_sampling_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _, generate_calls, _, _ = _install_fake_transformers(monkeypatch)

    HuggingFaceProvider(model_name="fake-model", do_sample=False, temperature=0.7).generate("prompt")
    assert "temperature" not in generate_calls[0]

    HuggingFaceProvider(model_name="fake-model", do_sample=True, temperature=0.7).generate("prompt")
    assert generate_calls[1]["temperature"] == 0.7


def test_raises_when_prompt_plus_max_new_tokens_exceeds_context_length(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, model_max_length=6)
    provider = HuggingFaceProvider(model_name="fake-model", max_new_tokens=10)

    with pytest.raises(GenerationError):
        provider.generate("prompt")


def test_metadata_includes_provider_model_and_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    text, metadata = provider.generate("prompt")

    assert text == "the answer"
    assert metadata["provider"] == "huggingface"
    assert metadata["model_name"] == "fake-model"
    assert metadata["input_token_count"] == 5
    assert metadata["output_token_count"] == 3
    assert metadata["do_sample"] is False


def test_raises_generation_error_when_response_has_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, decode_calls = _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    # Patch decode to return blank text for this one call.
    provider._tokenizer.decode = lambda token_ids, skip_special_tokens=True: "   "

    with pytest.raises(GenerationError):
        provider.generate("prompt")


def test_close_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch)
    provider = HuggingFaceProvider(model_name="fake-model")

    provider.close()
