"""Light unit tests for `LlmJudgeEvaluator` (fake `google.genai` module, no live API calls)."""

from __future__ import annotations

import json
import sys
import types

import pytest

from rag_eval_lab.core.exceptions import GenerationError, UnknownStrategyError
from rag_eval_lab.core.models import Chunk, EvaluationCase, EvaluationSample, GroundTruth, RetrievedChunk
from rag_eval_lab.evaluation.answer.llm_judge import LlmJudgeEvaluator


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch, response_text: str) -> list[dict]:
    """Installs a fake `google.genai` module and returns the list of `generate_content` call kwargs."""
    calls: list[dict] = []

    class _Response:
        text = response_text
        usage_metadata = None

    class _Models:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return _Response()

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            self.models = _Models()

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = lambda **kwargs: _FakeClient(**kwargs)
    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)

    return calls


def _sample(with_reference: bool) -> EvaluationSample:
    case = EvaluationCase(
        query_id="q1",
        question="What is the capital of France?",
        ground_truth=GroundTruth(reference_answer="Paris." if with_reference else None),
    )
    retrieved_chunks = [
        RetrievedChunk(chunk=Chunk(id="c1", document_id="d1", text="Paris is the capital of France."), score=1.0, rank=1)
    ]
    return EvaluationSample(evaluation_case=case, retrieved_chunks=retrieved_chunks, generated_answer="Paris.")


def test_unknown_provider_raises_unknown_strategy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, "{}")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(UnknownStrategyError):
        LlmJudgeEvaluator(provider="not_a_real_provider", model_name="gemini-test")


def test_valid_response_without_reference_returns_two_metrics_and_selects_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_genai(
        monkeypatch,
        '{"faithfulness": {"score": 4, "explanation": "ok"}, "relevance": {"score": 3, "explanation": "ok"}}',
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")
    results = evaluator.evaluate(_sample(with_reference=False))

    assert {r.name for r in results} == {"answer_judge_faithfulness", "answer_judge_relevance"}
    assert next(r for r in results if r.name == "answer_judge_faithfulness").value == 1.0
    assert calls[0]["config"]["response_schema"] is evaluator._schema_without_reference


def test_valid_response_with_reference_returns_three_metrics_and_selects_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_genai(
        monkeypatch,
        '{"faithfulness": {"score": 4, "explanation": "ok"}, "relevance": {"score": 4, "explanation": "ok"}, '
        '"correctness": {"score": 2, "explanation": "ok"}}',
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")
    results = evaluator.evaluate(_sample(with_reference=True))

    assert {r.name for r in results} == {
        "answer_judge_faithfulness",
        "answer_judge_relevance",
        "answer_judge_correctness",
    }
    assert next(r for r in results if r.name == "answer_judge_correctness").value == 0.5  # score 2 / 4
    assert calls[0]["config"]["response_schema"] is evaluator._schema_with_reference


def test_malformed_json_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, "not json")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")

    with pytest.raises(GenerationError):
        evaluator.evaluate(_sample(with_reference=False))


def test_missing_required_criterion_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch, '{"faithfulness": {"score": 4, "explanation": "ok"}}')  # missing relevance
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")

    with pytest.raises(GenerationError):
        evaluator.evaluate(_sample(with_reference=False))


@pytest.mark.parametrize("bad_score", [5, -1, 4.5, True, "4"])
def test_score_outside_strict_integer_bounds_raises_generation_error(
    monkeypatch: pytest.MonkeyPatch, bad_score
) -> None:
    response_text = json.dumps(
        {"faithfulness": {"score": bad_score, "explanation": "ok"}, "relevance": {"score": 3, "explanation": "ok"}}
    )
    _install_fake_genai(monkeypatch, response_text)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")

    with pytest.raises(GenerationError):
        evaluator.evaluate(_sample(with_reference=False))


@pytest.mark.parametrize("bad_explanation", ["", "   "])
def test_blank_explanation_raises_generation_error(monkeypatch: pytest.MonkeyPatch, bad_explanation) -> None:
    response_text = json.dumps(
        {
            "faithfulness": {"score": 4, "explanation": bad_explanation},
            "relevance": {"score": 3, "explanation": "ok"},
        }
    )
    _install_fake_genai(monkeypatch, response_text)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")

    with pytest.raises(GenerationError):
        evaluator.evaluate(_sample(with_reference=False))


def test_metadata_includes_raw_score_provider_and_rubric_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(
        monkeypatch,
        '{"faithfulness": {"score": 3, "explanation": "ok"}, "relevance": {"score": 4, "explanation": "ok"}}',
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    evaluator = LlmJudgeEvaluator(provider="gemini", model_name="gemini-test")
    results = evaluator.evaluate(_sample(with_reference=False))

    faithfulness = next(r for r in results if r.name == "answer_judge_faithfulness")
    assert faithfulness.metadata["score"] == 3
    assert faithfulness.metadata["explanation"] == "ok"
    assert faithfulness.metadata["provider"] == "gemini"
    assert faithfulness.metadata["rubric_version"] == "v2"


def test_retrieval_only_import_does_not_require_pydantic_or_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the module must never touch pydantic/google-genai -- only constructing does."""
    import importlib

    monkeypatch.delitem(sys.modules, "pydantic", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "rag_eval_lab.evaluation.answer.llm_judge", raising=False)

    module = importlib.import_module("rag_eval_lab.evaluation.answer.llm_judge")

    assert hasattr(module, "LlmJudgeEvaluator")


def test_backend_selected_through_registry_with_no_provider_branching(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake provider registered under a throwaway name proves the evaluator never branches on provider name."""
    calls: list[tuple[str, type]] = []

    class _FakeProvider:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate(self, prompt: str, response_schema: type | None = None):
            calls.append((prompt, response_schema))
            return (
                json.dumps(
                    {"faithfulness": {"score": 4, "explanation": "ok"}, "relevance": {"score": 4, "explanation": "ok"}}
                ),
                {"provider": "fake", "model_name": "fake-model"},
            )

        def close(self) -> None:
            pass

    from rag_eval_lab.generation.providers.registry import provider_registry

    monkeypatch.setitem(provider_registry._factories, "fake_provider", _FakeProvider)

    evaluator = LlmJudgeEvaluator(provider="fake_provider", model_name="fake-model")
    results = evaluator.evaluate(_sample(with_reference=False))

    assert len(calls) == 1
    assert {r.name for r in results} == {"answer_judge_faithfulness", "answer_judge_relevance"}
    evaluator.close()

