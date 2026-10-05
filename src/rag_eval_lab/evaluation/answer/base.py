"""Answer evaluators all implement the shared `AnswerEvaluator` protocol.

Unlike retrieval `Metric`s (selected automatically from `GroundTruth`
capabilities), answer evaluators are explicitly configured
(`evaluation.answer_metrics`) and declare `required_inputs` -- the
`EvaluationSample`-level inputs they need -- so applicability can be checked
without inspecting strategy names. `close_evaluator` is the same kind of
escape hatch as `generation.close_generator`: an optional `close()` for
evaluators holding a long-lived client (e.g. the Gemini judge), without
adding it to the shared protocol.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from rag_eval_lab.core.models import EvaluationSample, MetricResult

AnswerEvaluationInput = Literal["generated_answer", "reference_answer", "retrieved_context"]

__all__ = [
    "AnswerEvaluationInput",
    "AnswerEvaluator",
    "available_inputs",
    "close_evaluator",
    "is_applicable",
]


@runtime_checkable
class AnswerEvaluator(Protocol):
    """Evaluates a completed `EvaluationSample`'s generated answer.

    Returns a list (not a single `MetricResult`) so one model call can
    produce several related metrics without duplicating calls.
    """

    required_inputs: set[AnswerEvaluationInput]

    def evaluate(self, sample: EvaluationSample) -> list[MetricResult]: ...


def available_inputs(sample: EvaluationSample) -> set[AnswerEvaluationInput]:
    """The `EvaluationSample`-level inputs actually available for answer evaluation."""
    available: set[AnswerEvaluationInput] = set()
    if sample.generated_answer is not None and sample.generated_answer.strip():
        available.add("generated_answer")
    if sample.evaluation_case.ground_truth.reference_answer:
        available.add("reference_answer")
    if sample.retrieved_chunks:
        available.add("retrieved_context")
    return available


def is_applicable(evaluator: AnswerEvaluator, sample: EvaluationSample) -> bool:
    """True when every one of `evaluator.required_inputs` is available on `sample`."""
    return evaluator.required_inputs <= available_inputs(sample)


def close_evaluator(evaluator: AnswerEvaluator) -> None:
    """Calls `evaluator.close()` if it defines one; a no-op otherwise."""
    close = getattr(evaluator, "close", None)
    if callable(close):
        close()
