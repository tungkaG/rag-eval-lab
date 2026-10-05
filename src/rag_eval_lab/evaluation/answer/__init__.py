"""Optional answer evaluators: explicitly configured via `evaluation.answer_metrics`, disabled by default."""

from rag_eval_lab.evaluation.answer.base import (
    AnswerEvaluationInput,
    AnswerEvaluator,
    available_inputs,
    close_evaluator,
    is_applicable,
)
from rag_eval_lab.evaluation.answer.entailment import NliAnswerEvaluator
from rag_eval_lab.evaluation.answer.llm_judge import LlmJudgeEvaluator
from rag_eval_lab.evaluation.answer.registry import answer_evaluator_registry

__all__ = [
    "AnswerEvaluationInput",
    "AnswerEvaluator",
    "LlmJudgeEvaluator",
    "NliAnswerEvaluator",
    "answer_evaluator_registry",
    "available_inputs",
    "close_evaluator",
    "is_applicable",
]
