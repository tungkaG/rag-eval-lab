"""Registry of answer evaluators, keyed by strategy name."""

from __future__ import annotations

from rag_eval_lab.core.registry import Registry
from rag_eval_lab.evaluation.answer.base import AnswerEvaluator
from rag_eval_lab.evaluation.answer.entailment import NliAnswerEvaluator
from rag_eval_lab.evaluation.answer.llm_judge import LlmJudgeEvaluator

answer_evaluator_registry: Registry[AnswerEvaluator] = Registry(kind="answer_evaluator")

answer_evaluator_registry.register("nli")(NliAnswerEvaluator)
answer_evaluator_registry.register("llm_judge")(LlmJudgeEvaluator)
