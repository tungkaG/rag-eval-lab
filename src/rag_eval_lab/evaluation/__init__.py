"""Retrieval evaluation engine: automatic metric selection from ground truth capabilities."""

from rag_eval_lab.evaluation.answer.registry import answer_evaluator_registry
from rag_eval_lab.evaluation.engine import EvaluationEngine
from rag_eval_lab.evaluation.planner import plan_metrics
from rag_eval_lab.evaluation.registry import metric_registry
from rag_eval_lab.evaluation.result import group_by_name

__all__ = ["EvaluationEngine", "answer_evaluator_registry", "group_by_name", "metric_registry", "plan_metrics"]
