"""Retrieval evaluation engine: automatic metric selection from ground truth capabilities."""

from rag_eval_lab.evaluation.engine import EvaluationEngine
from rag_eval_lab.evaluation.planner import plan_metrics
from rag_eval_lab.evaluation.registry import metric_registry
from rag_eval_lab.evaluation.result import group_by_name

__all__ = ["EvaluationEngine", "group_by_name", "metric_registry", "plan_metrics"]
