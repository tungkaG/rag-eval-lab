"""A small orchestrator that evaluates a sample with planner-selected metrics."""

from __future__ import annotations

from rag_eval_lab.core.models import EvaluationSample, MetricResult
from rag_eval_lab.evaluation.planner import DEFAULT_K_VALUES, plan_metrics


class EvaluationEngine:
    """Evaluates one `EvaluationSample` at a time using automatically selected metrics.

    Experiment orchestration, aggregation across samples, persistence, and
    YAML configuration belong to later phases, not here.
    """

    def __init__(self, k_values: list[int] | None = None) -> None:
        self.k_values = list(k_values) if k_values is not None else list(DEFAULT_K_VALUES)

    def evaluate(self, sample: EvaluationSample) -> list[MetricResult]:
        ground_truth = sample.evaluation_case.ground_truth
        metrics = plan_metrics(ground_truth, self.k_values)
        return [metric.evaluate(sample) for metric in metrics]
