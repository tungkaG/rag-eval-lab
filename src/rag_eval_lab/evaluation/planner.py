"""Selects applicable metrics automatically from ground truth capabilities."""

from __future__ import annotations

from rag_eval_lab.core.models import GroundTruth
from rag_eval_lab.core.protocols import Metric
from rag_eval_lab.evaluation.registry import metric_registry

DEFAULT_K_VALUES = (1, 5, 10)


def plan_metrics(ground_truth: GroundTruth, k_values: list[int] | None = None) -> list[Metric]:
    """Instantiates one metric per (registered metric, k) pair that applies.

    A metric applies when its `required_capabilities` are a subset of
    `ground_truth.capabilities`. No dataset or benchmark specific logic
    lives here; selection is driven entirely by declared capabilities.
    """
    resolved_k_values = list(k_values) if k_values is not None else list(DEFAULT_K_VALUES)
    capabilities = ground_truth.capabilities

    metrics: list[Metric] = []
    for name in metric_registry.keys():
        for k in resolved_k_values:
            metric = metric_registry.create(name, k=k)
            if metric.required_capabilities <= capabilities:
                metrics.append(metric)
    return metrics
