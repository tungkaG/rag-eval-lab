"""Minimal helpers for working with `MetricResult` lists.

Aggregation across samples, persistence, and experiment orchestration belong
to later phases (`experiments/`), not here.
"""

from __future__ import annotations

from collections import defaultdict

from rag_eval_lab.core.models import MetricResult


def group_by_name(results: list[MetricResult]) -> dict[str, list[MetricResult]]:
    """Groups metric results by `name`, preserving their original order."""
    grouped: dict[str, list[MetricResult]] = defaultdict(list)
    for result in results:
        grouped[result.name].append(result)
    return dict(grouped)
