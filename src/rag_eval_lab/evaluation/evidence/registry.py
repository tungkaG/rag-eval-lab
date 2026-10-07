"""Registry of evidence matchers, keyed by matching strategy name."""

from __future__ import annotations

from typing import Any

from rag_eval_lab.core.models import GroundTruth
from rag_eval_lab.core.protocols import Metric
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.evaluation.evidence.base import EvidenceMatcher
from rag_eval_lab.evaluation.evidence.matchers import EmbeddingEvidenceMatcher, LexicalEvidenceMatcher
from rag_eval_lab.evaluation.evidence.metrics import EvidencePrecisionAtK, EvidenceRecallAtK

evidence_matcher_registry: Registry[EvidenceMatcher] = Registry(kind="evidence matcher")
evidence_metric_registry: Registry[Metric] = Registry(kind="evidence metric")

evidence_matcher_registry.register("lexical")(LexicalEvidenceMatcher)
evidence_matcher_registry.register("embedding")(EmbeddingEvidenceMatcher)
evidence_metric_registry.register("evidence_recall")(EvidenceRecallAtK)
evidence_metric_registry.register("evidence_precision")(EvidencePrecisionAtK)


def plan_evidence_metrics(
	ground_truth: GroundTruth,
	k_values: list[int],
	threshold: float,
	scores: list[list[float]],
	matcher_metadata: dict[str, Any],
) -> list[Metric]:
	"""Constructs each applicable registered evidence metric for every configured cutoff."""
	metrics: list[Metric] = []
	for name in evidence_metric_registry.keys():
		for k in k_values:
			metric = evidence_metric_registry.create(
				name,
				k=k,
				threshold=threshold,
				scores=scores,
				metadata=matcher_metadata,
			)
			if metric.required_capabilities <= ground_truth.capabilities:
				metrics.append(metric)
	return metrics