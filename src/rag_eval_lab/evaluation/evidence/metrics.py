"""Evidence retrieval metrics computed from a pre-scored text-match matrix."""

from __future__ import annotations

from typing import Any

from rag_eval_lab.core.models import EvaluationSample, MetricResult


class _EvidenceMetricAtK:
    required_capabilities = {"evidence_texts"}

    def __init__(self, k: int, threshold: float, scores: list[list[float]], metadata: dict[str, Any]) -> None:
        self.k = k
        self.threshold = threshold
        self.scores = scores
        self.metadata = {**metadata, "threshold": threshold}

    def _top_k_scores(self, sample: EvaluationSample) -> list[list[float]]:
        actual_count = min(self.k, len(sample.retrieved_chunks))
        return self.scores[:actual_count]


class EvidenceRecallAtK(_EvidenceMetricAtK):
    """Fraction of gold evidence texts matched by at least one top-`k` chunk."""

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        evidence_count = len(sample.evaluation_case.ground_truth.evidence_texts)
        top_k_scores = self._top_k_scores(sample)
        matched_evidence = sum(
            any(score >= self.threshold for score in row_scores)
            for row_scores in zip(*top_k_scores)
        ) if top_k_scores else 0
        value = matched_evidence / evidence_count if evidence_count else 0.0
        return MetricResult(name="evidence_recall", value=value, k=self.k, metadata=self.metadata)


class EvidencePrecisionAtK(_EvidenceMetricAtK):
    """Fraction of top-`k` chunks matching at least one gold evidence text."""

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        top_k_scores = self._top_k_scores(sample)
        matched_chunks = sum(any(score >= self.threshold for score in row_scores) for row_scores in top_k_scores)
        value = matched_chunks / len(top_k_scores) if top_k_scores else 0.0
        return MetricResult(name="evidence_precision", value=value, k=self.k, metadata=self.metadata)