"""A small orchestrator that evaluates a sample with planner-selected and configured answer metrics."""

from __future__ import annotations

from time import perf_counter

from rag_eval_lab.core.models import EvaluationSample, MetricResult
from rag_eval_lab.evaluation.answer.base import AnswerEvaluator, is_applicable
from rag_eval_lab.evaluation.evidence.base import EvidenceMatcher
from rag_eval_lab.evaluation.evidence.registry import plan_evidence_metrics
from rag_eval_lab.evaluation.planner import DEFAULT_K_VALUES, plan_metrics


class EvaluationEngine:
    """Evaluates one `EvaluationSample` at a time using automatically selected metrics.

    Retrieval metrics are selected automatically from ground truth
    capabilities, as before. `answer_evaluators` are explicitly configured
    and only run when each one's declared `required_inputs` are available on
    the sample (see `evaluation.answer.base.is_applicable`); answer metrics
    always use `k=None`. `answer_evaluation_seconds` accumulates the total
    time spent across all `evaluate()` calls on this instance, so callers
    can record it separately from the overall evaluation timing.

    Experiment orchestration, aggregation across samples, persistence, and
    YAML configuration belong to later phases, not here.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        answer_evaluators: list[AnswerEvaluator] | None = None,
        evidence_matcher: EvidenceMatcher | None = None,
        evidence_threshold: float = 0.0,
    ) -> None:
        self.k_values = list(k_values) if k_values is not None else list(DEFAULT_K_VALUES)
        self.answer_evaluators = list(answer_evaluators) if answer_evaluators is not None else []
        self.evidence_matcher = evidence_matcher
        self.evidence_threshold = evidence_threshold
        self.answer_evaluation_seconds = 0.0

    def evaluate(self, sample: EvaluationSample) -> list[MetricResult]:
        ground_truth = sample.evaluation_case.ground_truth
        metrics = plan_metrics(ground_truth, self.k_values)
        results = [metric.evaluate(sample) for metric in metrics]

        if self.evidence_matcher is not None and "evidence_texts" in ground_truth.capabilities:
            ranked_chunks = sorted(sample.retrieved_chunks, key=lambda result: result.rank)
            largest_k = max(self.k_values, default=0)
            evidence_scores = self.evidence_matcher.score(
                [result.chunk.text for result in ranked_chunks[:largest_k]],
                ground_truth.evidence_texts,
            )
            evidence_metrics = plan_evidence_metrics(
                ground_truth,
                self.k_values,
                self.evidence_threshold,
                evidence_scores,
                self.evidence_matcher.metadata,
            )
            results.extend(metric.evaluate(sample) for metric in evidence_metrics)

        if self.answer_evaluators:
            started = perf_counter()
            for evaluator in self.answer_evaluators:
                if is_applicable(evaluator, sample):
                    results.extend(evaluator.evaluate(sample))
            self.answer_evaluation_seconds += perf_counter() - started

        return results
