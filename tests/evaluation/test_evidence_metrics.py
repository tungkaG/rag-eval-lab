from rag_eval_lab.core.models import Chunk, EvaluationCase, EvaluationSample, GroundTruth, RetrievedChunk
from rag_eval_lab.evaluation.engine import EvaluationEngine
from rag_eval_lab.evaluation.evidence.matchers import LexicalEvidenceMatcher
from rag_eval_lab.evaluation.evidence.registry import evidence_metric_registry


def _sample(evidence_texts: list[str], chunk_texts: list[str]) -> EvaluationSample:
    chunks = [
        RetrievedChunk(
            chunk=Chunk(id=f"chunk-{index}", document_id=f"document-{index}", text=text),
            score=1.0,
            rank=index,
        )
        for index, text in enumerate(chunk_texts, start=1)
    ]
    return EvaluationSample(
        evaluation_case=EvaluationCase(
            query_id="question-1",
            question="question",
            ground_truth=GroundTruth(evidence_texts=evidence_texts),
        ),
        retrieved_chunks=chunks,
    )


def _evidence_results(sample: EvaluationSample, k_values: list[int]) -> dict[tuple[str, int], float]:
    results = EvaluationEngine(
        k_values=k_values,
        evidence_matcher=LexicalEvidenceMatcher(),
        evidence_threshold=1.0,
    ).evaluate(sample)
    return {(result.name, result.k): result.value for result in results}


def test_evidence_metrics_cover_full_partial_and_absent_matches_at_each_k() -> None:
    sample = _sample(
        ["Paris is the capital", "Everest is the tallest mountain"],
        ["Paris is the capital", "unrelated text", "Everest is the tallest mountain"],
    )

    results = _evidence_results(sample, [1, 2, 3])

    assert results[("evidence_recall", 1)] == 0.5
    assert results[("evidence_precision", 1)] == 1.0
    assert results[("evidence_recall", 2)] == 0.5
    assert results[("evidence_precision", 2)] == 0.5
    assert results[("evidence_recall", 3)] == 1.0
    assert results[("evidence_precision", 3)] == 2 / 3


def test_evidence_recall_counts_each_gold_text_once_and_one_chunk_can_match_many() -> None:
    sample = _sample(
        ["Paris is in France", "Paris is in France"],
        ["Paris is in France", "Paris is in France"],
    )

    results = _evidence_results(sample, [1, 2])

    assert results[("evidence_recall", 1)] == 1.0
    assert results[("evidence_recall", 2)] == 1.0
    assert results[("evidence_precision", 2)] == 1.0


def test_evidence_metrics_are_skipped_when_evidence_is_missing() -> None:
    sample = _sample([], ["Paris is the capital"])

    results = EvaluationEngine(
        k_values=[1],
        evidence_matcher=LexicalEvidenceMatcher(),
        evidence_threshold=1.0,
    ).evaluate(sample)

    assert results == []


def test_registered_evidence_metric_runs_with_one_score_matrix_per_case(monkeypatch) -> None:
    class _CountingMatcher:
        name = "counting"
        metadata = {"matcher": name}

        def __init__(self) -> None:
            self.calls = 0

        def score(self, retrieved_texts: list[str], evidence_texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0 for _ in evidence_texts] for _ in retrieved_texts]

    class _TestEvidenceMetric:
        required_capabilities = {"evidence_texts"}

        def __init__(self, k: int, threshold: float, scores: list[list[float]], metadata: dict) -> None:
            self.k = k
            self.scores = scores

        def evaluate(self, sample: EvaluationSample):
            return MetricResult(name="evidence_test_metric", value=float(len(self.scores)), k=self.k)

    from rag_eval_lab.core.models import MetricResult

    monkeypatch.setitem(evidence_metric_registry._factories, "evidence_test_metric", _TestEvidenceMetric)
    matcher = _CountingMatcher()
    results = EvaluationEngine(k_values=[1, 2], evidence_matcher=matcher, evidence_threshold=0.5).evaluate(
        _sample(["gold evidence"], ["first chunk", "second chunk"])
    )

    assert matcher.calls == 1
    assert {(result.name, result.k) for result in results} >= {
        ("evidence_test_metric", 1),
        ("evidence_test_metric", 2),
    }