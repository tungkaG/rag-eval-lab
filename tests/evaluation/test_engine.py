from rag_eval_lab.core.models import Chunk, EvaluationCase, EvaluationSample, GroundTruth, RetrievedChunk
from rag_eval_lab.evaluation.engine import EvaluationEngine


def _sample(document_relevance: dict[str, int], ranked_document_ids: list[str]) -> EvaluationSample:
    retrieved_chunks = [
        RetrievedChunk(
            chunk=Chunk(id=f"{document_id}_chunk_{rank}", document_id=document_id, text="text"),
            score=1.0,
            rank=rank,
        )
        for rank, document_id in enumerate(ranked_document_ids, start=1)
    ]
    case = EvaluationCase(
        query_id="q1",
        question="q?",
        ground_truth=GroundTruth(document_relevance=document_relevance),
    )
    return EvaluationSample(evaluation_case=case, retrieved_chunks=retrieved_chunks)


def test_evaluate_returns_results_for_each_selected_metric_and_k():
    sample = _sample({"d1": 1}, ["d1", "d2"])
    engine = EvaluationEngine(k_values=[1, 2])

    results = engine.evaluate(sample)

    assert len(results) == 10  # 5 document metrics x 2 k values
    assert all(result.k in (1, 2) for result in results)
    assert {result.name for result in results} == {
        "document_hit_rate",
        "document_recall",
        "document_precision",
        "document_mrr",
        "document_ndcg",
    }


def test_evaluate_returns_empty_list_when_ground_truth_has_no_capabilities():
    sample = _sample({}, ["d1"])

    assert EvaluationEngine().evaluate(sample) == []


def test_evaluate_uses_default_k_values():
    sample = _sample({"d1": 1}, ["d1"])

    results = EvaluationEngine().evaluate(sample)

    assert {result.k for result in results} == {1, 5, 10}


def test_evaluate_combines_retrieval_results_with_applicable_answer_evaluators():
    from rag_eval_lab.core.models import MetricResult

    class _FakeEvaluator:
        required_inputs = {"generated_answer"}

        def evaluate(self, sample):
            return [MetricResult(name="answer_fake_metric", value=1.0, k=None)]

    sample = _sample({"d1": 1}, ["d1"])
    sample.generated_answer = "an answer"

    engine = EvaluationEngine(k_values=[1], answer_evaluators=[_FakeEvaluator()])
    results = engine.evaluate(sample)

    assert any(result.name == "answer_fake_metric" for result in results)
    assert engine.answer_evaluation_seconds >= 0.0


def test_evaluate_skips_answer_evaluator_when_required_input_missing():
    class _FakeEvaluator:
        required_inputs = {"generated_answer"}

        def evaluate(self, sample):
            raise AssertionError("must not be called when generated_answer is missing")

    sample = _sample({"d1": 1}, ["d1"])  # generated_answer stays None

    engine = EvaluationEngine(k_values=[1], answer_evaluators=[_FakeEvaluator()])
    results = engine.evaluate(sample)

    assert all(not result.name.startswith("answer_") for result in results)
