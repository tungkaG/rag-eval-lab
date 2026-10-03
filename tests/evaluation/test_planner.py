from rag_eval_lab.core.models import GroundTruth, PageReference
from rag_eval_lab.evaluation.planner import plan_metrics


def test_selects_only_document_metrics_when_only_document_relevance_available():
    ground_truth = GroundTruth(document_relevance={"d1": 1})

    metrics = plan_metrics(ground_truth, k_values=[1, 5])

    assert len(metrics) == 10  # 5 document metrics x 2 k values
    assert all(metric.required_capabilities == {"document_relevance"} for metric in metrics)
    assert {metric.k for metric in metrics} == {1, 5}


def test_adds_page_metrics_when_page_references_available():
    ground_truth = GroundTruth(
        document_relevance={"d1": 1},
        page_references=[PageReference(document_id="d1", page_start=1, page_end=1)],
    )

    metrics = plan_metrics(ground_truth, k_values=[10])

    document_metrics = [m for m in metrics if m.required_capabilities == {"document_relevance"}]
    page_metrics = [m for m in metrics if m.required_capabilities == {"page_references"}]
    assert len(document_metrics) == 5
    assert len(page_metrics) == 3
    assert len(metrics) == 8


def test_returns_empty_list_when_no_capabilities_available():
    assert plan_metrics(GroundTruth()) == []


def test_default_k_values_are_one_five_and_ten():
    ground_truth = GroundTruth(document_relevance={"d1": 1})

    metrics = plan_metrics(ground_truth)

    assert {metric.k for metric in metrics} == {1, 5, 10}
