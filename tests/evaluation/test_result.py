from rag_eval_lab.core.models import MetricResult
from rag_eval_lab.evaluation.result import group_by_name


def test_groups_results_by_name_preserving_order():
    results = [
        MetricResult(name="document_hit_rate", value=1.0, k=1),
        MetricResult(name="document_recall", value=0.5, k=1),
        MetricResult(name="document_hit_rate", value=0.0, k=5),
    ]

    grouped = group_by_name(results)

    assert grouped["document_hit_rate"] == [results[0], results[2]]
    assert grouped["document_recall"] == [results[1]]


def test_groups_empty_results():
    assert group_by_name([]) == {}
