from rag_eval_lab.evaluation.registry import metric_registry


def test_registered_metric_names():
    assert metric_registry.keys() == [
        "document_hit_rate",
        "document_mrr",
        "document_ndcg",
        "document_precision",
        "document_recall",
        "page_hit_rate",
        "page_precision",
        "page_recall",
    ]


def test_creates_metric_with_k():
    metric = metric_registry.create("document_hit_rate", k=10)

    assert metric.k == 10
    assert metric.required_capabilities == {"document_relevance"}


def test_creates_page_metric_with_k():
    metric = metric_registry.create("page_recall", k=3)

    assert metric.k == 3
    assert metric.required_capabilities == {"page_references"}
