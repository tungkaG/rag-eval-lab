"""Registry of retrieval evaluation metrics, keyed by metric name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import Metric
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.evaluation.retrieval.document_metrics import (
    DocumentHitRateAtK,
    DocumentMRRAtK,
    DocumentNDCGAtK,
    DocumentPrecisionAtK,
    DocumentRecallAtK,
)
from rag_eval_lab.evaluation.retrieval.page_metrics import PageHitRateAtK, PagePrecisionAtK, PageRecallAtK

metric_registry: Registry[Metric] = Registry(kind="metric")

metric_registry.register("document_hit_rate")(DocumentHitRateAtK)
metric_registry.register("document_recall")(DocumentRecallAtK)
metric_registry.register("document_precision")(DocumentPrecisionAtK)
metric_registry.register("document_mrr")(DocumentMRRAtK)
metric_registry.register("document_ndcg")(DocumentNDCGAtK)
metric_registry.register("page_hit_rate")(PageHitRateAtK)
metric_registry.register("page_recall")(PageRecallAtK)
metric_registry.register("page_precision")(PagePrecisionAtK)
