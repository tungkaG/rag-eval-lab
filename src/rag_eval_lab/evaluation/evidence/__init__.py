"""Evidence matching strategies and evidence retrieval metrics."""

from rag_eval_lab.evaluation.evidence.base import EvidenceMatcher
from rag_eval_lab.evaluation.evidence.registry import evidence_matcher_registry, evidence_metric_registry

__all__ = ["EvidenceMatcher", "evidence_matcher_registry", "evidence_metric_registry"]