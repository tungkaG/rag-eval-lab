"""Registry of reranking strategies, keyed by strategy name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import RerankingStrategy
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.reranking.cross_encoder import CrossEncoderReranker
from rag_eval_lab.reranking.none import NoReranker

reranking_registry: Registry[RerankingStrategy] = Registry(kind="reranking")

reranking_registry.register("none")(NoReranker)
reranking_registry.register("cross_encoder")(CrossEncoderReranker)
