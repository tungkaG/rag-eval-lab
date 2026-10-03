"""Orchestrates one experiment run: dataset -> chunks -> retrieval -> reranking -> evaluation.

This module contains almost no algorithm-specific logic and never inspects
retrieval strategy names: the retriever and its dependencies (embedder,
indexes) are prepared by a `RetrievalBuilder` resolved from
`config.retrieval.strategy` via `retrieval_builder_registry`
(see `retrieval/builders.py`). Every other strategy is built through its
existing registry from an `ExperimentConfig`; this module only wires the
pieces together and records timings/metadata.
"""

from __future__ import annotations

import platform
import sys
import uuid
from datetime import datetime, timezone
from time import perf_counter

import rag_eval_lab
from rag_eval_lab.chunking.registry import chunking_registry
from rag_eval_lab.core.models import EvaluationSample, ExperimentResult, MetricResult, Query
from rag_eval_lab.core.protocols import ResultStore
from rag_eval_lab.datasets.registry import dataset_registry
from rag_eval_lab.evaluation.engine import EvaluationEngine
from rag_eval_lab.experiments.config import ExperimentConfig
from rag_eval_lab.experiments.result_store import JsonResultStore
from rag_eval_lab.reranking.registry import reranking_registry
from rag_eval_lab.retrieval.builders import retrieval_builder_registry


def _software_versions() -> dict[str, str]:
    return {
        "rag_eval_lab": rag_eval_lab.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class ExperimentRunner:
    """Runs one experiment end to end from an `ExperimentConfig`.

    Models and indexes are instantiated once per run and reused for every
    evaluation case; they are never rebuilt per query.
    """

    def __init__(self, result_store: ResultStore | None = None) -> None:
        self.result_store = result_store

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        # Falls back to the configured output directory, not a fixed default,
        # so direct usage and the CLI always write where the config says.
        result_store = self.result_store if self.result_store is not None else JsonResultStore(config.output.directory)

        timings: dict[str, float] = {}
        total_started = perf_counter()

        started = perf_counter()
        dataset = dataset_registry.create(config.dataset.adapter, **config.dataset.params)
        documents = dataset.documents()
        evaluation_cases = dataset.evaluation_cases()
        timings["dataset_loading_seconds"] = perf_counter() - started

        started = perf_counter()
        chunker = chunking_registry.create(config.chunking.strategy, **config.chunking.params)
        chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
        timings["chunking_seconds"] = perf_counter() - started

        builder = retrieval_builder_registry.create(config.retrieval.strategy)
        prepared_retrieval = builder.build(config, chunks)
        retriever = prepared_retrieval.retriever
        timings.update(prepared_retrieval.timings)

        reranker = reranking_registry.create(config.reranking.strategy, **config.reranking.params)
        evaluation_engine = EvaluationEngine(k_values=config.evaluation.k_values)

        metric_values: dict[tuple[str, int | None], list[float]] = {}
        retrieval_seconds = 0.0
        reranking_seconds = 0.0
        evaluation_seconds = 0.0

        for case in evaluation_cases:
            query = Query(id=case.query_id, text=case.question)

            started = perf_counter()
            retrieved_chunks = retriever.retrieve(query, config.retrieval.top_k)
            retrieval_seconds += perf_counter() - started

            started = perf_counter()
            reranked_chunks = reranker.rerank(query, retrieved_chunks, config.reranking.final_k)
            reranking_seconds += perf_counter() - started

            sample = EvaluationSample(evaluation_case=case, retrieved_chunks=reranked_chunks)

            started = perf_counter()
            for metric_result in evaluation_engine.evaluate(sample):
                metric_values.setdefault((metric_result.name, metric_result.k), []).append(metric_result.value)
            evaluation_seconds += perf_counter() - started

        timings["retrieval_seconds"] = retrieval_seconds
        timings["reranking_seconds"] = reranking_seconds
        timings["evaluation_seconds"] = evaluation_seconds

        averaged_metrics = [
            MetricResult(name=name, value=sum(values) / len(values), k=k)
            for (name, k), values in metric_values.items()
        ]

        timings["total_seconds"] = perf_counter() - total_started

        metadata = {
            "name": config.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "query_count": len(evaluation_cases),
            "software_versions": _software_versions(),
        }
        metadata.update(prepared_retrieval.metadata)

        experiment_result = ExperimentResult(
            experiment_id=str(uuid.uuid4()),
            config=config.raw,
            metrics=averaged_metrics,
            timings=timings,
            metadata=metadata,
        )

        result_store.save(experiment_result)
        return experiment_result
