"""Orchestrates one experiment run: dataset -> chunks -> retrieval -> reranking -> evaluation.

This module contains almost no algorithm-specific logic and never inspects
retrieval strategy names: the retriever and its dependencies (embedder,
indexes) are prepared by a `RetrievalBuilder` resolved from
`config.retrieval.strategy` via `retrieval_builder_registry`
(see `retrieval/builders.py`). Every other strategy is built through its
existing registry from an `ExperimentConfig`; this module only wires the
pieces together and records timings/metadata.

Answer generation (`config.generation`) is entirely optional: when disabled
(the default), no generator is constructed and no API key or SDK is
required. When enabled, a `RagGenerator` is built once -- directly, not
through a registry keyed by provider name -- and reused for every case; it
resolves its backend from `config.generation.provider` through the shared
`generation.providers.provider_registry`, so a new provider only needs a
backend implementation and a registry entry there, never a runner change.

Answer evaluation (`config.evaluation.answer_metrics`) is independently
optional: evaluators are built once via `answer_evaluator_registry` and only
run when their declared inputs are available on a given sample -- an
evaluator never fabricates missing ground truth or reports a skipped metric
as zero.
"""

from __future__ import annotations

import platform
import sys
import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import rag_eval_lab
from rag_eval_lab.chunking.registry import chunking_registry
from rag_eval_lab.core.models import EvaluationSample, ExperimentResult, MetricResult, Query
from rag_eval_lab.core.protocols import ResultStore
from rag_eval_lab.datasets.registry import dataset_registry
from rag_eval_lab.evaluation.answer.base import close_evaluator
from rag_eval_lab.evaluation.answer.registry import answer_evaluator_registry
from rag_eval_lab.evaluation.engine import EvaluationEngine
from rag_eval_lab.experiments.config import ExperimentConfig
from rag_eval_lab.experiments.result_store import JsonResultStore
from rag_eval_lab.generation.base import close_generator
from rag_eval_lab.generation.generator import RagGenerator
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

        answer_evaluators = [
            answer_evaluator_registry.create(item.strategy, **item.params)
            for item in config.evaluation.answer_metrics
        ]
        evaluation_engine = EvaluationEngine(k_values=config.evaluation.k_values, answer_evaluators=answer_evaluators)

        generator = None
        if config.generation.enabled:
            generator = RagGenerator(provider=config.generation.provider, **config.generation.params)

        metric_values: dict[tuple[str, int | None], list[float]] = {}
        retrieval_seconds = 0.0
        reranking_seconds = 0.0
        evaluation_seconds = 0.0
        generation_seconds = 0.0
        generation_records: list[dict[str, Any]] = []
        answer_metric_records: list[dict[str, Any]] = []

        try:
            for case in evaluation_cases:
                query = Query(id=case.query_id, text=case.question)

                started = perf_counter()
                retrieved_chunks = retriever.retrieve(query, config.retrieval.top_k)
                retrieval_seconds += perf_counter() - started

                started = perf_counter()
                reranked_chunks = reranker.rerank(query, retrieved_chunks, config.reranking.final_k)
                reranking_seconds += perf_counter() - started

                sample = EvaluationSample(evaluation_case=case, retrieved_chunks=reranked_chunks)

                if generator is not None:
                    started = perf_counter()
                    answer, generation_metadata = generator.generate(query, reranked_chunks)
                    sample_generation_seconds = perf_counter() - started
                    generation_seconds += sample_generation_seconds

                    sample.generated_answer = answer
                    sample.timings["generation_seconds"] = sample_generation_seconds

                    generation_records.append(
                        {
                            "query_id": case.query_id,
                            "question": case.question,
                            "generated_answer": answer,
                            "reference_answer": case.ground_truth.reference_answer,
                            "context_chunk_ids": [result.chunk.id for result in reranked_chunks],
                            "generation_metadata": generation_metadata,
                            "generation_seconds": sample_generation_seconds,
                        }
                    )

                started = perf_counter()
                for metric_result in evaluation_engine.evaluate(sample):
                    metric_values.setdefault((metric_result.name, metric_result.k), []).append(metric_result.value)
                    if metric_result.name.startswith("answer_"):
                        answer_metric_records.append(
                            {
                                "query_id": case.query_id,
                                "name": metric_result.name,
                                "value": metric_result.value,
                                "metadata": metric_result.metadata,
                            }
                        )
                evaluation_seconds += perf_counter() - started
        finally:
            if generator is not None:
                close_generator(generator)
            for evaluator in answer_evaluators:
                close_evaluator(evaluator)

        timings["retrieval_seconds"] = retrieval_seconds
        timings["reranking_seconds"] = reranking_seconds
        timings["evaluation_seconds"] = evaluation_seconds
        if config.generation.enabled:
            timings["generation_seconds"] = generation_seconds
        if answer_evaluators:
            timings["answer_evaluation_seconds"] = evaluation_engine.answer_evaluation_seconds

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
        if generation_records:
            metadata["generation_records"] = generation_records
        if answer_metric_records:
            metadata["answer_metric_records"] = answer_metric_records

        experiment_result = ExperimentResult(
            experiment_id=str(uuid.uuid4()),
            config=config.raw,
            metrics=averaged_metrics,
            timings=timings,
            metadata=metadata,
        )

        result_store.save(experiment_result)
        return experiment_result
