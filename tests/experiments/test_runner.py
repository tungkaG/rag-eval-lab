"""Unit tests for `ExperimentRunner` orchestration using a tiny local dataset.

These tests use the real BM25/fixed-token-chunking/no-reranker pipeline so no
model downloads or network access are required; dense/hybrid retrieval (which
would need a `sentence-transformers` model download) is only exercised for
its config-validation failure path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval_lab.core.exceptions import ConfigurationError
from rag_eval_lab.experiments.config import load_config
from rag_eval_lab.experiments.result_store import JsonResultStore
from rag_eval_lab.experiments.runner import ExperimentRunner

_CORPUS_RECORDS = [
    {"id": "doc1", "text": "The capital of France is Paris, a major city in Europe."},
    {"id": "doc2", "text": "Mount Everest is the tallest mountain in the world."},
    {"id": "doc3", "text": "Python is a popular programming language for data science."},
]

_EVALUATION_RECORDS = [
    {
        "query_id": "q1",
        "question": "What is the capital of France?",
        "ground_truth": {"document_relevance": {"doc1": 1}},
    },
    {
        "query_id": "q2",
        "question": "What is the tallest mountain?",
        "ground_truth": {"document_relevance": {"doc2": 1}},
    },
]


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def _write_bm25_config(tmp_path: Path, corpus_path: Path, evaluation_path: Path, results_dir: Path) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
experiment:
  name: bm25_smoke_test

dataset:
  adapter: custom
  params:
    corpus_path: {corpus_path.as_posix()}
    evaluation_path: {evaluation_path.as_posix()}
    corpus_format: jsonl

chunking:
  strategy: fixed_tokens
  params:
    chunk_size: 256
    overlap: 32

indexing:
  sparse:
    strategy: bm25
    params: {{}}

retrieval:
  strategy: bm25
  top_k: 10
  params: {{}}

reranking:
  strategy: none
  final_k: 5
  params: {{}}

evaluation:
  k_values: [1, 5]

output:
  directory: {results_dir.as_posix()}
""",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def bm25_config(tmp_path: Path):
    corpus_path = _write_jsonl(tmp_path / "corpus.jsonl", _CORPUS_RECORDS)
    evaluation_path = _write_jsonl(tmp_path / "evaluation.jsonl", _EVALUATION_RECORDS)
    results_dir = tmp_path / "results"
    config_path = _write_bm25_config(tmp_path, corpus_path, evaluation_path, results_dir)
    return load_config(config_path), results_dir


def test_bm25_experiment_runs_end_to_end_and_averages_metrics(bm25_config) -> None:
    config, results_dir = bm25_config

    result = ExperimentRunner(result_store=JsonResultStore(results_dir)).run(config)

    assert result.metadata["name"] == "bm25_smoke_test"
    assert result.metadata["document_count"] == 3
    assert result.metadata["chunk_count"] == 3
    assert result.metadata["query_count"] == 2
    assert "software_versions" in result.metadata
    assert "timestamp" in result.metadata

    names_and_ks = {(metric.name, metric.k) for metric in result.metrics}
    assert ("document_hit_rate", 1) in names_and_ks
    assert ("document_hit_rate", 5) in names_and_ks

    # Both queries' only chunk corpus is tiny (3 chunks total), so the
    # relevant document is always within the top 5 results regardless of
    # exact BM25 ranking -- this assertion verifies averaging, not ranking.
    hit_rate_at_5 = next(m for m in result.metrics if m.name == "document_hit_rate" and m.k == 5)
    assert hit_rate_at_5.value == pytest.approx(1.0)

    for expected_timing in (
        "dataset_loading_seconds",
        "chunking_seconds",
        "index_building_seconds",
        "retrieval_seconds",
        "reranking_seconds",
        "evaluation_seconds",
        "total_seconds",
    ):
        assert expected_timing in result.timings

    # BM25-only experiments must never create an embedding model.
    assert "corpus_embedding_seconds" not in result.timings

    saved_path = results_dir / f"{result.experiment_id}.json"
    assert saved_path.exists()


def test_bm25_experiment_does_not_create_an_embedder(bm25_config, monkeypatch: pytest.MonkeyPatch) -> None:
    config, results_dir = bm25_config

    from rag_eval_lab.retrieval import builders as builders_module

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("embedding_registry.create must not be called for a BM25-only experiment")

    monkeypatch.setattr(builders_module.embedding_registry, "create", _fail)

    ExperimentRunner(result_store=JsonResultStore(results_dir)).run(config)


def test_runner_without_explicit_store_saves_to_configured_output_directory(bm25_config) -> None:
    config, results_dir = bm25_config

    result = ExperimentRunner().run(config)

    assert (results_dir / f"{result.experiment_id}.json").exists()


def test_runner_uses_explicitly_injected_result_store(bm25_config) -> None:
    config, results_dir = bm25_config

    class _RecordingResultStore:
        def __init__(self) -> None:
            self.saved: list = []

        def save(self, result) -> None:
            self.saved.append(result)

        def load(self, experiment_id: str):
            raise NotImplementedError

        def list_experiments(self) -> list[str]:
            return [result.experiment_id for result in self.saved]

    store = _RecordingResultStore()

    result = ExperimentRunner(result_store=store).run(config)

    assert store.saved == [result]
    # The injected store was used instead of a JsonResultStore, so nothing
    # was written under the config's output directory.
    assert not results_dir.exists()


def test_additional_retrieval_builder_can_be_registered_and_run_without_modifying_runner(
    bm25_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, results_dir = bm25_config

    from dataclasses import replace

    from rag_eval_lab.core.models import RetrievedChunk
    from rag_eval_lab.retrieval.builders import PreparedRetrieval, retrieval_builder_registry

    class _FakeRetriever:
        def retrieve(self, query, top_k) -> list[RetrievedChunk]:
            return []

    class _FakeRetrievalBuilder:
        def build(self, config, chunks) -> PreparedRetrieval:
            return PreparedRetrieval(retriever=_FakeRetriever(), timings={"fake_preparation_seconds": 0.0})

    monkeypatch.setitem(retrieval_builder_registry._factories, "fake_strategy", _FakeRetrievalBuilder)
    fake_config = replace(config, retrieval=replace(config.retrieval, strategy="fake_strategy"))

    result = ExperimentRunner(result_store=JsonResultStore(results_dir)).run(fake_config)

    assert result.timings["fake_preparation_seconds"] == 0.0
    hit_rate_at_5 = next(m for m in result.metrics if m.name == "document_hit_rate" and m.k == 5)
    assert hit_rate_at_5.value == pytest.approx(0.0)  # the fake retriever never returns a chunk


def test_dense_retrieval_without_embedding_section_raises_configuration_error(tmp_path: Path) -> None:
    corpus_path = _write_jsonl(tmp_path / "corpus.jsonl", _CORPUS_RECORDS)
    evaluation_path = _write_jsonl(tmp_path / "evaluation.jsonl", _EVALUATION_RECORDS)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
experiment:
  name: missing_embedding

dataset:
  adapter: custom
  params:
    corpus_path: {corpus_path.as_posix()}
    evaluation_path: {evaluation_path.as_posix()}
    corpus_format: jsonl

chunking:
  strategy: fixed_tokens
  params:
    chunk_size: 256
    overlap: 32

retrieval:
  strategy: dense
  top_k: 10
  params: {{}}

reranking:
  strategy: none
  final_k: 5
  params: {{}}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(ConfigurationError):
        ExperimentRunner(result_store=JsonResultStore(tmp_path / "results")).run(config)


def _fake_provider_factory():
    """Returns (factory, instances) for a fake `LlmProvider` -- mirrors the factory pattern used elsewhere.

    `RagGenerator` only passes the built RAG prompt (not the query/chunks) to
    the provider, so the fake recovers the query id from the question text
    that `build_prompt` embeds in the prompt.
    """
    instances: list = []
    question_to_query_id = {record["question"]: record["query_id"] for record in _EVALUATION_RECORDS}

    class _FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.calls: list[str] = []
            self.closed = False
            instances.append(self)

        def generate(self, prompt: str, response_schema: type | None = None):
            self.calls.append(prompt)
            query_id = next(qid for question, qid in question_to_query_id.items() if question in prompt)
            return f"answer-for-{query_id}", {"provider": "fake", "model_name": self.kwargs.get("model_name")}

        def close(self) -> None:
            self.closed = True

    return _FakeProvider, instances


def test_generation_disabled_by_default_leaves_no_generation_metadata_or_timings(bm25_config) -> None:
    config, results_dir = bm25_config

    result = ExperimentRunner(result_store=JsonResultStore(results_dir)).run(config)

    assert "generation_records" not in result.metadata
    assert "generation_seconds" not in result.timings


def test_generation_enabled_populates_answers_timings_and_records(
    bm25_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, results_dir = bm25_config

    from dataclasses import replace

    from rag_eval_lab.experiments.config import GenerationConfig
    from rag_eval_lab.generation.providers.registry import provider_registry

    factory, instances = _fake_provider_factory()
    monkeypatch.setitem(provider_registry._factories, "fake_gemini", factory)

    generation_config = GenerationConfig(enabled=True, provider="fake_gemini", params={"model_name": "fake-model"})
    config_with_generation = replace(config, generation=generation_config)

    result = ExperimentRunner(result_store=JsonResultStore(results_dir)).run(config_with_generation)

    # Constructed exactly once and reused for both queries, then closed.
    assert len(instances) == 1
    assert len(instances[0].calls) == 2
    assert instances[0].closed is True

    assert "generation_seconds" in result.timings

    records = result.metadata["generation_records"]
    assert len(records) == 2
    for record in records:
        assert record.keys() == {
            "query_id",
            "question",
            "generated_answer",
            "reference_answer",
            "context_chunk_ids",
            "generation_metadata",
            "generation_seconds",
        }
        assert record["generated_answer"] == f"answer-for-{record['query_id']}"
        assert record["reference_answer"] is None  # bm25_config ground truth has no reference answer

    # Existing retrieval metrics are still computed alongside generation.
    assert any(metric.name == "document_hit_rate" for metric in result.metrics)


def test_answer_evaluation_enabled_records_per_query_results_and_closes_evaluator(
    bm25_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, results_dir = bm25_config

    from dataclasses import replace

    from rag_eval_lab.core.models import MetricResult
    from rag_eval_lab.evaluation.answer.registry import answer_evaluator_registry
    from rag_eval_lab.experiments.config import GenerationConfig, StrategyConfig
    from rag_eval_lab.generation.providers.registry import provider_registry

    provider_factory, _ = _fake_provider_factory()
    monkeypatch.setitem(provider_registry._factories, "fake_gemini", provider_factory)

    instances: list = []

    class _FakeAnswerEvaluator:
        required_inputs = {"generated_answer"}

        def __init__(self, **kwargs: object) -> None:
            self.closed = False
            instances.append(self)

        def evaluate(self, sample):
            return [MetricResult(name="answer_fake_metric", value=1.0, k=None, metadata={"note": "fake"})]

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(answer_evaluator_registry._factories, "fake_evaluator", _FakeAnswerEvaluator)

    config_with_answer_eval = replace(
        config,
        generation=GenerationConfig(enabled=True, provider="fake_gemini", params={"model_name": "fake-model"}),
        evaluation=replace(config.evaluation, answer_metrics=[StrategyConfig(strategy="fake_evaluator")]),
    )

    result = ExperimentRunner(result_store=JsonResultStore(results_dir)).run(config_with_answer_eval)

    assert len(instances) == 1  # constructed once, reused across both queries
    assert instances[0].closed is True
    assert "answer_evaluation_seconds" in result.timings

    records = result.metadata["answer_metric_records"]
    assert len(records) == 2  # one per evaluation case
    assert all(record["name"] == "answer_fake_metric" for record in records)
    assert all(record["metadata"] == {"note": "fake"} for record in records)
