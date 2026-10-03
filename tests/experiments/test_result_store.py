"""Unit tests for the JSON `ResultStore` implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_lab.core.exceptions import RagEvalLabError
from rag_eval_lab.core.models import ExperimentResult, MetricResult
from rag_eval_lab.experiments.result_store import JsonResultStore


def test_save_and_load_round_trips_experiment_result(tmp_path: Path) -> None:
    store = JsonResultStore(tmp_path)
    result = ExperimentResult(
        experiment_id="exp-123",
        config={"experiment": {"name": "test"}},
        metrics=[
            MetricResult(name="document_hit_rate", value=0.75, k=5),
            MetricResult(name="document_mrr", value=1.0, k=None),
        ],
        timings={"total_seconds": 1.23},
        metadata={"name": "test", "document_count": 3},
    )

    store.save(result)
    loaded = store.load("exp-123")

    assert loaded == result


def test_save_writes_one_json_file_per_experiment_id(tmp_path: Path) -> None:
    store = JsonResultStore(tmp_path)
    store.save(ExperimentResult(experiment_id="exp-abc", config={}))

    assert (tmp_path / "exp-abc.json").exists()


def test_list_experiments_returns_sorted_ids(tmp_path: Path) -> None:
    store = JsonResultStore(tmp_path)
    store.save(ExperimentResult(experiment_id="b", config={}))
    store.save(ExperimentResult(experiment_id="a", config={}))

    assert store.list_experiments() == ["a", "b"]


def test_list_experiments_empty_when_directory_missing(tmp_path: Path) -> None:
    store = JsonResultStore(tmp_path / "does_not_exist")

    assert store.list_experiments() == []


def test_load_raises_for_unknown_experiment_id(tmp_path: Path) -> None:
    store = JsonResultStore(tmp_path)

    with pytest.raises(RagEvalLabError):
        store.load("missing")
