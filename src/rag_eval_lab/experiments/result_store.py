"""JSON file based `ResultStore`: one `<experiment_id>.json` file per experiment.

SQLite, comparison tables, and dashboards belong to Phase 10.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_eval_lab.core.exceptions import RagEvalLabError
from rag_eval_lab.core.models import ExperimentResult, MetricResult


class JsonResultStore:
    """Saves and loads `ExperimentResult`s as plain JSON files in `directory`."""

    def __init__(self, directory: str | Path = "results") -> None:
        self.directory = Path(directory)

    def save(self, result: ExperimentResult) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{result.experiment_id}.json"
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    def load(self, experiment_id: str) -> ExperimentResult:
        path = self.directory / f"{experiment_id}.json"
        if not path.exists():
            raise RagEvalLabError(f"No stored experiment result for id '{experiment_id}'")
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        metrics = [MetricResult(**metric) for metric in payload.get("metrics", [])]
        return ExperimentResult(
            experiment_id=payload["experiment_id"],
            config=payload.get("config", {}),
            metrics=metrics,
            timings=payload.get("timings", {}),
            metadata=payload.get("metadata", {}),
        )

    def list_experiments(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json"))
