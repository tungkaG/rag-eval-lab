"""CLI entry point: `python -m rag_eval_lab.experiments <config.yaml>`."""

from __future__ import annotations

import sys
from pathlib import Path

from rag_eval_lab.experiments.config import load_config
from rag_eval_lab.experiments.runner import ExperimentRunner


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python -m rag_eval_lab.experiments <config.yaml>", file=sys.stderr)
        return 2

    config = load_config(argv[0])
    result = ExperimentRunner().run(config)

    result_path = Path(config.output.directory) / f"{result.experiment_id}.json"
    print(f"Saved experiment result: {result_path}")
    print(f"Experiment: {result.metadata.get('name')} ({result.experiment_id})")
    for metric in sorted(result.metrics, key=lambda m: (m.name, m.k if m.k is not None else -1)):
        print(f"  {metric.name}@{metric.k} = {metric.value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
