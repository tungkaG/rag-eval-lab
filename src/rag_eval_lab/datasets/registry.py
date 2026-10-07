"""Registry of dataset adapters, keyed by adapter name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import DatasetAdapter
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.datasets.beir_adapter import BeirDatasetAdapter
from rag_eval_lab.datasets.custom_dataset import CustomDatasetAdapter
from rag_eval_lab.datasets.musique_adapter import MusiqueDatasetAdapter
from rag_eval_lab.datasets.squad_adapter import SquadDatasetAdapter

dataset_registry: Registry[DatasetAdapter] = Registry(kind="dataset")

dataset_registry.register("custom")(CustomDatasetAdapter)
dataset_registry.register("beir")(BeirDatasetAdapter)
dataset_registry.register("squad")(SquadDatasetAdapter)
dataset_registry.register("musique")(MusiqueDatasetAdapter)
