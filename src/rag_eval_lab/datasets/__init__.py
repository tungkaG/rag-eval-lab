"""Dataset adapters: normalize custom and BEIR datasets into core models."""

from rag_eval_lab.datasets.beir_adapter import BeirDatasetAdapter
from rag_eval_lab.datasets.custom_dataset import CustomDatasetAdapter
from rag_eval_lab.datasets.registry import dataset_registry

__all__ = ["BeirDatasetAdapter", "CustomDatasetAdapter", "dataset_registry"]
