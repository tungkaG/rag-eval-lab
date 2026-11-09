"""Dataset adapters: normalize custom, BEIR, and SQuAD datasets into core models."""

from rag_eval_lab.datasets.beir_adapter import BeirDatasetAdapter
from rag_eval_lab.datasets.bq40z80_adapter import Bq40z80DatasetAdapter
from rag_eval_lab.datasets.custom_dataset import CustomDatasetAdapter
from rag_eval_lab.datasets.musique_adapter import MusiqueDatasetAdapter
from rag_eval_lab.datasets.registry import dataset_registry
from rag_eval_lab.datasets.squad_adapter import SquadDatasetAdapter

__all__ = ["BeirDatasetAdapter", "Bq40z80DatasetAdapter", "CustomDatasetAdapter", "MusiqueDatasetAdapter", "SquadDatasetAdapter", "dataset_registry"]
