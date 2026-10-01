"""Registry of ingestion loaders, keyed by normalized input format name."""

from __future__ import annotations

from rag_eval_lab.core.protocols import DocumentLoader
from rag_eval_lab.core.registry import Registry
from rag_eval_lab.ingestion.jsonl_loader import JsonlLoader
from rag_eval_lab.ingestion.markdown_loader import MarkdownLoader
from rag_eval_lab.ingestion.txt_loader import TxtLoader

ingestion_registry: Registry[DocumentLoader] = Registry(kind="ingestion")

ingestion_registry.register("txt")(TxtLoader)
ingestion_registry.register("markdown")(MarkdownLoader)
ingestion_registry.register("jsonl")(JsonlLoader)
