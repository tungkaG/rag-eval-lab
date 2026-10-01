"""Generic ingestion: normalized TXT, Markdown, and JSONL loaders into `Document`s."""

from rag_eval_lab.ingestion.jsonl_loader import JsonlLoader
from rag_eval_lab.ingestion.markdown_loader import MarkdownLoader
from rag_eval_lab.ingestion.registry import ingestion_registry
from rag_eval_lab.ingestion.txt_loader import TxtLoader

__all__ = ["JsonlLoader", "MarkdownLoader", "TxtLoader", "ingestion_registry"]
