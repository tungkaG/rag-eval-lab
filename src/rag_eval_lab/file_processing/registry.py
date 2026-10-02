"""Registry of file processors, keyed by processor name.

Empty for now; future phases register concrete processors here (e.g. PDF via
Azure Document Intelligence, DOCX, XLSX, HTML).
"""

from __future__ import annotations

from rag_eval_lab.core.registry import Registry
from rag_eval_lab.file_processing.base import FileProcessor

file_processor_registry: Registry[FileProcessor] = Registry(kind="file_processor")
