"""File processing placeholder: reserves the architecture for future source file conversion.

See `README.md` for scope and the architectural rule that file processors
must not create `Document` objects directly.
"""

from rag_eval_lab.file_processing.base import FileProcessor
from rag_eval_lab.file_processing.registry import file_processor_registry

__all__ = ["FileProcessor", "file_processor_registry"]
