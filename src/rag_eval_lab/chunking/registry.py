"""Registry of chunking strategies, keyed by strategy name."""

from __future__ import annotations

from rag_eval_lab.chunking.fixed_tokens import FixedTokenChunker
from rag_eval_lab.chunking.recursive import RecursiveChunker
from rag_eval_lab.core.protocols import ChunkingStrategy
from rag_eval_lab.core.registry import Registry

chunking_registry: Registry[ChunkingStrategy] = Registry(kind="chunking")

chunking_registry.register("fixed_tokens")(FixedTokenChunker)
chunking_registry.register("recursive")(RecursiveChunker)
