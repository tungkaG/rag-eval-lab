"""Optional answer generation: disabled unless an experiment's YAML enables it."""

from rag_eval_lab.generation.base import GenerationStrategy, close_generator
from rag_eval_lab.generation.generator import RagGenerator
from rag_eval_lab.generation.prompt_builder import build_prompt
from rag_eval_lab.generation.providers.registry import provider_registry

__all__ = [
    "GenerationStrategy",
    "RagGenerator",
    "build_prompt",
    "close_generator",
    "provider_registry",
]
