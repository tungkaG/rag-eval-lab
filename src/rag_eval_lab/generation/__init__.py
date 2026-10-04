"""Optional answer generation: disabled unless an experiment's YAML enables it."""

from rag_eval_lab.generation.base import GenerationStrategy, close_generator
from rag_eval_lab.generation.gemini import GeminiGenerator
from rag_eval_lab.generation.openai import OpenAIGenerator
from rag_eval_lab.generation.prompt_builder import build_prompt
from rag_eval_lab.generation.registry import generation_registry

__all__ = [
    "GenerationStrategy",
    "GeminiGenerator",
    "OpenAIGenerator",
    "build_prompt",
    "close_generator",
    "generation_registry",
]
