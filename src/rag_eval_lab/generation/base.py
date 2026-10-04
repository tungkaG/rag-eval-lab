"""Generation strategies all implement the shared `GenerationStrategy` protocol from core.

`close_generator` is a small escape hatch for generators that hold a
long-lived client (e.g. `GeminiGenerator`) without adding `close()` to the
shared protocol itself.
"""

from __future__ import annotations

from rag_eval_lab.core.protocols import GenerationStrategy

__all__ = ["GenerationStrategy", "close_generator"]


def close_generator(generator: GenerationStrategy) -> None:
    """Calls `generator.close()` if it defines one; a no-op otherwise."""
    close = getattr(generator, "close", None)
    if callable(close):
        close()
