"""A generic, reusable name -> factory registry.

Every pluggable layer (ingestion, chunking, embedding, indexing, retrieval,
reranking, datasets, evaluation metrics, result stores, ...) gets its own
`Registry` instance instead of a bespoke implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from rag_eval_lab.core.exceptions import RegistryError, UnknownStrategyError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string keys to factories that build strategy instances."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers `factory` under `name`."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise RegistryError(f"{self._kind} '{name}' is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Builds a strategy instance registered under `name`."""
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise UnknownStrategyError(
                f"Unknown {self._kind} strategy '{name}'. Available: {self.keys()}"
            ) from exc
        return factory(*args, **kwargs)

    def keys(self) -> list[str]:
        """Returns the sorted list of registered strategy names."""
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories
