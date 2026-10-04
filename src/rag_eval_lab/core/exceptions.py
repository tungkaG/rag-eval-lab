"""Custom exception hierarchy used across rag_eval_lab.

Every error raised by this project should subclass :class:`RagEvalLabError` so
callers can catch project errors without catching unrelated stdlib exceptions.
"""

from __future__ import annotations


class RagEvalLabError(Exception):
    """Base class for all rag_eval_lab errors."""


class ValidationError(RagEvalLabError):
    """Raised when a core model is constructed with invalid data."""


class RegistryError(RagEvalLabError):
    """Raised for strategy registry registration or lookup failures."""


class UnknownStrategyError(RegistryError):
    """Raised when a registry is asked to create an unregistered strategy name."""


class ConfigurationError(RagEvalLabError):
    """Raised when experiment or strategy configuration is invalid."""


class GenerationError(RagEvalLabError):
    """Raised when an answer generation strategy fails to produce a usable answer."""
