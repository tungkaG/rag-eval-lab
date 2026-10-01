import pytest

from rag_eval_lab.core.exceptions import RegistryError, UnknownStrategyError
from rag_eval_lab.core.registry import Registry


def test_register_and_create():
    registry: Registry[str] = Registry(kind="chunking")

    @registry.register("fixed_tokens")
    def _build(size: int) -> str:
        return f"fixed_tokens:{size}"

    assert registry.create("fixed_tokens", size=512) == "fixed_tokens:512"


def test_keys_returns_sorted_registered_names():
    registry: Registry[str] = Registry(kind="retrieval")
    registry.register("bm25")(lambda: "bm25")
    registry.register("dense")(lambda: "dense")

    assert registry.keys() == ["bm25", "dense"]


def test_contains():
    registry: Registry[str] = Registry(kind="retrieval")
    registry.register("dense")(lambda: "dense")

    assert "dense" in registry
    assert "bm25" not in registry


def test_duplicate_registration_raises():
    registry: Registry[str] = Registry(kind="reranking")
    registry.register("none")(lambda: "none")

    with pytest.raises(RegistryError):
        registry.register("none")(lambda: "none-again")


def test_unknown_strategy_raises():
    registry: Registry[str] = Registry(kind="embedding")

    with pytest.raises(UnknownStrategyError):
        registry.create("sentence_transformer")
