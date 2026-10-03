from unittest.mock import patch

from rag_eval_lab.reranking.cross_encoder import CrossEncoderReranker
from rag_eval_lab.reranking.none import NoReranker
from rag_eval_lab.reranking.registry import reranking_registry


def test_registered_reranking_strategy_names():
    assert reranking_registry.keys() == ["cross_encoder", "none"]


def test_creates_none_reranker():
    assert isinstance(reranking_registry.create("none"), NoReranker)


def test_creates_cross_encoder_reranker_with_constructor_arguments():
    with patch("rag_eval_lab.reranking.cross_encoder.CrossEncoder"):
        reranker = reranking_registry.create("cross_encoder", model_name="fake-model")

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "fake-model"
