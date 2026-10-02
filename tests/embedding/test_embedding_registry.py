from unittest.mock import patch

from rag_eval_lab.embedding.registry import embedding_registry
from rag_eval_lab.embedding.sentence_transformers import SentenceTransformerEmbedding


def test_registered_embedding_strategy_names():
    assert embedding_registry.keys() == ["sentence_transformer"]


def test_creates_sentence_transformer_embedding_with_constructor_arguments():
    with patch("rag_eval_lab.embedding.sentence_transformers.SentenceTransformer"):
        embedding = embedding_registry.create("sentence_transformer", model_name="fake-model")

    assert isinstance(embedding, SentenceTransformerEmbedding)
    assert embedding.model_name == "fake-model"
