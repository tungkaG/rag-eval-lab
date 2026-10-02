from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_eval_lab.embedding.sentence_transformers import SentenceTransformerEmbedding


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.encode_document.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    model.encode_query.return_value = np.array([[0.5, 0.5]])
    return model


def _embedding(mock_model, **kwargs):
    with patch(
        "rag_eval_lab.embedding.sentence_transformers.SentenceTransformer",
        return_value=mock_model,
    ):
        return SentenceTransformerEmbedding(model_name="fake-model", **kwargs)


def test_embed_documents_dispatches_to_encode_document(mock_model):
    embedding = _embedding(mock_model)

    result = embedding.embed_documents(["doc one", "doc two"])

    mock_model.encode_document.assert_called_once()
    assert mock_model.encode_document.call_args.args[0] == ["doc one", "doc two"]
    mock_model.encode_query.assert_not_called()


def test_embed_queries_dispatches_to_encode_query(mock_model):
    embedding = _embedding(mock_model)

    embedding.embed_queries(["what is x?"])

    mock_model.encode_query.assert_called_once()
    assert mock_model.encode_query.call_args.args[0] == ["what is x?"]
    mock_model.encode_document.assert_not_called()


def test_returns_ordinary_python_lists_in_input_order(mock_model):
    embedding = _embedding(mock_model)

    result = embedding.embed_documents(["doc one", "doc two"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert isinstance(result, list)
    assert all(isinstance(vector, list) for vector in result)
    assert all(isinstance(value, float) for vector in result for value in vector)


def test_passes_batch_size_normalize_and_prompt_settings(mock_model):
    embedding = _embedding(
        mock_model,
        batch_size=8,
        document_prompt="passage: ",
        query_prompt="query: ",
    )

    embedding.embed_documents(["doc"])
    embedding.embed_queries(["q"])

    doc_kwargs = mock_model.encode_document.call_args.kwargs
    assert doc_kwargs["batch_size"] == 8
    assert doc_kwargs["normalize_embeddings"] is True
    assert doc_kwargs["prompt"] == "passage: "

    query_kwargs = mock_model.encode_query.call_args.kwargs
    assert query_kwargs["prompt"] == "query: "


def test_model_is_loaded_once_per_instance(mock_model):
    with patch(
        "rag_eval_lab.embedding.sentence_transformers.SentenceTransformer",
        return_value=mock_model,
    ) as mock_cls:
        embedding = SentenceTransformerEmbedding(model_name="fake-model", device="cpu")
        embedding.embed_documents(["a"])
        embedding.embed_queries(["b"])

    mock_cls.assert_called_once_with("fake-model", device="cpu")
