from rag_eval_lab.embedding.registry import embedding_registry
from rag_eval_lab.evaluation.evidence.matchers import EmbeddingEvidenceMatcher, LexicalEvidenceMatcher
from rag_eval_lab.evaluation.evidence.registry import evidence_matcher_registry


def test_lexical_matcher_uses_lowercase_word_token_f1() -> None:
    matcher = LexicalEvidenceMatcher()

    scores = matcher.score(["Paris, France!"], ["paris france", "Everest"])

    assert scores == [[1.0, 0.0]]


def test_embedding_matcher_uses_configured_embedding_strategy(monkeypatch) -> None:
    class _FakeEmbedder:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors = {"chunk": [1.0, 0.0], "matching evidence": [2.0, 0.0], "other evidence": [0.0, 3.0]}
            return [vectors[text] for text in texts]

    monkeypatch.setitem(embedding_registry._factories, "fake_evidence_embedding", _FakeEmbedder)
    matcher = EmbeddingEvidenceMatcher("fake_evidence_embedding", {"unused": "value"})

    scores = matcher.score(["chunk"], ["matching evidence", "other evidence"])

    assert scores == [[1.0, 0.0]]
    assert matcher.metadata == {
        "matcher": "embedding",
        "embedding_strategy": "fake_evidence_embedding",
        "embedding_params": {"unused": "value"},
    }


def test_evidence_matchers_are_selected_through_the_registry() -> None:
    matcher = evidence_matcher_registry.create("lexical")

    assert evidence_matcher_registry.keys() == ["embedding", "lexical"]
    assert isinstance(matcher, LexicalEvidenceMatcher)