from rag_eval_lab.core.models import Chunk, Document, EvaluationSample, MetricResult, Query, RetrievedChunk
from rag_eval_lab.core.protocols import (
    ChunkingStrategy,
    DocumentLoader,
    EmbeddingStrategy,
    Metric,
    RetrievalStrategy,
)


class _InMemoryLoader:
    def load(self, source: list[str]) -> list[Document]:
        return [Document(id=str(i), text=t) for i, t in enumerate(source)]


class _FixedSizeChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        return [Chunk(id=f"{document.id}_0", document_id=document.id, text=document.text)]


class _DummyRetriever:
    def retrieve(self, query: Query, top_k: int) -> list[RetrievedChunk]:
        chunk = Chunk(id="c", document_id="d", text="t")
        return [RetrievedChunk(chunk=chunk, score=1.0, rank=0)][:top_k]


class _SymmetricEmbedder:
    """Uses the same underlying call for both documents and queries."""

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


class _DummyMetric:
    required_capabilities = {"document_relevance"}

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        return MetricResult(name="dummy", value=1.0)


def test_document_loader_protocol_conformance():
    assert isinstance(_InMemoryLoader(), DocumentLoader)


def test_chunking_strategy_protocol_conformance():
    assert isinstance(_FixedSizeChunker(), ChunkingStrategy)


def test_retrieval_strategy_protocol_conformance():
    assert isinstance(_DummyRetriever(), RetrievalStrategy)


def test_embedding_strategy_protocol_conformance():
    assert isinstance(_SymmetricEmbedder(), EmbeddingStrategy)


def test_embedding_strategy_preserves_input_order():
    embedder = _SymmetricEmbedder()
    texts = ["a", "bb", "ccc"]
    assert embedder.embed_documents(texts) == [[1.0], [2.0], [3.0]]
    assert embedder.embed_queries(texts) == [[1.0], [2.0], [3.0]]


def test_metric_protocol_conformance():
    assert isinstance(_DummyMetric(), Metric)


def test_non_conforming_object_fails_protocol_check():
    assert not isinstance(object(), DocumentLoader)

