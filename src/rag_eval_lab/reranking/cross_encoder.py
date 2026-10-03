"""Cross-encoder reranking, wrapping `sentence_transformers.CrossEncoder`."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_eval_lab.core.models import Query, RetrievedChunk


class CrossEncoderReranker:
    """Rescores a candidate list with a cross-encoder over `(query, chunk text)` pairs.

    Loads the model once per instance. Does not retrieve candidates itself;
    the caller decides how many candidates to pass in versus `top_k`.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: Query,
        retrieved_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not retrieved_chunks:
            return []

        pairs = [(query.text, result.chunk.text) for result in retrieved_chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(
            enumerate(retrieved_chunks),
            key=lambda indexed: (-float(scores[indexed[0]]), indexed[0]),
        )

        return [
            RetrievedChunk(
                chunk=result.chunk,
                score=float(scores[index]),
                rank=rank,
                retriever_metadata=dict(result.retriever_metadata),
            )
            for rank, (index, result) in enumerate(ranked[:top_k], start=1)
        ]
