"""Dense embedding, wrapping `sentence_transformers.SentenceTransformer`.

`SentenceTransformer` and its NumPy/tensor outputs stay internal; only plain
`list[list[float]]` crosses this class's boundary.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedding:
    """Embeds documents and queries with a Sentence Transformers model.

    Loads the model once per instance. Documents are encoded with
    `encode_document`, queries with `encode_query`, so asymmetric models
    (e.g. instructions required only on the query side) are handled
    correctly. Some models require an explicit instruction prompt on one or
    both sides; `document_prompt` / `query_prompt` are passed straight
    through to the corresponding `encode_*` call when set.

    `cl100k_base` chunk counts (used by the chunking layer) do not
    correspond to this model's own tokenizer; inputs longer than the
    model's max sequence length may be truncated by the model itself.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        document_prompt: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.document_prompt = document_prompt
        self.query_prompt = query_prompt
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode_document(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            prompt=self.document_prompt,
        )
        return embeddings.tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode_query(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            prompt=self.query_prompt,
        )
        return embeddings.tolist()
