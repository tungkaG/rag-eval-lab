"""Fixed-size token window chunking, wrapping `langchain_text_splitters.TokenTextSplitter`."""

from __future__ import annotations

from langchain_text_splitters import TokenTextSplitter

from rag_eval_lab.core.models import Chunk, Document

STRATEGY_NAME = "fixed_tokens"


class FixedTokenChunker:
    """Splits a `Document` into fixed-size token windows with overlap.

    `chunk_size` and `overlap` are both measured in tokens, using the
    `encoding_name` tokenizer. Only internal `Chunk` objects cross this
    class's boundary; `TokenTextSplitter` stays private.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding_name = encoding_name
        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            encoding_name=encoding_name,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self._splitter.split_text(document.text)
        return [self._to_chunk(document, index, text) for index, text in enumerate(texts)]

    def _to_chunk(self, document: Document, index: int, text: str) -> Chunk:
        metadata = {
            **document.metadata,
            "strategy": STRATEGY_NAME,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "encoding_name": self.encoding_name,
        }
        return Chunk(
            id=f"{document.id}_chunk_{index}",
            document_id=document.id,
            text=text,
            page_start=document.metadata.get("page_start"),
            page_end=document.metadata.get("page_end"),
            metadata=metadata,
        )
