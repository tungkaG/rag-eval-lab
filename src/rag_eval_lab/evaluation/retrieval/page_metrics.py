"""Page-level retrieval metrics.

Operates directly on the top-`k` retrieved chunks (not deduplicated by
parent document, unlike the document metrics in `document_metrics.py`).
Pages are represented as unique `(document_id, page_number)` pairs; gold and
retrieved page ranges are expanded into these pairs, inclusive of both
endpoints. Matching requires the same document and page number, which
implements page range overlap. Chunks without a page range are ignored.
"""

from __future__ import annotations

from rag_eval_lab.core.models import EvaluationSample, MetricResult, PageReference, RetrievedChunk

Page = tuple[str, int]


def _expand_pages(document_id: str, page_start: int, page_end: int) -> set[Page]:
    return {(document_id, page) for page in range(page_start, page_end + 1)}


def _gold_pages(page_references: list[PageReference]) -> set[Page]:
    pages: set[Page] = set()
    for reference in page_references:
        pages |= _expand_pages(reference.document_id, reference.page_start, reference.page_end)
    return pages


def _predicted_pages(retrieved_chunks: list[RetrievedChunk], k: int) -> set[Page]:
    top_k = sorted(retrieved_chunks, key=lambda result: result.rank)[:k]
    pages: set[Page] = set()
    for result in top_k:
        chunk = result.chunk
        if chunk.page_start is None or chunk.page_end is None:
            continue
        pages |= _expand_pages(chunk.document_id, chunk.page_start, chunk.page_end)
    return pages


class PageHitRateAtK:
    """1 if any predicted page (within the top `k` chunks) matches a gold page, else 0."""

    required_capabilities = {"page_references"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        gold = _gold_pages(sample.evaluation_case.ground_truth.page_references)
        predicted = _predicted_pages(sample.retrieved_chunks, self.k)
        hit = 1.0 if gold & predicted else 0.0
        return MetricResult(name="page_hit_rate", value=hit, k=self.k)


class PageRecallAtK:
    """Matching pages divided by gold pages; 0 when there are no gold pages."""

    required_capabilities = {"page_references"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        gold = _gold_pages(sample.evaluation_case.ground_truth.page_references)
        if not gold:
            return MetricResult(name="page_recall", value=0.0, k=self.k)
        predicted = _predicted_pages(sample.retrieved_chunks, self.k)
        return MetricResult(name="page_recall", value=len(gold & predicted) / len(gold), k=self.k)


class PagePrecisionAtK:
    """Matching pages divided by predicted pages; 0 when nothing was predicted."""

    required_capabilities = {"page_references"}

    def __init__(self, k: int) -> None:
        self.k = k

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        predicted = _predicted_pages(sample.retrieved_chunks, self.k)
        if not predicted:
            return MetricResult(name="page_precision", value=0.0, k=self.k)
        gold = _gold_pages(sample.evaluation_case.ground_truth.page_references)
        return MetricResult(name="page_precision", value=len(gold & predicted) / len(predicted), k=self.k)
