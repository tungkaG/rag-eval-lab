"""Reciprocal Rank Fusion: merges ranked result lists independent of any retriever implementation."""

from __future__ import annotations

from rag_eval_lab.core.models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuses multiple ranked `RetrievedChunk` lists by chunk id using RRF.

    For each chunk, sums `1 / (rrf_k + rank)` across every input list it
    appears in (ranks start at 1). Chunks are merged by `chunk.id`, so
    different chunks from the same parent document remain separate. Input
    lists are read only, never modified. Results are sorted by descending
    fused score, breaking ties by ascending `chunk.id`.
    """
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + result.rank)
            chunks_by_id.setdefault(chunk_id, result.chunk)

    ranked_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))

    return [
        RetrievedChunk(chunk=chunks_by_id[chunk_id], score=fused_scores[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1)
    ]
