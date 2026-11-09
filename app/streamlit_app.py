"""Streamlit demo: ask questions against the BQ40Z80 datasheet through the configured RAG pipeline.

Run from the repository root: `streamlit run app/streamlit_app.py`.
"""

from __future__ import annotations

import re
from dataclasses import replace
from time import perf_counter

import streamlit as st

from rag_eval_lab.chunking.registry import chunking_registry
from rag_eval_lab.core.exceptions import GenerationError
from rag_eval_lab.core.models import Query, RetrievedChunk
from rag_eval_lab.core.paths import REPO_ROOT
from rag_eval_lab.datasets.registry import dataset_registry
from rag_eval_lab.experiments.config import load_config
from rag_eval_lab.generation.generator import RagGenerator
from rag_eval_lab.reranking.registry import reranking_registry
from rag_eval_lab.retrieval.builders import retrieval_builder_registry

CONFIG = load_config(REPO_ROOT / "configs" / "demo_bq40z80.yaml")


@st.cache_resource(show_spinner="Loading and chunking the datasheet...")
def load_chunks():
    dataset = dataset_registry.create(CONFIG.dataset.adapter, **CONFIG.dataset.params)
    chunker = chunking_registry.create(CONFIG.chunking.strategy, **CONFIG.chunking.params)
    return [chunk for document in dataset.documents() for chunk in chunker.chunk(document)]


@st.cache_resource(show_spinner="Building the index...")
def load_retriever(strategy: str):
    config = replace(CONFIG, retrieval=replace(CONFIG.retrieval, strategy=strategy))
    return retrieval_builder_registry.create(strategy).build(config, load_chunks()).retriever


@st.cache_resource(show_spinner="Loading the reranker...")
def load_reranker(strategy: str):
    params = CONFIG.reranking.params if strategy == CONFIG.reranking.strategy else {}
    return reranking_registry.create(strategy, **params)


@st.cache_resource(show_spinner="Loading the LLM...")
def load_generator():
    return RagGenerator(provider=CONFIG.generation.provider, **CONFIG.generation.params)


def chunk_rows(results: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "rank": result.rank,
            "score": round(result.score, 4),
            "section": result.chunk.metadata.get("section"),
            "title": result.chunk.metadata.get("title"),
            "pages": f"{result.chunk.page_start}-{result.chunk.page_end}",
            "chunk_id": result.chunk.id,
            "text": result.chunk.text,
        }
        for result in results
    ]


st.set_page_config(page_title="BQ40Z80 RAG demo", layout="wide")
st.title("BQ40Z80 datasheet assistant")

with st.sidebar:
    st.header("Settings")
    retrieval_strategy = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    top_k = st.slider("Candidates (top_k)", 5, 100, CONFIG.retrieval.top_k)
    use_reranker = st.toggle("Cross-encoder reranking", value=True)
    final_k = st.slider("Context chunks (final_k)", 1, 10, CONFIG.reranking.final_k)

question = st.text_input("Question", placeholder="What happens when cell overvoltage is detected?")

if st.button("Ask", type="primary") and question.strip():
    query = Query(id="demo", text=question.strip())
    retriever = load_retriever(retrieval_strategy)
    reranker = load_reranker(CONFIG.reranking.strategy if use_reranker else "none")
    generator = load_generator()
    timings: dict[str, float] = {}

    started = perf_counter()
    candidates = retriever.retrieve(query, top_k)
    timings["retrieval_seconds"] = perf_counter() - started

    started = perf_counter()
    final_chunks = reranker.rerank(query, candidates, final_k)
    timings["reranking_seconds"] = perf_counter() - started

    with st.spinner("Generating answer..."):
        started = perf_counter()
        try:
            answer, generation_metadata = generator.generate(query, final_chunks)
        except GenerationError as exc:
            answer, generation_metadata = None, {"error": str(exc)}
        timings["generation_seconds"] = perf_counter() - started

    st.subheader("Answer")
    if answer is None:
        st.error(generation_metadata["error"])
    else:
        st.markdown(answer)

    cited = [r for r in final_chunks if answer and re.search(re.escape(r.chunk.id) + r"\b", answer)]
    st.subheader("Sources")
    for result in cited or final_chunks:
        chunk = result.chunk
        st.markdown(
            f"- **{chunk.metadata.get('section')} {chunk.metadata.get('title')}** "
            f"(pages {chunk.page_start}-{chunk.page_end}) `{chunk.id}`"
        )

    with st.expander(f"Relevant chunks before reranking ({len(candidates)})"):
        st.dataframe(chunk_rows(candidates), use_container_width=True)
    with st.expander(f"Relevant chunks after reranking ({len(final_chunks)})"):
        st.dataframe(chunk_rows(final_chunks), use_container_width=True)
    with st.expander("Configuration"):
        st.json(
            {
                "config": CONFIG.raw,
                "overrides": {
                    "retrieval.strategy": retrieval_strategy,
                    "retrieval.top_k": top_k,
                    "reranking.strategy": CONFIG.reranking.strategy if use_reranker else "none",
                    "reranking.final_k": final_k,
                },
                "chunk_count": len(load_chunks()),
                "timings": timings,
                "generation_metadata": generation_metadata,
            }
        )
