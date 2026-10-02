# RAG Evaluation Lab

A modular RAG experimentation framework where chunking, retrieval, reranking,
and evaluation strategies can be swapped through configuration alone.

See [copilot.md](copilot.md) for the full phased architecture plan.

## Status

Phase 0 — core contracts (`models`, `protocols`, `registry`, `exceptions`) are in place.
Phase 1 — generic ingestion (`txt`, `markdown`, `jsonl` loaders) is in place.
Phase 2 — dataset and ground truth layer (`custom`, `beir` adapters) is in place.
Phase 3 — file processing placeholder (see [file_processing/README.md](src/rag_eval_lab/file_processing/README.md)) is in place.
Phase 4 — chunking strategy layer (`fixed_tokens`, `recursive`) is in place.
Phase 5 — embedding and indexing (`sentence_transformer` embedding, `faiss`/`bm25` indexes) is in place.
No experiment can be run yet; that is intentional.

## Usage

```python
from rag_eval_lab.ingestion import ingestion_registry

# TXT / Markdown: pass a file path or a directory path.
loader = ingestion_registry.create("txt")
documents = loader.load("datasets/manuals")

# JSONL: field names are configurable per dataset schema.
loader = ingestion_registry.create(
    "jsonl",
    fields={"id": "_id", "text": "text", "title": "title"},
)
documents = loader.load("datasets/corpus.jsonl")
```

Every loader returns `list[Document]`, regardless of input format.

### Dataset adapters

```python
from rag_eval_lab.datasets import dataset_registry

# Custom native dataset: a corpus (any ingestion format) plus evaluation JSONL.
custom = dataset_registry.create(
    "custom",
    corpus_path="datasets/manuals/corpus.jsonl",
    evaluation_path="datasets/manuals/evaluation.jsonl",
    corpus_format="jsonl",
    corpus_fields={"id": "id", "text": "text"},
)
documents = custom.documents()
evaluation_cases = custom.evaluation_cases()

# BEIR: point at a locally extracted dataset directory, e.g. SciFact
# downloaded and unzipped to `datasets/scifact/` containing
# corpus.jsonl, queries.jsonl, and qrels/test.tsv.
beir = dataset_registry.create("beir", dataset_dir="datasets/scifact", split="test")
documents = beir.documents()
evaluation_cases = beir.evaluation_cases()
```

Both adapters expose the same `documents()` / `evaluation_cases()` shape;
downstream code never needs to know which adapter produced them.

### Chunking

```python
from rag_eval_lab.chunking import chunking_registry
from rag_eval_lab.datasets import dataset_registry

dataset = dataset_registry.create("beir", dataset_dir="datasets/scifact", split="test")
documents = dataset.documents()

# Swap "fixed_tokens" for "recursive" to change strategy; no other code changes.
chunker = chunking_registry.create("fixed_tokens", chunk_size=256, overlap=32)
chunks = [chunk for document in documents for chunk in chunker.chunk(document)]

print(f"documents={len(documents)} chunks={len(chunks)}")
print(chunks[0])
```

Both chunkers measure `chunk_size` and `overlap` in tokens (`cl100k_base` by
default) and return the same internal `list[Chunk]` shape.

### Embedding and indexing

```python
import time

from rag_eval_lab.chunking import chunking_registry
from rag_eval_lab.datasets import dataset_registry
from rag_eval_lab.embedding import embedding_registry
from rag_eval_lab.indexing import dense_index_registry, sparse_index_registry

dataset = dataset_registry.create("beir", dataset_dir="datasets/scifact", split="test")
documents = dataset.documents()
evaluation_cases = dataset.evaluation_cases()

chunker = chunking_registry.create("fixed_tokens", chunk_size=256, overlap=32)
chunks = [chunk for document in documents for chunk in chunker.chunk(document)]

# BAAI/bge-base-en-v1.5 documents a query-side search instruction and no
# passage-side instruction. `query_prompt` is forwarded as `prompt=` to
# `SentenceTransformer.encode_query`; `document_prompt` stays unset here.
embedding = embedding_registry.create(
    "sentence_transformer",
    model_name="BAAI/bge-base-en-v1.5",
    query_prompt="Represent this sentence for searching relevant passages: ",
)
document_embeddings = embedding.embed_documents([chunk.text for chunk in chunks])

start = time.perf_counter()
dense_index = dense_index_registry.create("faiss")
dense_index.add(chunks, document_embeddings)
dense_build_seconds = time.perf_counter() - start

start = time.perf_counter()
sparse_index = sparse_index_registry.create("bm25")
sparse_index.build(chunks)
sparse_build_seconds = time.perf_counter() - start

print(f"documents={len(documents)} chunks={len(chunks)}")
print(f"faiss build: {dense_build_seconds:.2f}s, bm25 build: {sparse_build_seconds:.2f}s")

question = evaluation_cases[0].question
query_embedding = embedding.embed_queries([question])[0]

print("dense:", dense_index.search(query_embedding, top_k=3))
print("sparse:", sparse_index.search(question, top_k=3))
```

`FaissIndex` wraps `faiss.IndexFlatIP` over L2-normalized vectors (cosine
similarity); `BM25Index` wraps `rank_bm25.BM25Okapi` and needs no
embeddings. Both return the same internal `list[RetrievedChunk]` shape.

`cl100k_base` (used for `chunk_size`/`overlap`) is unrelated to a given
embedding model's own tokenizer, so a chunk's token count under one
tokenizer is not its token count under the other; inputs longer than the
embedding model's max sequence length may be silently truncated by the
model itself. This phase does not reconcile the two tokenizers.

Model download and real embedding execution only happen when this example
is run directly — the automated test suite mocks the Sentence Transformer
model, so it never downloads weights.

## Development

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```
