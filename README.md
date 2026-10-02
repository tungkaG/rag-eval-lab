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

## Development

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```
