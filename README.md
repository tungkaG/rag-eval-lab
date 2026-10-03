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
Phase 6 — retrieval strategies (`dense`, `bm25`, `hybrid` with RRF fusion) is in place.
Phase 7 — reranking strategies (`none`, `cross_encoder`) is in place.
Phase 8 — retrieval evaluation engine (automatic metric selection from ground truth capabilities) is in place.
Phase 9 — experiment configuration and runner (YAML configs, `ExperimentRunner`, JSON `ResultStore`, CLI) is in place.
Before/after comparison reports and SQLite storage are intentionally deferred to Phase 10.

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
# downloaded and unzipped to `data/scifact/` containing
# corpus.jsonl, queries.jsonl, and qrels/test.tsv.
beir = dataset_registry.create("beir", dataset_dir="data/scifact", split="test")
documents = beir.documents()
evaluation_cases = beir.evaluation_cases()
```

Both adapters expose the same `documents()` / `evaluation_cases()` shape;
downstream code never needs to know which adapter produced them.

### Chunking

```python
from rag_eval_lab.chunking import chunking_registry
from rag_eval_lab.datasets import dataset_registry

dataset = dataset_registry.create("beir", dataset_dir="data/scifact", split="test")
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

dataset = dataset_registry.create("beir", dataset_dir="data/scifact", split="test")
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

### Retrieval

```python
from rag_eval_lab.core.models import Query
from rag_eval_lab.retrieval import retrieval_registry

# Reuses the indexes and embedding strategy built above (Phase 5).
dense_retriever = retrieval_registry.create("dense", embedder=embedding, index=dense_index)
bm25_retriever = retrieval_registry.create("bm25", index=sparse_index)
hybrid_retriever = retrieval_registry.create(
    "hybrid",
    dense_retriever=dense_retriever,
    sparse_retriever=bm25_retriever,
    dense_top_k=50,
    sparse_top_k=50,
)

case = evaluation_cases[0]
query = Query(id=case.query_id, text=case.question)

for name, retriever in (("dense", dense_retriever), ("bm25", bm25_retriever), ("hybrid", hybrid_retriever)):
    print(f"\n{name}:")
    for result in retriever.retrieve(query, top_k=5):
        print(f"  rank={result.rank} score={result.score:.4f} chunk={result.chunk.id} document={result.chunk.document_id}")
```

All three retrievers share the same `retrieve(query, top_k) -> list[RetrievedChunk]`
shape. `HybridRetriever` only fuses dense and sparse candidates with
Reciprocal Rank Fusion (`retrieval.rrf.reciprocal_rank_fusion`); candidate
counts (`dense_top_k`/`sparse_top_k`) are independent of the final `top_k`.
Neither retriever builds an index or loads a model — both are constructed
from the already-built indexes and embedding strategy.

### Reranking

```python
from rag_eval_lab.reranking import reranking_registry

# Retrieve a wider candidate pool once, then compare rerankers on the same list.
candidates = dense_retriever.retrieve(query, top_k=50)

baseline = reranking_registry.create("none")
cross_encoder = reranking_registry.create(
    "cross_encoder",
    model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
)

for name, reranker in (("none", baseline), ("cross_encoder", cross_encoder)):
    print(f"\n{name}:")
    for result in reranker.rerank(query, candidates, top_k=10):
        print(f"  rank={result.rank} score={result.score:.4f} chunk={result.chunk.id} document={result.chunk.document_id}")

# `candidates` is never mutated by either reranker, so it stays available
# for further comparisons.
```

`NoReranker` is the baseline: it returns the first `top_k` candidates
unchanged. `CrossEncoderReranker` wraps `sentence_transformers.CrossEncoder`,
scoring `(query.text, chunk.text)` pairs and returning new `RetrievedChunk`s
with fresh ranks; it does not retrieve candidates itself — candidate count
(e.g. retrieve 50, rerank to 10) is entirely the caller's responsibility.

### Evaluation

```python
from rag_eval_lab.evaluation import EvaluationEngine
from rag_eval_lab.core.models import EvaluationSample

engine = EvaluationEngine(k_values=[1, 5, 10])

# Same evaluation_case, two separate samples: before and after reranking.
before = EvaluationSample(evaluation_case=case, retrieved_chunks=candidates[:10])
after = EvaluationSample(
    evaluation_case=case,
    retrieved_chunks=cross_encoder.rerank(query, candidates, top_k=10),
)

for label, sample in (("before reranking", before), ("after reranking", after)):
    print(f"\n{label}:")
    for result in engine.evaluate(sample):
        print(f"  {result.name}@{result.k} = {result.value:.4f}")
```

`EvaluationEngine` asks the planner which metrics apply by comparing each
metric's `required_capabilities` against `case.ground_truth.capabilities` —
for a SciFact-style case (document relevance only) that automatically
selects the five document metrics at each `k`; page metrics are added only
when `page_references` are also present. No dataset adapter or metric
contains BEIR-specific logic.

### Experiments

A complete experiment (dataset → chunking → embedding/indexing → retrieval →
reranking → evaluation) is fully describable through a single YAML file, run
with:

```powershell
& .venv\Scripts\python.exe -m rag_eval_lab.experiments configs/experiments/experiment_a.yaml
```

```yaml
experiment:
  name: fixed256_dense_none

dataset:
  adapter: beir
  params:
    dataset_dir: data/scifact
    split: test

chunking:
  strategy: fixed_tokens
  params:
    chunk_size: 256
    overlap: 32

embedding:
  strategy: sentence_transformer
  params:
    model_name: BAAI/bge-base-en-v1.5
    batch_size: 32
    device: cpu
    query_prompt: "Represent this sentence for searching relevant passages: "

indexing:
  dense:
    strategy: faiss
    params: {}

retrieval:
  strategy: dense
  top_k: 50
  params: {}

reranking:
  strategy: none
  final_k: 10
  params: {}

evaluation:
  k_values: [1, 5, 10]

output:
  directory: results
```

Strategy constructor parameters are kept as plain dictionaries (`params`)
and passed straight through to the existing registries via `**params` —
there are no per-strategy configuration classes. Relative paths inside a
config file (`dataset.params.dataset_dir`, `output.directory`) are
interpreted relative to the current working directory the command is run
from (expected to be the repository root), not relative to the YAML file.

For BM25-only retrieval, omit `embedding` and `indexing.dense` entirely and
use `indexing.sparse` + `retrieval.strategy: bm25` — the runner never
instantiates an embedding model in that case. `hybrid` retrieval needs both
`indexing.dense` and `indexing.sparse`, with `dense_top_k` / `sparse_top_k`
/ `rrf_k` under `retrieval.params`. CrossEncoder reranking uses
`reranking.strategy: cross_encoder` with `model_name` (and other
`CrossEncoderReranker` constructor arguments) under `reranking.params`.

#### Corpus embedding caching

`embedding` accepts an optional `cache` section, disabled by default:

```yaml
embedding:
  strategy: sentence_transformer
  params:
    model_name: BAAI/bge-base-en-v1.5
  cache:
    enabled: true
    directory: .cache/embeddings
    force_recompute: false
```

Cache settings never reach the embedding constructor — only `params` does.
When enabled, corpus embeddings are cached under `cache.directory` as one
`<key>.npy` file per experiment configuration, keyed by a SHA256 fingerprint
of the embedding strategy, its `params`, the ordered chunk ids/texts, and a
cache format version; different configurations coexist as separate files,
and changing any of those inputs produces a different key (a cache miss).
`force_recompute: true` bypasses an existing entry and overwrites it. The
embedding model itself is still constructed on a cache hit (query embedding
needs it) and the dense index is still rebuilt every run — only the corpus
`embed_documents` call is skipped. BM25 never touches the embedding cache.
Unlike `dataset.params.dataset_dir` / `output.directory` (resolved relative
to the current working directory), a relative `cache.directory` always
resolves under the repository root — so `.cache/embeddings` always means
`<repo root>/.cache/embeddings`, no matter where a command is run from.
Because `corpus_embedding_seconds` measures loading-or-computing corpus
embeddings (including model construction), cached runs report a much
smaller `corpus_embedding_seconds` than uncached runs — compare timings
with caching consistently enabled or disabled, not across the two. Whether
caching was enabled, whether it was a hit, and the cache key are recorded in
`ExperimentResult.metadata` (`embedding_cache_enabled`, `embedding_cache_hit`,
`embedding_cache_key`).

[configs/experiments/](configs/experiments/) contains three ready-to-run
SciFact configs that only change chunking/reranking: `experiment_a.yaml`
(fixed tokens 256, dense, no reranker), `experiment_b.yaml` (fixed tokens
512, dense, no reranker), and `experiment_c.yaml` (same as B, with
`cross_encoder` reranking).

`ExperimentRunner.run(config)` instantiates every model/index once per run
and reuses them for every evaluation case. Metric values are averaged
across cases, grouped by `(metric name, k)`, using the number of cases that
produced that metric as the denominator — a case whose ground truth lacks
a capability (e.g. no `page_references`) simply does not contribute to that
metric's average. Evaluation always runs on the final (reranked) results;
automatic before/after comparison is deferred to Phase 10. As in Phase 8,
document metrics deduplicate the supplied chunks by `document_id` before
applying `k`, so returning 10 chunks can yield fewer than 10 unique
documents.

`JsonResultStore` (the `ResultStore` implementation) saves one
`<experiment_id>.json` file per experiment under `output.directory` and
reconstructs `ExperimentResult`/`MetricResult` dataclasses on `load()`.

```python
from rag_eval_lab.experiments import ExperimentRunner, load_config

config = load_config("configs/experiments/experiment_a.yaml")
result = ExperimentRunner().run(config)  # saves under config.output.directory

for metric in result.metrics:
    print(f"{metric.name}@{metric.k} = {metric.value:.4f}")
```

`ExperimentRunner(result_store=...)` takes an optional `ResultStore`; when
omitted, `run()` creates a `JsonResultStore(config.output.directory)` itself,
so both direct usage and the CLI always write where the config says. Inject
your own `ResultStore` (e.g. a fake in tests, or a future SQLite store) to
override this.

#### Adding a retrieval strategy

`retrieval.strategy` is resolved through `retrieval_builder_registry`
(`retrieval/builders.py`), not inspected by the runner. Each registered
`RetrievalBuilder.build(config, chunks) -> PreparedRetrieval` owns preparing
whatever its strategy needs (embedder, dense/sparse index) and returns the
constructed retriever plus its preparation timings; `_prepare_dense_index`
and `_prepare_sparse_index` are shared helpers so `hybrid` doesn't duplicate
`dense`'s or `bm25`'s setup. To add a new strategy: implement its
`RetrievalStrategy` (and register it in `retrieval_registry` if new),
implement a `RetrievalBuilder` for it, and register that builder:

```python
retrieval_builder_registry.register("my_strategy")(MyStrategyBuilder)
```

No change to `runner.py` or `config.py` is required.

## Development

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```
