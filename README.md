# RAG Evaluation Lab

A modular RAG experimentation framework where chunking, retrieval, reranking,
and evaluation strategies can be swapped through configuration alone.

See [copilot.md](copilot.md) for the full phased architecture plan.

## Status

Phase 0 — core contracts (`models`, `protocols`, `registry`, `exceptions`) are in place.
Phase 1 — generic ingestion (`txt`, `markdown`, `jsonl` loaders) is in place.
Phase 2 — dataset and ground truth layer (`custom`, `beir`, `squad` adapters) is in place.
Phase 3 — file processing placeholder (see [file_processing/README.md](src/rag_eval_lab/file_processing/README.md)) is in place.
Phase 4 — chunking strategy layer (`fixed_tokens`, `recursive`) is in place.
Phase 5 — embedding and indexing (`sentence_transformer` embedding, `faiss`/`bm25` indexes) is in place.
Phase 6 — retrieval strategies (`dense`, `bm25`, `hybrid` with RRF fusion) is in place.
Phase 7 — reranking strategies (`none`, `cross_encoder`) is in place.
Phase 8 — retrieval evaluation engine (automatic metric selection from ground truth capabilities) is in place.
Phase 9 — experiment configuration and runner (YAML configs, `ExperimentRunner`, JSON `ResultStore`, CLI) is in place.
Phase 11 — optional answer generation (`gemini`/`openai`/`huggingface` providers, disabled by default) is in place.
Phase 12 — optional answer evaluation (`nli`, `llm_judge` evaluators, disabled by default) is in place.
Phase 13 — optional evidence evaluation (`lexical`, `embedding` matchers, disabled by default) is in place.
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

#### SQuAD adapter

Download the official SQuAD 1.1 dev set (~4.8MB; creates the directory and
skips the download if the file already exists; falls back to the raw GitHub
mirror if the primary endpoint fails):

```powershell
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path data\squad | Out-Null
if (-not (Test-Path data\squad\dev-v1.1.json)) {
  try {
    Invoke-WebRequest -Uri "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" `
      -OutFile data\squad\dev-v1.1.json -TimeoutSec 60
  } catch {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json" `
      -OutFile data\squad\dev-v1.1.json -TimeoutSec 60
  }
}
```

`data/` is gitignored, so downloaded dataset files (SQuAD, SciFact) are
never committed.

```python
from rag_eval_lab.datasets import dataset_registry

# limit trims evaluation_cases() (first N questions in file order); the
# shared corpus always contains every paragraph in the file regardless.
squad = dataset_registry.create(
    "squad",
    dataset_path="data/squad/dev-v1.1.json",
    limit=50,
)
documents = squad.documents()
evaluation_cases = squad.evaluation_cases()
```

`SquadDatasetAdapter` reads the nested `data -> articles -> paragraphs -> qas`
structure: article `title` → `Document.metadata`, paragraph `context` →
`Document.text`, QA `id`/`question` → the `EvaluationCase`, the paragraph's
own id → `document_relevance` with grade `1`, and the first annotated answer
text → `reference_answer` (the current answer evaluators use one reference
answer only; alternatives are never concatenated into it). All unique
annotated answer texts are kept as
`ground_truth.metadata["reference_answer_alternatives"]`. Answer spans never
populate `evidence_texts` (an answer span is not necessarily sufficient
supporting evidence), and no answers or relevance labels ever reach
`Document.metadata`. Paragraphs are deduplicated by exact article title
**and** context text (document ids are a deterministic SHA256 hash of both),
and the shared corpus is built from **every** paragraph in the file
independent of `limit` — so even a small generation test still retrieves
against realistic distractors, not just the paragraphs its own (limited)
questions happen to reference.

Source paragraphs provide **positive** relevance labels only — not an
exhaustive judgment that every other paragraph was checked and found
irrelevant — and SQuAD answers are often short spans, so reference-mode NLI
tends to be less informative than context-mode NLI here. These experiments
are a custom reduced retrieval setup and are **not** directly comparable to
the official SQuAD reading comprehension leaderboard.

#### MuSiQue adapter

`MusiqueDatasetAdapter` reads a local MuSiQue JSONL file; it never downloads
data. It selects the first `limit` answerable records in file order and
builds one shared corpus from every selected record's paragraphs, including
distractors. Paragraphs are deduplicated by exact title and text with stable
hash-based document IDs. Supporting paragraphs become positive document
judgments and whole-paragraph `evidence_texts`; answer, support flags, and
question decomposition never enter document metadata.

The included configurations are custom shared-corpus MuSiQue retrieval
experiments, not official benchmark scores. Their evidence matching compares
whole supporting paragraphs, so a match is a retrieval heuristic rather than
proof of multi-hop reasoning or claim support:

```powershell
& .venv\Scripts\python.exe -m rag_eval_lab.experiments configs/experiment_musique_evidence_lexical.yaml
& .venv\Scripts\python.exe -m rag_eval_lab.experiments configs/experiment_musique_evidence_embedding.yaml
```

Both use the same dense retrieval configuration over the first 50 local
answerable validation records with generation and answer metrics disabled.
The lexical experiment uses token-F1 evidence matching; the embedding
experiment additionally loads the configured evidence embedding model.

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

#### Evidence metrics (optional, Phase 13)

Evidence metrics are disabled unless `evaluation.evidence_metrics` is
configured. They require manually supplied `ground_truth.evidence_texts`;
cases without evidence simply emit no evidence metrics. They work with
generation disabled.

```yaml
evaluation:
  k_values: [1, 5, 10]
  evidence_metrics:
    matcher:
      strategy: lexical
      params: {}
    threshold: 0.5
```

For semantic matching, the matcher owns a separate embedding strategy and
constructs it once per experiment:

```yaml
evaluation:
  k_values: [1, 5, 10]
  evidence_metrics:
    matcher:
      strategy: embedding
      params:
        embedding_strategy: sentence_transformer
        embedding_params:
          model_name: sentence-transformers/all-MiniLM-L6-v2
    threshold: 0.8
```

At each $k$, the first $k$ final retrieved chunks are evaluated directly in
rank order, without parent-document deduplication. A chunk/evidence pair
matches when its score is at least the configured threshold.
`evidence_recall@k` is the fraction of gold evidence texts matched by at
least one chunk, so each gold evidence text counts once. `evidence_precision@k`
is the fraction of actually retrieved chunks in the top-$k$ prefix that
match at least one gold evidence text. The score matrix is calculated once
per case for the largest configured $k$ and reused for all cutoffs. Metric
metadata records the matcher, threshold, and, for embedding matching, its
embedding strategy and parameters.

The lexical matcher uses lowercase word-token F1; the embedding matcher uses
cosine similarity. Both are matching heuristics, not proof that a retrieved
passage supports a claim. Thresholds above are examples rather than
universally validated cutoffs. SciFact and SQuAD are intentionally not given
fabricated evidence texts, and SQuAD answer spans remain answer ground truth,
not supporting passages.

See [configs/experiment_evidence_example.yaml](configs/experiment_evidence_example.yaml)
and its tiny manually annotated [configs/evidence_example/](configs/evidence_example/)
dataset for a BM25-only runnable example with no API calls or model downloads.
To add a matcher, implement `EvidenceMatcher.score(retrieved_texts,
evidence_texts)` in `evaluation/evidence/`, provide a stable `name` and
metadata, then register it with `evidence_matcher_registry.register(...)`.
The runner and evaluation engine use only that protocol and require no
matcher-name changes.

### Experiments

A complete experiment (dataset → chunking → embedding/indexing → retrieval →
reranking → evaluation) is fully describable through a single YAML file, run
with:

```powershell
& .venv\Scripts\python.exe -m rag_eval_lab.experiments configs/experiment_a.yaml
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

[configs/](configs/) contains three ready-to-run
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

config = load_config("configs/experiment_a.yaml")
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

### Answer generation (optional, Phase 11)

Generation is disabled unless `generation.enabled: true`; when disabled (or
the section is absent), no generator is constructed and neither an API key
nor any generation SDK is required. Install the optional extra, then either
export key(s) in the shell or copy [.env.example](.env.example) to `.env`
(gitignored) and fill them in — `python-dotenv` loads it automatically, no
further setup needed:

```powershell
pip install -e ".[generation]"
$env:GEMINI_API_KEY = "your-key"   # and/or
$env:OPENAI_API_KEY = "your-key"
```

```yaml
generation:
  enabled: true
  provider: gemini   # or: openai
  params:
    model_name: REPLACE_WITH_YOUR_GEMINI_MODEL_ID  # e.g. gemini-2.0-flash
    temperature: 0.0
    max_output_tokens: 1024
```

See [configs/experiment_generation_example.yaml](configs/experiment_generation_example.yaml)
and [configs/experiment_generation_example_openai.yaml](configs/experiment_generation_example_openai.yaml)
for full runnable configs with the model ID placeholder marked. `provider`
is resolved through `generation_registry` (currently `"gemini"` and
`"openai"`), the same registry pattern as every other strategy — the runner
never branches on provider name, so adding another provider is just a new
`GenerationStrategy` implementation registered under a new name. `params`
(e.g. `model_name`) is passed straight through to the registered generator's
constructor; `provider`/`model_name` are only required when `enabled: true`.

When enabled, the generator is constructed once per experiment and reused
for every query. After reranking, it is called with the question and the
final reranked chunks only — it never sees ground truth, reference answers,
relevance judgments, or gold evidence. `EvaluationSample.generated_answer`
is populated, `sample.timings["generation_seconds"]` and
`ExperimentResult.timings["generation_seconds"]` (total) are recorded, and
retrieval metrics are still computed exactly as before (Phase 11 adds no
answer metrics). Per-query generation records — `query_id`, `question`,
`generated_answer`, ordered `context_chunk_ids`, `generation_metadata`
(provider, model name, available token usage, finish reason — never the API
key), and `generation_seconds` — are saved under
`ExperimentResult.metadata["generation_records"]` using the existing JSON
`ResultStore`; no new core models or full SDK responses are stored. API keys
(`GEMINI_API_KEY`, `OPENAI_API_KEY`) are read directly from the environment
by each generator and never written to YAML, logs, saved configuration, or
generation metadata.

Each provider SDK (`from google import genai`, `from openai import OpenAI`)
is imported lazily inside its generator's `__init__`, so retrieval-only
experiments never require either installed. The client is built once and
closed after the experiment via `generation.close_generator` (a small
helper, not part of the shared `GenerationStrategy` protocol, since not
every provider needs explicit cleanup).

#### Local Hugging Face generation

`provider: huggingface` runs a local instruction-tuned causal LM with
Transformers instead of a hosted API — no network calls or API keys, once
the model weights are cached:

```powershell
pip install -e ".[nli]"   # covers transformers (shared with the nli evaluator)
```

```yaml
generation:
  enabled: true
  provider: huggingface
  params:
    model_name: YOUR_LOCAL_INSTRUCTION_MODEL_ID  # e.g. Qwen/Qwen2.5-0.5B-Instruct
    device: cpu
    max_new_tokens: 256
    do_sample: false
```

`HuggingFaceGenerator` loads the tokenizer and model once per instance
(`AutoTokenizer`/`AutoModelForCausalLM`, imported lazily so retrieval-only
experiments never need `transformers` installed) and reuses them across
every query. It builds the same RAG prompt (`generation.build_prompt`) as
every other provider, applies the model's chat template as a single `user`
message (for compatibility with models that reject a system role), generates
under `torch.inference_mode()`, and decodes only the newly generated tokens
(the prompt is excluded from the output by slicing past the input length —
never re-parsed out of the decoded text). The tokenized prompt is checked
against the tokenizer's `model_max_length` with room left for
`max_new_tokens`; it raises `GenerationError` rather than silently
truncating context if it doesn't fit. `temperature` is only passed through
to `generate()` (and only recorded in metadata) when `do_sample: true`.
Metadata is plain JSON: `provider`, `model_name`, `input_token_count`,
`output_token_count`, `do_sample`, `max_new_tokens`, and `temperature` when
sampling. No hosted inference, quantization, streaming, or automatic model
selection are implemented.

See [configs/experiment_squad_local_generation_example.yaml](configs/experiment_squad_local_generation_example.yaml)
for a full runnable example: SQuAD (small `limit`), local Hugging Face
generation, and context-mode NLI — the Gemini judge is intentionally left
out of that example (its API quota is currently exhausted); add a
`llm_judge` entry back once quota is restored.

### Answer evaluation (optional, Phase 12)

Unlike retrieval metrics (selected automatically from ground truth
capabilities), answer evaluators are explicitly listed and default to
none configured -- i.e. disabled:

```yaml
evaluation:
  k_values: [1, 5, 10]
  answer_metrics:
    - strategy: nli
      params:
        model_name: facebook/bart-large-mnli
        device: cpu
        comparison: reference   # or: context
    - strategy: llm_judge
      params:
        provider: gemini
        model_name: REPLACE_WITH_YOUR_GEMINI_MODEL_ID
        temperature: 0.0
```

`answer_metrics` is a plain list, so the same strategy can appear more than
once with different `params` (e.g. both `comparison: reference` and
`comparison: context` NLI entries, as above). Each entry is resolved through
`evaluation.answer.answer_evaluator_registry` (currently `"nli"` and
`"llm_judge"`) and constructed once per experiment, reused across every
query — the runner never branches on evaluator name. An evaluator only runs
on a sample when its declared `required_inputs` are actually available
(`generated_answer`, and `reference_answer` and/or `retrieved_context`
depending on the evaluator); a missing generated answer always means answer
evaluation is skipped for that sample, reference comparisons additionally
need a `reference_answer` in the ground truth, and context comparisons need
retrieved chunks. Skipped metrics are simply absent, never reported as
zero or fabricated.

See
[configs/experiment_answer_evaluation_example.yaml](configs/experiment_answer_evaluation_example.yaml)
for a full runnable config, paired with a small committed example dataset at
[configs/custom_example/](configs/custom_example/) that includes
`reference_answer`s (SciFact has none, so with that dataset only
context-based NLI and judge faithfulness/relevance run — reference-based NLI
and judge correctness are skipped, not reported as zero).

`NliAnswerEvaluator` (`evaluation/answer/entailment.py`) wraps a
three-label Hugging Face sequence classification model: the premise is
either the reference answer or the concatenated final retrieved chunk
texts, the hypothesis is the generated answer, tokenized as a text pair and
classified; entailment/contradiction/neutral labels are resolved from the
model's own `config.id2label` (never assumed numeric positions), and the
model's input limit is respected (long inputs are truncated, recorded as
`metadata["truncated"]`). This is whole-answer comparison, not claim-level
verification. Returns `answer_nli_{reference,context}_{entailment,contradiction,neutral}`.
`transformers` is optional (the `nli` extra) and imported lazily, so
retrieval-only experiments need no NLI model dependencies or downloads:
`pip install -e ".[nli]"`.

`GeminiJudgeEvaluator` (`evaluation/answer/llm_judge.py`) reuses the Phase 11
Gemini conventions (lazy SDK import, `GEMINI_API_KEY` from the environment,
one client per instance reused across every sample) and sends the question,
generated answer, final retrieved context, and reference answer (only when
available) as material to evaluate — never as instructions, with no web
search or grounding. One API call per sample uses the current google-genai
structured output API (`response_mime_type: application/json` plus a
`response_schema`, selected by whether a reference answer is available: one
schema has `faithfulness`/`relevance`, the other adds `correctness`); the
returned JSON is then independently validated with Pydantic v2 (used only at
this external response boundary — internal models stay stdlib dataclasses).
Each criterion must be a strict integer 0-4 (booleans rejected) with a
required, nonempty explanation; **schema validation enforces response
structure, not judgment accuracy** — a well-formed `{"score": 4, ...}` is
always accepted regardless of whether the score itself is a *good* judgment.
A malformed response, a missing criterion, or a score/explanation that fails
validation raises `GenerationError` rather than becoming a zero score (no
retry framework).

The rubric (`v2`) scores each criterion 0 (worst) to 4 (best) against fixed
behavioral anchors, documented in full in `llm_judge.py`:

- **Faithfulness** (support from retrieved context): 0 — central answer
  contradicts the context or is entirely unsupported; 1 — most substantive
  claims unsupported/contradicted; 2 — a mixture of supported and
  unsupported substantive claims; 3 — all central claims supported, minor
  unsupported details; 4 — all substantive claims supported, no
  contradictions.
- **Relevance** (addressing the question): 0 — does not address the
  question; 1 — mostly unrelated or avoids answering; 2 — addresses part of
  the question but misses major requested elements; 3 — addresses the main
  question with minor omissions/distractions; 4 — directly addresses all
  requested elements without material distractions.
- **Correctness** (agreement with the reference, only scored when one
  exists): 0 — conflicts with the reference or no substantive answer; 1 —
  major factual errors with little correct information; 2 — partially
  correct with a major error/omission; 3 — correct on the main points with a
  minor error/omission; 4 — correct and complete relative to the question
  and reference.

Equivalent paraphrases are accepted; verbosity, writing style, and
word-for-word matching are never rewarded. Facts the reference doesn't cover
are treated as unknown, not automatically false (their *context* support is
still assessed under faithfulness). An explicit abstention ("I don't know")
scores faithfulness 4 only if the context genuinely cannot answer the
question (otherwise 0, for an unjustified claim of insufficiency), relevance
2 if it clearly acknowledges the inability to answer, and correctness 0 when
the reference has a substantive answer. Scores are normalized to 0-1 in
`MetricResult.value` (`score / 4`). Returns `answer_judge_faithfulness` and
`answer_judge_relevance` always, plus `answer_judge_correctness` only when a
reference answer is available. Metric metadata records `provider`,
`model_name`, `rubric_version`, the raw integer `score`, the `explanation`,
and available token usage — never the API key. `pydantic` lives in the same
optional `generation` extra as `google-genai` and is imported lazily inside
the evaluator's constructor, so retrieval-only experiments never require it.

`EvaluationEngine(k_values=..., answer_evaluators=...)` combines the
automatically-selected retrieval metrics with applicable answer evaluator
results; answer metrics always use `k=None`, and aggregation by
`(metric name, k)` is unchanged. `answer_evaluation_seconds` (cumulative
across all samples) is recorded separately in `ExperimentResult.timings`,
alongside the existing `evaluation_seconds` which still times the whole
evaluation step as before. Per-query answer metric results — `query_id`,
`name`, `value`, and `metadata` — are saved under
`ExperimentResult.metadata["answer_metric_records"]`; averaged values stay
in `ExperimentResult.metrics` like every other metric. The full evaluator
configuration (model names, `comparison` mode, rubric) is saved verbatim in
`ExperimentResult.config` — when comparing experiments, keep the same NLI
model/comparison mode and the same judge model/rubric version, otherwise the
comparison is not apples-to-apples. Evaluators are closed after the
experiment via `evaluation.answer.close_evaluator`, the same kind of small
lifecycle helper as `generation.close_generator`.

## Development

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```
