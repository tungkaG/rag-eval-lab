# RAG Evaluation Lab

## Project Goal

Build a modular RAG experimentation framework where individual strategies can be swapped through configuration.

The framework must allow experiments such as:

```text
Experiment A

Fixed token chunking
Dense retrieval
No reranking
```

```text
Experiment B

Recursive chunking
Dense retrieval
No reranking
```

```text
Experiment C

Recursive chunking
Hybrid retrieval
Cross encoder reranking
```

The framework must then evaluate each experiment using whatever ground truth information is available.

The architecture must not depend on BEIR or any particular benchmark.

BEIR should simply be one dataset adapter.

The first version should focus on retrieval experiments. Generation and more advanced answer evaluation can be added after the retrieval pipeline is stable.

File processing should exist only as a placeholder initially.

# Strategy Implementation Policy

For algorithmic components such as chunking, embedding, indexing, retrieval, reranking, and evaluation metrics, do not implement established algorithms from scratch unless necessary.

Use this priority order:

1. Prefer a mature existing Python package.

2. Wrap the external implementation behind the project's internal protocol.

3. External library objects must not escape the adapter or strategy implementation.

4. Convert inputs from the project's internal models into the external library format internally.

5. Convert outputs back into the project's internal models before returning.

6. If no suitable package exists, inspect a specified open source repository for a reference implementation.

7. Before adapting source code from another repository, verify its license.

8. Adapt only the minimum required implementation.

9. Do not copy an entire framework simply to obtain one strategy.

10. Implement an algorithm locally only when no appropriate reusable implementation exists.

The project owns orchestration, interfaces, normalization, provenance, configuration, and evaluation.

External libraries may own the underlying algorithm implementation.

# Phase 0, Define the Core Contracts

## Objective

Create the internal data model that every component uses.

Do this before implementing algorithms.

External libraries such as Haystack, Sentence Transformers, FAISS, BEIR, or RAGAS must never become the internal data model of the project.

## Create

```text
src/rag_eval_lab/core/
    models.py
    protocols.py
    registry.py
    exceptions.py
```

## Core models

Implement at minimum:

```python
Document
Chunk
Query
GroundTruth
PageReference
RetrievedChunk
EvaluationCase
EvaluationSample
MetricResult
ExperimentResult
```

Suggested conceptual structure:

```python
Document

id
text
metadata
```

```python
Chunk

id
document_id
text
page_start
page_end
char_start
char_end
metadata
```

```python
GroundTruth

document_relevance
page_references
reference_answer
evidence_texts
keywords
metadata
```

`document_relevance` maps a document identifier to an integer relevance grade.
Use `1` for relevant and `0` for explicitly judged irrelevant under binary
relevance; preserve the dataset's original integer grades otherwise. An
absent document identifier means no judgment is provided; do not populate
the mapping with every irrelevant document in the corpus.

```python
EvaluationCase

query_id
question
ground_truth
```

```python
RetrievedChunk

chunk
score
rank
retriever_metadata
```

```python
EvaluationSample

evaluation_case
retrieved_chunks
generated_answer
timings
```

## Protocols

Define interfaces for:

```python
DatasetAdapter
DocumentLoader
ChunkingStrategy
EmbeddingStrategy
Index
RetrievalStrategy
RerankingStrategy
GenerationStrategy
Metric
ResultStore
```

Do not implement algorithm logic here.

## Acceptance criteria

The package imports successfully.

All models have unit tests.

Strategies depend on protocols rather than concrete implementations.

No component depends on BEIR specific classes.

No component depends on Haystack specific classes.

No experiment can run yet.

That is intentional.


# Phase 1, Generic Ingestion

## Objective

Support the three normalized input formats:

```text
TXT
Markdown
JSONL
```

Every format must produce:

```python
list[Document]
```

## Create

```text
src/rag_eval_lab/ingestion/
    base.py
    txt_loader.py
    markdown_loader.py
    jsonl_loader.py
    registry.py
```

## TXT loader

Support:

```text
folder/
    document1.txt
    document2.txt
    document3.txt
```

Each file becomes one `Document`.

The document identifier should be deterministic.

Prefer filename based identifiers unless configuration specifies another rule.

## Markdown loader

Same behavior as TXT initially.

Do not parse Markdown structure yet.

Preserve the raw Markdown content.

Later versions can introduce heading aware chunking.

## JSONL loader

Make field mappings configurable.

For example:

```yaml
ingestion:
  format: jsonl

  fields:
    id: "_id"
    text: "text"
    title: "title"
```

This is important because different benchmarks use different schemas.

Do not create a BEIR specific JSONL loader.

The BEIR adapter should configure the generic JSONL loader.

## Registry

Allow:

```python
loader = ingestion_registry.create(
    "jsonl",
    config
)
```

## Tests

Test:

```text
single TXT file
multiple TXT files
Markdown loading
JSONL loading
missing configured JSONL field
duplicate document identifiers
empty documents
```

## Acceptance criteria

The following formats all produce the same internal object:

```text
TXT
    ↓
Document[]

Markdown
    ↓
Document[]

JSONL
    ↓
Document[]
```


# Phase 2, Dataset and Ground Truth Layer

## Objective

Separate corpus ingestion from benchmark specific evaluation data.

A dataset adapter should expose:

```python
documents
evaluation_cases
```

## Create

```text
src/rag_eval_lab/datasets/
    base.py
    custom_dataset.py
    beir_adapter.py
    registry.py
```

and:

```text
src/rag_eval_lab/ground_truth/
    adapters.py
    validator.py
```

`GroundTruth`, `PageReference`, and `GroundTruthCapability` already live in
`core/models.py` (Phase 0). This package only adds native record conversion
(`adapters.py`) and the basic normalization and relation checks
(`validator.py`) on top of them; it does not define parallel models.

## Custom dataset

Define a simple native evaluation format for this project.

Example:

```json
{
  "query_id": "q1",
  "question": "What is the maximum operating temperature?",
  "ground_truth": {
    "document_relevance": {"manual_1": 1},
    "page_references": [
      {"document_id": "manual_1", "page_start": 23, "page_end": 23}
    ],
    "reference_answer": "85 degrees Celsius",
    "evidence_texts": [],
    "keywords": [],
    "metadata": {}
  }
}
```

Only `query_id` and `question` are required; a missing or partial
`ground_truth` object falls back to the `GroundTruth` field defaults.

## BEIR adapter

Implement BEIR as the first external benchmark adapter.

Map:

```text
BEIR corpus
    ↓
Document[]

BEIR queries and qrels
    ↓
EvaluationCase[]
```

BEIR logic must remain entirely inside the dataset adapter.

The rest of the application should not know that BEIR exists.

## Ground truth capabilities

Do not classify examples into five fixed case types.

Instead determine independent capabilities.

Implement capabilities as the literal values already defined on
`GroundTruth.capabilities` in `core/models.py`:

```text
document_relevance
page_references
reference_answer
evidence_texts
keywords
```

`claims` is a planned future addition (Phase 13) once `GroundTruth` gains a
claims field; it does not exist yet.

`GroundTruth.capabilities` is a computed property, not a stored field, so it
can never drift out of sync with the populated data. For example:

```python
GroundTruth(
    document_relevance={"manual_1": 1},
    reference_answer="85 degrees Celsius",
).capabilities
# {"document_relevance", "reference_answer"}
```

## Ground truth validator

Validate relations such as:

```text
page references must contain a document identifier

document identifiers referenced by ground truth should exist when document ground truth is used

empty answer strings should become None

empty lists should be normalized consistently
```

## Acceptance criteria

Both of these should work:

```text
BEIR SciFact
    ↓
DatasetAdapter
    ↓
Document[]
EvaluationCase[]
```

and:

```text
Custom JSONL evaluation dataset
    ↓
DatasetAdapter
    ↓
Document[]
EvaluationCase[]
```

Everything downstream receives identical internal objects.


# Phase 3, File Processing Placeholder

## Objective

Reserve the architecture for future conversion of PDF, Word, Excel, HTML, scanned files, and other messy sources.

Do not implement Azure Document Intelligence yet.

## Create

```text
src/rag_eval_lab/file_processing/
    base.py
    registry.py
    README.md
```

## Interface

Define something conceptually similar to:

```python
class FileProcessor:

    def process(
        self,
        input_path,
        output_path
    ):
        ...
```

The future expectation is:

```text
PDF
    ↓
Azure Document Intelligence
    ↓
Markdown
```

or:

```text
DOCX
    ↓
processor
    ↓
Markdown
```

or:

```text
XLSX
    ↓
processor
    ↓
JSONL
```

The processed result then enters the normal ingestion layer.

## Important architectural rule

`file_processing` must not create `Document` objects directly.

Its responsibility is file conversion.

`ingestion` remains responsible for converting normalized files into `Document` objects.

## Acceptance criteria

Placeholder interfaces exist.

No external document processing dependency is required.

Nothing in the experiment pipeline depends on file processing.


# Phase 4, Chunking Strategy Layer

## Objective

Introduce the first interchangeable experiment variable.

Start with only two chunkers.

## Create

```text
src/rag_eval_lab/chunking/
    base.py
    fixed_tokens.py
    recursive.py
    registry.py
```

## Strategy 1

Fixed token chunking.

Configuration:

```yaml
chunking:
  strategy: fixed_tokens
  chunk_size: 512
  overlap: 64
```

## Strategy 2

Recursive chunking.

Use an existing implementation where practical.

Do not write a sophisticated recursive algorithm unnecessarily.

The project should wrap external implementations behind `ChunkingStrategy`.

## Provenance

Every produced chunk must retain:

```text
chunk identifier
parent document identifier
page information if available
character offsets where practical
strategy metadata
```

Example:

```python
Chunk(
    id="manual_1_chunk_7",
    document_id="manual_1",
    text="...",
    page_start=23,
    page_end=24,
    char_start=15520,
    char_end=16950
)
```

## Acceptance criteria

The same documents can be processed using:

```text
fixed_tokens
```

or:

```text
recursive
```

without changing any other pipeline code.

The output is always:

```python
list[Chunk]
```


# Phase 5, Embedding and Indexing

## Objective

Make vector representation and storage replaceable.

## Create

```text
src/rag_eval_lab/embedding/
    base.py
    sentence_transformers.py
    registry.py
```

```text
src/rag_eval_lab/indexing/
    base.py
    faiss_index.py
    bm25_index.py
    registry.py
```

## First dense embedding implementation

Use Sentence Transformers.

Configuration example:

```yaml
embedding:
  strategy: sentence_transformer
  model: BAAI/bge_base_en_v1_5
```

## Dense index

Use FAISS initially.

Do not introduce a server based vector database yet.

The experiment lab does not need Qdrant, Elasticsearch, or a cloud vector database at this stage.

## Sparse index

Provide BM25 indexing.

Reuse an established BM25 implementation.

Do not implement BM25 mathematics manually.

## Acceptance criteria

The following indexes can be constructed from the same chunks:

```text
FAISS
BM25
```

Index construction time should be recorded.


# Phase 6, Retrieval Strategies

## Objective

Make retrieval interchangeable independently from chunking.

## Create

```text
src/rag_eval_lab/retrieval/
    base.py
    dense.py
    bm25.py
    hybrid.py
    rrf.py
    registry.py
```

## Dense retriever

Input:

```text
query
```

Output:

```python
list[RetrievedChunk]
```

## BM25 retriever

Same interface.

## Hybrid retriever

Combine dense and sparse results.

## Reciprocal Rank Fusion

Implement RRF as a reusable fusion strategy.

RRF should not depend on FAISS or BM25 internals.

It simply receives ranked result lists.

## Configuration

Example:

```yaml
retrieval:
  strategy: dense
  top_k: 50
```

or:

```yaml
retrieval:
  strategy: bm25
  top_k: 50
```

or:

```yaml
retrieval:
  strategy: hybrid
  fusion: rrf
  dense_top_k: 50
  sparse_top_k: 50
  final_top_k: 50
```

## Acceptance criteria

Changing only:

```yaml
strategy: dense
```

to:

```yaml
strategy: bm25
```

runs a new experiment without application code changes.


# Phase 7, Reranking Strategies

## Objective

Allow reranking to be enabled, disabled, or replaced independently.

## Create

```text
src/rag_eval_lab/reranking/
    base.py
    none.py
    cross_encoder.py
    registry.py
```

## No reranker

Simply return results unchanged.

This is important because it provides a baseline.

## Cross encoder

Use Sentence Transformers or another established implementation.

Configuration example:

```yaml
reranking:
  strategy: cross_encoder
  model: cross_encoder/ms_marco_MiniLM_L_6_v2
  candidate_k: 50
  final_k: 10
```

## Acceptance criteria

Both must run:

```text
Dense
    ↓
No reranker
```

and:

```text
Dense
    ↓
Cross encoder
```

The retrieval result before reranking should optionally be retained so that reranking impact can be measured separately.


# Phase 8, Retrieval Evaluation Engine

## Objective

Implement automatic metric selection based on ground truth capabilities.

This is one of the most important phases.

## Create

```text
src/rag_eval_lab/evaluation/
    engine.py
    planner.py
    registry.py
    result.py
```

```text
src/rag_eval_lab/evaluation/retrieval/
    document_metrics.py
    page_metrics.py
    evidence_metrics.py
```

## Metric contract

Every metric declares the capabilities it requires.

Conceptually:

```python
class Metric:

    required_capabilities = set()

    def evaluate(self, sample):
        ...
```

Example:

```python
class DocumentRecallAtK:

    required_capabilities = {
        "document_relevance"
    }
```

Example:

```python
class PageHitRate:

    required_capabilities = {
        "page_references"
    }
```

## Initial document metrics

Implement:

```text
Hit Rate at K

Recall at K

Precision at K

MRR

NDCG
```

Optionally:

```text
MAP
```

## Parent document mapping

When chunks are retrieved, document level metrics must map:

```text
document_17_chunk_3
    ↓
document_17
```

Duplicate parent documents must be handled correctly.

## Page metrics

If page ground truth exists, implement:

```text
Page Hit Rate at K

Page Recall at K

Page Precision at K
```

A chunk counts as a page hit when its page range overlaps an accepted gold page range.

## Evaluation planner

The planner inspects capabilities.

Example:

```text
Ground truth:

document available
page unavailable
answer unavailable
```

Automatically select:

```text
Document Hit Rate
Recall
Precision
MRR
NDCG
```

Example:

```text
document available
page available
answer unavailable
```

Automatically add:

```text
Page Hit Rate
Page Recall
Page Precision
```

## Acceptance criteria

Metrics are selected automatically.

No dataset adapter contains metric logic.

No metric contains BEIR specific logic.


# Phase 9, Experiment Configuration and Runner (implemented)

## Objective

Create the actual plug and play experiment lab.

## Created

```text
src/rag_eval_lab/experiments/
    __init__.py
    __main__.py
    config.py
    runner.py
    result_store.py
```

`comparison.py` stays Phase 10 scope, as originally planned.

## Configuration

Experiments are completely describable through YAML. Strategy constructor
parameters are kept as plain `params` dictionaries and passed straight
through to the existing registries via `**params` — there are no
per-strategy config classes. Actual shape (see
[README.md](README.md#experiments) and
[configs/](configs/) for runnable examples):

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
  sparse:
    strategy: bm25
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

`embedding` and `indexing.dense` are optional and only required when
`retrieval.strategy` is `dense` or `hybrid`; a BM25-only experiment omits
them and the runner never instantiates an embedding model. Relative paths
(e.g. `dataset.params.dataset_dir`, `output.directory`) are interpreted
relative to the current working directory the command is run from (expected
to be the repository root), not relative to the YAML file itself.

## Runner

The runner owns orchestration only; algorithms stay in their existing
components (`ExperimentRunner.run(config) -> ExperimentResult` in
`runner.py`):

```text
create dataset adapter, load documents and evaluation cases

create chunker, chunk documents

build the components required by the selected retrieval strategy
(dense: embedder + dense index; bm25: sparse index only; hybrid: both)

create retriever and reranker through their registries

for each evaluation case:

    create a Query from the case's query_id and question

    retrieve top_k candidates

    rerank to final_k

    build an EvaluationSample and evaluate it

average metrics across cases, grouped by (name, k)

save and return the ExperimentResult
```

Models and indexes are instantiated once per experiment run, not rebuilt
per query.

## Experiment metadata

Stored in `ExperimentResult.config` (the full parsed YAML) and
`ExperimentResult.metadata`:

```text
experiment name

UTC timestamp

document count

chunk count

query count

software version information (rag_eval_lab, python, platform)
```

Timings (`ExperimentResult.timings`, all `perf_counter`-based seconds):
dataset loading, chunking, corpus embedding (dense only), index building,
retrieval (includes query embedding), reranking, evaluation, and total.

## Corpus embedding caching (optional)

`embedding.cache` (`enabled`/`directory`/`force_recompute`, disabled by
default, kept separate from `embedding.params`) lets `_prepare_dense_index`
(`retrieval/builders.py`) skip `embed_documents` on a cache hit. The cache
key (`embedding/cache.py`) is a SHA256 hash of the embedding strategy, its
params, the ordered chunk ids/texts, and a cache format version; one
`<key>.npy` file per configuration. The embedding model is still constructed
and the dense index still rebuilt every run; BM25 never touches the cache.
A relative `cache.directory` resolves under the repository root
(`core/paths.py`'s `REPO_ROOT`/`resolve_under_repo_root`), not the current
working directory, so caches land in one stable place regardless of where a
command is run from.
`ExperimentResult.metadata` records `embedding_cache_enabled`,
`embedding_cache_hit`, and `embedding_cache_key`.

## Result storage

`result_store.py` implements the existing `ResultStore` protocol as
`JsonResultStore`: one `<experiment_id>.json` file per experiment under
`output.directory`, using `dataclasses.asdict` / manual reconstruction for
`save()` / `load()`. SQLite, comparison tables, and dashboards stay Phase 10
scope.

## Command

```powershell
python -m rag_eval_lab.experiments configs/experiment_a.yaml
```

Prints the saved result path and a compact `name@k = value` metric summary.

## Acceptance criteria

Three experiments can be launched by changing YAML only.

For example:

```text
Experiment A

Fixed 256
Dense
No reranker
```

```text
Experiment B

Fixed 512
Dense
No reranker
```

```text
Experiment C

Fixed 512
Dense
Cross encoder
```


# Phase 10, Result Storage and Comparison

## Objective

Make strategy changes measurable.

## Storage

Initially use:

```text
JSON
```

or:

```text
SQLite
```

I would prefer SQLite once the basic runner works because experiment comparisons become significantly easier.

## Store quality metrics

Examples:

```text
Recall at 1
Recall at 5
Recall at 10

Precision at 10

MRR

NDCG at 10

Page Recall
```

## Store system metrics

Also collect:

```text
number of documents

number of chunks

average chunk length

index build time

retrieval latency

reranking latency

total experiment duration
```

Later:

```text
embedding token count

API cost

LLM judge cost

generation latency
```

## Comparison module

Allow comparisons such as:

```text
Experiment A
versus
Experiment B
```

Output differences:

```text
Recall at 10

0.81
    ↓
0.85

change
plus 0.04
```

The module should also identify which configuration parameters changed.

Example:

```text
Changed:

chunk_size
256 → 512

Unchanged:

retriever
reranker
embedding model
dataset
```

This is extremely useful for controlled experiments.

## Acceptance criteria

A user can run several configurations and obtain a comparison report without manually collecting numbers.


# Phase 11, Answer Generation (implemented)

## Objective

Add end to end RAG evaluation without disturbing the retrieval architecture.

## Created

```text
src/rag_eval_lab/generation/
    __init__.py
    base.py
    prompt_builder.py
    gemini.py
    openai.py
    registry.py
```

Reuses the existing `GenerationStrategy` protocol from `core/protocols.py`
unchanged: `generate(query: Query, retrieved_chunks: list[RetrievedChunk]) ->
tuple[str, dict[str, Any]]`. No new core models were added; generation
output is stored in `EvaluationSample.generated_answer` (already existed)
and plain dicts under `ExperimentResult.metadata`.

## Prompt builder

`prompt_builder.build_prompt(query, retrieved_chunks) -> str` only accepts a
`Query` and `list[RetrievedChunk]` — structurally, it cannot leak ground
truth, reference answers, relevance judgments, or gold evidence. Passages
are kept in retrieval order and labeled `[chunk_id | document=document_id]`.
The instruction text tells the model to answer from context only, say so
when context is insufficient, cite passage labels, and treat passages as
evidence rather than instructions.

## Generators

`GeminiGenerator(model_name, temperature=0.0, max_output_tokens=1024)` wraps
`google.genai.Client` (the Gemini Developer API); `OpenAIGenerator` (same
constructor signature) wraps `openai.OpenAI`'s Chat Completions API. Both
import their SDK lazily inside `__init__` so retrieval-only experiments
never need either installed, read their API key (`GEMINI_API_KEY` /
`OPENAI_API_KEY`) from the environment via `core.env.load_env_file()` (loads
a repo-root `.env` through `python-dotenv` if present, never from YAML, and
never written to logs, saved configuration, or generation metadata), build
one client per instance reused across every `generate()` call, and raise
`core.exceptions.GenerationError` rather than fabricating an answer when the
response has no text. Returned metadata: `provider`, `model_name`, available
token usage counts, and `finish_reason` when present — never the API key or
the full SDK response. Registered as `"gemini"` / `"openai"` in
`generation_registry` (the same generic `Registry` every other strategy
uses); the runner selects a provider through the registry only, never
through provider branches. `google-genai`, `openai`, and `python-dotenv` are
all optional dependencies (the `generation` extra in `pyproject.toml`):
`pip install -e ".[generation]"`.

## Configuration

```yaml
generation:
  enabled: false
```

or:

```yaml
generation:
  enabled: true
  provider: gemini   # or: openai
  params:
    model_name: REPLACE_WITH_YOUR_GEMINI_MODEL_ID
    temperature: 0.0
    max_output_tokens: 1024
```

Missing the `generation` section, or `enabled: false`, both mean disabled;
`provider` and `params.model_name` are only required when `enabled: true`.
See [configs/experiment_generation_example.yaml](configs/experiment_generation_example.yaml)
and [configs/experiment_generation_example_openai.yaml](configs/experiment_generation_example_openai.yaml).

## Runner integration

When disabled, `runner.py` never constructs a generator and never requires
an API key or the SDK — existing retrieval experiment behavior is
unchanged. When enabled, a generator is built once (like every other
strategy) and called after reranking with the final chunks; its answer
populates `EvaluationSample.generated_answer`, `sample.timings["generation_seconds"]`
and the experiment's total `timings["generation_seconds"]` are recorded, and
retrieval metrics still run exactly as before. Per-query records (`query_id`,
`question`, `generated_answer`, ordered `context_chunk_ids`,
`generation_metadata`, `generation_seconds`) are saved under
`ExperimentResult.metadata["generation_records"]` via the existing JSON
`ResultStore` — no new core models, no full SDK responses, no duplicated
chunk text. The generator is closed in a `finally` block via
`generation.close_generator`, a small helper outside the shared protocol.

## Acceptance criteria

Retrieval experiments remain possible without an LLM.

Generation is completely optional.


# Phase 12, Answer Evaluation

## Objective

Use answer ground truth when available.

## Create

```text
src/rag_eval_lab/evaluation/answer/
    exact_match.py
    token_f1.py
    keyword_metrics.py
    semantic_similarity.py
    entailment.py
    llm_judge.py
```

## Initial deterministic metrics

Implement first:

```text
Exact Match

Token F1

Keyword Precision

Keyword Recall
```

## Semantic metrics

Then:

```text
Sentence embedding similarity

BERTScore if useful
```

## Entailment

Add NLI based comparison.

This should help distinguish semantic equivalence from vocabulary similarity.

## LLM judge

Add last.

LLM judging should be optional because it introduces:

```text
cost

latency

nondeterminism

provider dependency
```

Use structured scoring rather than free form judging.

## Automatic metric selection

If:

```text
ANSWER_GT = false
```

answer metrics are skipped.

If:

```text
ANSWER_GT = true
```

applicable answer metrics are enabled.

## Acceptance criteria

The same evaluation engine handles retrieval and answer evaluation based purely on available capabilities.


# Phase 13, Evidence and Claim Evaluation

## Objective

Support benchmarks where explicit evidence or reference claims exist.

## Evidence metrics

Compare retrieved chunks against gold evidence using:

```text
lexical overlap

embedding similarity

evidence recall

evidence precision
```

## Claim metrics

Normalize reference answers into claims when claim ground truth exists.

Later optionally support automatic claim extraction.

Evaluate:

```text
Claim Recall

Claim Precision

Claim F1

Claim Support
```

Keep automatic claim extraction separate from manually provided claim ground truth.

## Acceptance criteria

A dataset providing evidence text can evaluate retrieval more precisely than document level relevance alone.


# Phase 14, Additional Strategies

## Objective

Expand the experiment space only after the architecture has proven stable.

Add strategies independently.

## Chunking

Candidates:

```text
sentence chunking

semantic chunking

heading aware Markdown chunking

sentence window chunking

parent child chunking
```

## Retrieval

Candidates:

```text
hybrid weighted retrieval

query expansion

multi query retrieval

HyDE

metadata filtering
```

## Reranking

Candidates:

```text
BGE reranker

bi encoder reranker

LLM reranker

Cohere reranker
```

Each new implementation should require only:

```text
one strategy implementation

registry registration

configuration schema

tests
```

The experiment runner should not change.


# Phase 15, Additional Dataset Adapters

## Objective

Validate that the architecture truly is benchmark independent.

Add adapters such as:

```text
FinanceBench

other BEIR datasets

RAGBench

custom enterprise benchmark

future internal datasets
```

Each adapter is responsible only for translating external representation into:

```text
Document[]

EvaluationCase[]
```

Nothing else.


# Phase 16, CLI

## Objective

Make experiments easy to launch.

## Create

```text
src/rag_eval_lab/cli/
    main.py
```

Commands could eventually look like:

```text
raglab run configs/experiments/baseline.yaml
```

```text
raglab compare run_001 run_002
```

```text
raglab datasets
```

```text
raglab strategies
```

The CLI should remain thin.

It should call the existing experiment services.


# Phase 17, Experiment Matrix Runner

## Objective

Support systematic strategy searches.

Instead of manually creating thirty YAML files, allow:

```yaml
matrix:

  chunking:
    strategy:
      [fixed_tokens, recursive]

    chunk_size:
      [256, 512, 1024]

  retrieval:
    strategy:
      [dense, bm25, hybrid]

  reranking:
    strategy:
      [none, cross_encoder]
```

This generates every valid combination.

For the example above:

```text
2 chunkers

3 chunk sizes

3 retrieval strategies

2 reranking strategies
```

produce the relevant experiment combinations.

The system should preserve configuration hashes so identical experiments are not accidentally repeated.

## Acceptance criteria

One matrix configuration can execute multiple reproducible experiments.


# Phase 18, Reporting and Visualization

## Objective

Turn experiment results into actionable comparisons.

Do this after the underlying results are trustworthy.

Possible outputs:

```text
leaderboard

metric comparison

latency comparison

quality versus latency

quality versus index size

per query failures

retrieval examples

strategy differences
```

Useful visualization:

```text
Recall at 10
versus
retrieval latency
```

Another:

```text
NDCG
versus
number of chunks
```

Another useful view:

```text
Query q27

Gold documents

Retrieved before reranking

Retrieved after reranking

Final ranks

Metric impact
```

This failure analysis view may ultimately be more useful than the leaderboard itself.


# Recommended First Milestone

Do not attempt all phases before testing the architecture.

The first meaningful milestone should stop after Phase 10.

At that point the framework should support:

```text
Input

TXT
Markdown
JSONL


Datasets

custom
BEIR


Chunking

fixed token
recursive


Retrieval

dense
BM25
hybrid


Reranking

none
cross encoder


Ground truth

document
document plus page


Evaluation

Hit Rate
Recall
Precision
MRR
NDCG
Page Hit Rate
Page Recall


Experiment tracking

configuration
metrics
latency
comparison
```

That alone is already a useful RAG experimentation framework.


# Recommended First End to End Test

Use BEIR SciFact only as the first integration test, not as an architectural dependency.

Run:

```text
Run 001

Fixed tokens 256
Dense
No reranking
```

```text
Run 002

Fixed tokens 512
Dense
No reranking
```

```text
Run 003

Recursive
Dense
No reranking
```

```text
Run 004

Fixed tokens 512
BM25
No reranking
```

```text
Run 005

Fixed tokens 512
Hybrid
No reranking
```

```text
Run 006

Fixed tokens 512
Hybrid
Cross encoder
```

The resulting comparison should make it immediately visible what changed between runs and what effect that had.


# Architectural Rules for Copilot

1. Keep external libraries behind project interfaces.

2. Do not expose Haystack, BEIR, FAISS, Sentence Transformers, RAGAS, or other third party types through the core interfaces.

3. Dataset adapters only normalize data.

4. File processors only convert source files.

5. Ingestion only creates `Document` objects.

6. Chunkers only create `Chunk` objects.

7. Retrievers only return ranked `RetrievedChunk` objects.

8. Rerankers only reorder or rescore retrieved results.

9. Metrics only evaluate completed outputs.

10. The experiment runner orchestrates components but contains no algorithm implementation.

11. Metrics declare their required ground truth capabilities.

12. The evaluation planner automatically skips metrics whose requirements are unavailable.

13. Preserve provenance from source document through chunk through retrieval result.

14. Every experiment must be reproducible from configuration.

15. Strategies must be independently replaceable.

16. Prefer wrapping proven libraries over reimplementing algorithms.

17. Keep generation optional.

18. Keep LLM based evaluation optional.

19. Do not build a UI until the experiment engine and result comparison are reliable.

20. Add advanced strategies only after the baseline architecture is covered by tests.


# Target Repository After the First Major Milestone

```text
rag_eval_lab/
│
├── pyproject.toml
├── README.md
│
├── configs/
│   ├── datasets/
│   └── experiments/
│
├── src/
│   └── rag_eval_lab/
│       │
│       ├── core/
│       │   ├── models.py
│       │   ├── protocols.py
│       │   ├── registry.py
│       │   └── exceptions.py
│       │
│       ├── ingestion/
│       │   ├── base.py
│       │   ├── txt_loader.py
│       │   ├── markdown_loader.py
│       │   ├── jsonl_loader.py
│       │   └── registry.py
│       │
│       ├── file_processing/
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── README.md
│       │
│       ├── datasets/
│       │   ├── base.py
│       │   ├── custom_dataset.py
│       │   ├── beir_adapter.py
│       │   └── registry.py
│       │
│       ├── ground_truth/
│       │   ├── capabilities.py
│       │   ├── validator.py
│       │   └── adapters.py
│       │
│       ├── chunking/
│       │   ├── base.py
│       │   ├── fixed_tokens.py
│       │   ├── recursive.py
│       │   └── registry.py
│       │
│       ├── embedding/
│       │   ├── base.py
│       │   ├── sentence_transformers.py
│       │   └── registry.py
│       │
│       ├── indexing/
│       │   ├── base.py
│       │   ├── faiss_index.py
│       │   ├── bm25_index.py
│       │   └── registry.py
│       │
│       ├── retrieval/
│       │   ├── base.py
│       │   ├── dense.py
│       │   ├── bm25.py
│       │   ├── hybrid.py
│       │   ├── rrf.py
│       │   └── registry.py
│       │
│       ├── reranking/
│       │   ├── base.py
│       │   ├── none.py
│       │   ├── cross_encoder.py
│       │   └── registry.py
│       │
│       ├── evaluation/
│       │   ├── engine.py
│       │   ├── planner.py
│       │   ├── registry.py
│       │   ├── result.py
│       │   │
│       │   └── retrieval/
│       │       ├── document_metrics.py
│       │       ├── page_metrics.py
│       │       └── evidence_metrics.py
│       │
│       ├── experiments/
│       │   ├── config.py
│       │   ├── runner.py
│       │   ├── result_store.py
│       │   └── comparison.py
│       │
│       └── cli/
│           └── main.py
│
├── tests/
│
├── datasets/
│
└── results/
```

# Development Principle

The main test for whether the architecture is successful should be simple:

Adding a new strategy must not require modifying the experiment runner.

Adding a new benchmark must not require modifying the evaluation engine.

Adding a new metric must not require modifying dataset adapters.

Adding a new ingestion format must not require modifying chunkers.

If any of those statements becomes false, coupling has entered the architecture and should be corrected before adding more functionality.