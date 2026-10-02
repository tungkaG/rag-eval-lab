# File Processing

Placeholder for converting messy source files into normalized formats before
ingestion. No processors are implemented yet, and none are required by the
current pipeline.

## Responsibility

- File processing converts source files (e.g. PDF, DOCX, XLSX, HTML, scanned
  images) into normalized formats such as Markdown or JSONL.
- Ingestion (`rag_eval_lab.ingestion`) then loads those normalized files into
  `Document` objects.
- File processors must not create `Document` objects directly — that
  responsibility belongs to ingestion alone.

## Future processors

Later phases may add processors such as:

- PDF → Azure Document Intelligence → Markdown
- DOCX → processor → Markdown
- XLSX → processor → JSONL

Each future processor implements the `FileProcessor` protocol (`base.py`)
and registers itself in `file_processor_registry` (`registry.py`).
