# RAG Evaluation Lab

A modular RAG experimentation framework where chunking, retrieval, reranking,
and evaluation strategies can be swapped through configuration alone.

See [copilot.md](copilot.md) for the full phased architecture plan.

## Status

Phase 0 — core contracts (`models`, `protocols`, `registry`, `exceptions`) are in place.
No experiment can be run yet; that is intentional.

## Development

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```
