"""Loads a `.env` file from the repository root once, for generation provider API keys.

Provider API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, ...) are always read
from the process environment by each generator; this only gives that
environment a chance to be pre-populated from a local `.env` file so users
do not have to export keys in every shell session. Requires the optional
`python-dotenv` package (installed with the `generation` extra); silently
does nothing if it is not installed or no `.env` file exists. The file's
contents are never logged or stored anywhere -- only `os.environ` is
populated.
"""

from __future__ import annotations

from rag_eval_lab.core.paths import REPO_ROOT

_loaded = False


def load_env_file() -> None:
    """Loads `<repo root>/.env` into `os.environ`, if present. Safe to call repeatedly."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(REPO_ROOT / ".env", override=False)
